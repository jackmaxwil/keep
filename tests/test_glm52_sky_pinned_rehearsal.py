from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "aws/glm52-gpu/scripts"
STRICT = SCRIPTS / "rehearse_staged_control_plane.sh"
WRAPPER = SCRIPTS / "rehearse_sky_control_plane.sh"
RUN_ID = "glm52-sky-pinned-test"
SUBMISSION_ID = "qualification-test"
BUCKET = "keep-glm52-us-west-2-246813579024"
DESCRIPTOR_KEY = (
    f"campaigns/{RUN_ID}/submissions/{SUBMISSION_ID}/campaign-descriptor-v2.json"
)
REPOSITORY_KEY = f"campaigns/{RUN_ID}/repository/repo.tar.gz"
DESCRIPTOR_URI = f"s3://{BUCKET}/{DESCRIPTOR_KEY}"
REPOSITORY_URI = f"s3://{BUCKET}/{REPOSITORY_KEY}"


def _load_campaign_api() -> object:
    path = ROOT / "src/mlx_vq/quality/glm52_sky_campaign.py"
    spec = importlib.util.spec_from_file_location(
        "_glm52_pinned_rehearsal_test_campaign",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CAMPAIGN_API = _load_campaign_api()


def _load_artifact_api() -> object:
    path = ROOT / "src/mlx_vq/quality/glm52_s3_artifact_audit.py"
    spec = importlib.util.spec_from_file_location(
        "_glm52_pinned_rehearsal_test_artifact",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ARTIFACT_API = _load_artifact_api()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _body_sha(value: dict[str, object], field: str) -> str:
    body = {key: item for key, item in value.items() if key != field}
    return _sha_bytes(_canonical(body)[:-1])


def _write_repository_tar(path: Path, *, wrapper_complete: bool = False) -> None:
    members = {"marker.txt": b"exact repository fixture\n"}
    if wrapper_complete:
        members = {
            name: b"fixture\n"
            for name in (
                "aws/glm52-gpu/skypilot/glm52-campaign.yaml",
                "aws/glm52-gpu/skypilot/bootstrap_campaign.sh",
                "aws/glm52-gpu/skypilot/prepare_nvme_storage.sh",
                "aws/glm52-gpu/skypilot/run_managed_campaign.sh",
                "aws/glm52-gpu/skypilot/keep-glm52-campaign.service",
                "aws/glm52-gpu/scripts/run_campaign.sh",
                "aws/glm52-gpu/scripts/manage_gpu_spend.py",
                "aws/glm52-gpu/scripts/verify_sky_terminal_state.py",
                "benchmarks/run_glm52_campaign.py",
            )
        }
    source = path.parent / "tar-source"
    source.mkdir()
    for name, raw in members.items():
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    with tarfile.open(path, "w:gz") as archive:
        for name in sorted(members):
            archive.add(source / name, arcname=name)


def _descriptor_bytes(
    *,
    repository_sha256: str,
    repository_key: str = REPOSITORY_KEY,
) -> bytes:
    approval = CAMPAIGN_API.build_gpu_spend_approval(
        ingested_at=datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc),
        slack_permalink=None,
    )
    descriptor = CAMPAIGN_API.build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc),
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
        worker_identity=("arn:aws:iam::246813579024:role/keep-glm52-gpu-worker"),
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket=BUCKET,
        jobs_bucket=BUCKET,
        repo_tar_key=repository_key,
        repo_tar_sha256=repository_sha256,
        campaign_descriptor_key=DESCRIPTOR_KEY,
        approval_key=(f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json"),
        approval_sha256=str(approval["approval_body_sha256"]),
        artifacts={
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
            "training_config_key": (f"campaigns/{RUN_ID}/authorities/training.json"),
            "training_config_sha256": "7" * 64,
            "artifact_inventory_key": (
                f"campaigns/{RUN_ID}/inventories/artifact-inventory-{'8' * 64}.json"
            ),
            "artifact_inventory_sha256": "8" * 64,
            "qualification_cache_prefix": (
                f"qualification-cache/seeds/{RUN_ID}/{'9' * 64}/"
            ),
            "qualification_cache_manifest_sha256": "9" * 64,
        },
    )
    return _canonical(descriptor)


def _remote_chain_fixture(
    tmp_path: Path,
    *,
    inventory_row_mutation: dict[str, object] | None = None,
) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    repository = tmp_path / "remote-repo.tar.gz"
    with tarfile.open(repository, "w:gz") as archive:
        for relative in (
            "src/mlx_vq/quality/glm52_sky_campaign.py",
            "src/mlx_vq/quality/glm52_s3_artifact_audit.py",
        ):
            archive.add(ROOT / relative, arcname=relative)
    repository_raw = repository.read_bytes()
    repository_sha = _sha_bytes(repository_raw)

    approval = CAMPAIGN_API.build_gpu_spend_approval(
        ingested_at=datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc),
        slack_permalink=None,
    )
    approval_raw = _canonical(approval)
    approval_key = f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL.json"
    training_raw = b'{"batch_size":1}\n'
    training_key = f"campaigns/{RUN_ID}/authorities/training.json"
    watchdog_raw = b"exact watchdog fixture\n"
    watchdog_key = f"campaigns/{RUN_ID}/lambda/watchdog.zip"
    inventory = ARTIFACT_API.build_s3_artifact_inventory(
        run_id=RUN_ID,
        bucket=BUCKET,
        objects=[
            {
                "key": REPOSITORY_KEY,
                "size": len(repository_raw),
                "sha256": repository_sha,
                "kind": "repository_tar",
                "safetensors": False,
                "run_scope": RUN_ID,
            },
            {
                "key": training_key,
                "size": len(training_raw),
                "sha256": _sha_bytes(training_raw),
                "kind": "training_configuration",
                "safetensors": False,
                "run_scope": RUN_ID,
                "version_id": "training_config-version-1",
            },
        ],
    )
    if inventory_row_mutation is not None:
        versioned_row = next(
            item
            for item in inventory["objects"]
            if item["key"] == training_key
        )
        versioned_row.update(inventory_row_mutation)
        inventory["inventory_body_sha256"] = _body_sha(
            inventory, "inventory_body_sha256"
        )
    inventory_raw = _canonical(inventory)
    inventory_key = (
        f"campaigns/{RUN_ID}/inventories/"
        f"artifact-inventory-{_sha_bytes(inventory_raw)}.json"
    )
    descriptor = CAMPAIGN_API.build_sky_campaign_descriptor(
        run_id=RUN_ID,
        must_start_by=datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc),
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
        worker_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-gpu-worker"
        ),
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket=BUCKET,
        jobs_bucket=BUCKET,
        repo_tar_key=REPOSITORY_KEY,
        repo_tar_sha256=repository_sha,
        campaign_descriptor_key=DESCRIPTOR_KEY,
        approval_key=approval_key,
        approval_sha256=_sha_bytes(approval_raw),
        artifacts={
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
            "training_config_key": training_key,
            "training_config_sha256": _sha_bytes(training_raw),
            "artifact_inventory_key": inventory_key,
            "artifact_inventory_sha256": _sha_bytes(inventory_raw),
            "qualification_cache_prefix": (
                f"qualification-cache/seeds/{RUN_ID}/{'9' * 64}/"
            ),
            "qualification_cache_manifest_sha256": "9" * 64,
        },
    )
    descriptor_raw = _canonical(descriptor)
    descriptor_path = tmp_path / "remote-descriptor.json"
    descriptor_path.write_bytes(descriptor_raw)
    role_values = (
        ("repository_tar", "repo.tar.gz", REPOSITORY_KEY, repository_raw, 10),
        ("approval", "GPU_SPEND_APPROVAL.json", approval_key, approval_raw, 20),
        ("training_config", "training.json", training_key, training_raw, 30),
        ("watchdog", "watchdog.zip", watchdog_key, watchdog_raw, 40),
        (
            "artifact_inventory",
            "artifact-inventory-v1.json",
            inventory_key,
            inventory_raw,
            50,
        ),
        (
            "descriptor",
            "campaign-descriptor-v2.json",
            DESCRIPTOR_KEY,
            descriptor_raw,
            60,
        ),
    )
    manifest_body = {
        "schema_version": 1,
        "record_type": "glm52_sky_campaign_bundle_v1",
        "run_id": RUN_ID,
        "bucket": BUCKET,
        "files": [
            {
                "local_name": local_name,
                "key": key,
                "role": role,
                "stage_order": order,
                "size": len(raw),
                "sha256": _sha_bytes(raw),
            }
            for role, local_name, key, raw, order in role_values
        ],
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
    }
    manifest = {
        **manifest_body,
        "bundle_manifest_body_sha256": _sha_bytes(
            _canonical(manifest_body)[:-1]
        ),
    }
    manifest_raw = _canonical(manifest)
    manifest_key = (
        f"campaigns/{RUN_ID}/submissions/{SUBMISSION_ID}/bundle-manifests/"
        f"{manifest['bundle_manifest_body_sha256']}/bundle-manifest-v1.json"
    )
    audit = {
        "schema_version": 1,
        "record_type": "glm52_s3_artifact_audit_v1",
        "audit_pass": True,
        "run_id": RUN_ID,
        "bucket": BUCKET,
        "inventory_body_sha256": inventory["inventory_body_sha256"],
        "object_count": 2,
        "object_bytes": len(repository_raw) + len(training_raw),
        "safetensors_object_count": 0,
        "safetensors_tensor_count": 0,
    }
    audit_raw = _canonical(audit)
    audit_key = (
        f"campaigns/{RUN_ID}/audits/"
        f"artifact-audit-{_sha_bytes(audit_raw)}.json"
    )
    versions = {
        role: f"{role}-version-1"
        for role, _local_name, _key, _raw, _order in role_values
    }
    versions["artifact_audit"] = "artifact-audit-version-1"
    readiness_body = {
        "schema_version": 2,
        "record_type": "glm52_staged_control_plane_ready_v2",
        "run_id": RUN_ID,
        "descriptor_key": DESCRIPTOR_KEY,
        "descriptor_sha256": _sha_bytes(descriptor_raw),
        "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
        "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
        "bundle_manifest_key": manifest_key,
        "bundle_manifest_file_sha256": _sha_bytes(manifest_raw),
        "bundle_manifest_body_sha256": manifest[
            "bundle_manifest_body_sha256"
        ],
        "bundle_manifest_version_id": "bundle-manifest-version-1",
        "staged_object_version_ids": versions,
        "artifact_audit_key": audit_key,
        "artifact_audit_sha256": _sha_bytes(audit_raw),
        "staged_at": "2026-07-26T12:00:00Z",
    }
    readiness = {
        **readiness_body,
        "ready_body_sha256": _sha_bytes(_canonical(readiness_body)[:-1]),
    }
    readiness_raw = _canonical(readiness)
    readiness_key = (
        f"campaigns/{RUN_ID}/submissions/{SUBMISSION_ID}/"
        "STAGED_CONTROL_PLANE_READY.json"
    )
    raw_by_role = {
        role: (key, raw)
        for role, _local_name, key, raw, _order in role_values
    }
    object_rows: dict[str, dict[str, str]] = {
        readiness_key: {
            "version": "readiness-version-1",
            "path": str(tmp_path / "readiness.json"),
        },
        manifest_key: {
            "version": "bundle-manifest-version-1",
            "path": str(tmp_path / "manifest.json"),
        },
        audit_key: {
            "version": "artifact-audit-version-1",
            "path": str(tmp_path / "audit.json"),
        },
    }
    (tmp_path / "readiness.json").write_bytes(readiness_raw)
    (tmp_path / "manifest.json").write_bytes(manifest_raw)
    (tmp_path / "audit.json").write_bytes(audit_raw)
    for role, version in versions.items():
        if role == "artifact_audit":
            continue
        key, raw = raw_by_role[role]
        path = tmp_path / f"{role}.remote"
        path.write_bytes(raw)
        object_rows[key] = {"version": version, "path": str(path)}
    object_map = tmp_path / "versioned-objects.json"
    object_map.write_text(json.dumps(object_rows, sort_keys=True))
    return {
        "descriptor_raw": descriptor_raw,
        "descriptor_sha": _sha_bytes(descriptor_raw),
        "repository_raw": repository_raw,
        "repository_sha": repository_sha,
        "readiness_key": readiness_key,
        "readiness_version": "readiness-version-1",
        "readiness": readiness,
        "inventory": inventory,
        "manifest_key": manifest_key,
        "object_map": object_map,
    }


def _write_fake_aws(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    log = tmp_path / "aws-calls.jsonl"
    aws = fake_bin / "aws"
    aws.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import sys

args = sys.argv[1:]
with Path(os.environ["FAKE_AWS_LOG"]).open("a") as handle:
    handle.write(json.dumps(args) + "\\n")

if args[:2] == ["sts", "get-caller-identity"]:
    print(json.dumps({
        "Account": "246813579024",
        "Arn": (
            "arn:aws:sts::246813579024:assumed-role/"
            "AWSReservedSSO_AdministratorAccess/fake"
        ),
        "UserId": "fake",
    }))
elif args[:2] == ["ssm", "get-parameter"]:
    sys.stdout.buffer.write(Path(os.environ["FAKE_DESCRIPTOR"]).read_bytes())
elif args[:2] == ["s3", "cp"]:
    uri = args[2]
    destination = Path(args[3])
    if uri == os.environ["FAKE_DESCRIPTOR_URI"]:
        source = Path(os.environ["FAKE_DESCRIPTOR"])
    elif uri == os.environ["FAKE_REPOSITORY_URI"]:
        source = Path(os.environ["FAKE_REPOSITORY"])
    else:
        raise SystemExit(88)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
elif args[:2] == ["s3api", "head-object"]:
    key = args[args.index("--key") + 1]
    objects = json.loads(Path(os.environ["FAKE_VERSIONED_OBJECTS"]).read_text())
    row = objects[key]
    raw = Path(row["path"]).read_bytes()
    print(json.dumps({
        "ContentLength": len(raw),
        "VersionId": row["version"],
    }))
elif args[:2] == ["s3api", "get-object"]:
    key = args[args.index("--key") + 1]
    version = args[args.index("--version-id") + 1]
    objects = json.loads(Path(os.environ["FAKE_VERSIONED_OBJECTS"]).read_text())
    row = objects[key]
    if version != row["version"]:
        raise SystemExit(87)
    destination = Path(args[-1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(row["path"], destination)
    returned_version = os.environ.get("FAKE_RETURNED_VERSION", version)
    print(json.dumps({
        "ContentLength": destination.stat().st_size,
        "VersionId": returned_version,
    }))
else:
    raise SystemExit(89)
"""
    )
    aws.chmod(0o755)
    return fake_bin, log


def _base_pinned_args(
    descriptor_sha256: str,
    repository_sha256: str,
) -> list[str]:
    return [
        "--pinned-authority",
        "--descriptor-uri",
        DESCRIPTOR_URI,
        "--descriptor-file-sha256",
        descriptor_sha256,
        "--repo-tar-uri",
        REPOSITORY_URI,
        "--repo-tar-file-sha256",
        repository_sha256,
    ]


def _run_strict(
    tmp_path: Path,
    args: list[str],
    *,
    descriptor_raw: bytes | None = None,
    repository_raw: bytes | None = None,
    environment: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    descriptor = tmp_path / "remote-descriptor.json"
    descriptor.write_bytes(descriptor_raw or b"{}\n")
    repository = tmp_path / "remote-repo.tar.gz"
    repository.write_bytes(repository_raw or b"not a repository")
    fake_bin, log = _write_fake_aws(tmp_path)
    work = tmp_path / "work"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "AWS_PROFILE": "keep-gpu",
        "REGION": "us-west-2",
        "WORK": str(work),
        "KEEP_REHEARSAL_WORK": "1",
        "REHEARSAL_EVIDENCE_OUTPUT": str(
            tmp_path / "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
        ),
        "FAKE_AWS_LOG": str(log),
        "FAKE_DESCRIPTOR": str(descriptor),
        "FAKE_REPOSITORY": str(repository),
        "FAKE_DESCRIPTOR_URI": DESCRIPTOR_URI,
        "FAKE_REPOSITORY_URI": REPOSITORY_URI,
        "FAKE_VERSIONED_OBJECTS": str(
            tmp_path / "absent-versioned-objects.json"
        ),
        **(environment or {}),
    }
    result = subprocess.run(
        ["bash", str(STRICT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    calls = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return result, calls, work


@pytest.mark.parametrize(
    "args_factory",
    (
        lambda base: [],
        lambda base: [base[0]],
        lambda base: base[:1] + base[3:],
        lambda base: base[:3] + base[5:],
        lambda base: base[:5] + base[7:],
        lambda base: base[:7],
        lambda base: [*base, "--legacy-active-parameter", "/keep/glm52"],
        lambda base: [*base, "--descriptor-uri", DESCRIPTOR_URI],
        lambda base: [*base, "--unknown"],
    ),
    ids=(
        "no-arguments",
        "partial",
        "omitted-descriptor-uri",
        "omitted-descriptor-sha",
        "omitted-repository-uri",
        "omitted-repository-sha",
        "mixed",
        "duplicate",
        "unknown",
    ),
)
def test_incomplete_or_ambiguous_modes_fail_before_guard_aws_and_workdir(
    tmp_path: Path,
    args_factory: object,
) -> None:
    base = _base_pinned_args("a" * 64, "b" * 64)
    args = args_factory(base)  # type: ignore[operator]

    result, calls, work = _run_strict(tmp_path, args)

    assert result.returncode == 64
    assert calls == []
    assert not work.exists()


@pytest.mark.parametrize(
    ("descriptor_uri", "repository_uri", "descriptor_sha", "repository_sha"),
    (
        (
            f"s3://{BUCKET}/campaigns/{RUN_ID}/submissions/../bad/"
            "campaign-descriptor-v2.json",
            REPOSITORY_URI,
            "a" * 64,
            "b" * 64,
        ),
        (
            f"s3://{BUCKET}/campaigns/{RUN_ID}/submissions/"
            f"{SUBMISSION_ID}//campaign-descriptor-v2.json",
            REPOSITORY_URI,
            "a" * 64,
            "b" * 64,
        ),
        (
            f"s3://Bad_Bucket/{DESCRIPTOR_KEY}",
            REPOSITORY_URI,
            "a" * 64,
            "b" * 64,
        ),
        (
            DESCRIPTOR_URI,
            f"s3://other-bucket/{REPOSITORY_KEY}",
            "a" * 64,
            "b" * 64,
        ),
        (
            DESCRIPTOR_URI,
            (f"s3://{BUCKET}/campaigns/foreign-run/repository/repo.tar.gz"),
            "a" * 64,
            "b" * 64,
        ),
        (
            DESCRIPTOR_URI,
            f"s3://{BUCKET}/campaigns/{RUN_ID}/foreign/repo.tar.gz",
            "a" * 64,
            "b" * 64,
        ),
        (DESCRIPTOR_URI, REPOSITORY_URI, "A" * 64, "b" * 64),
        (DESCRIPTOR_URI, REPOSITORY_URI, "a" * 63, "b" * 64),
        (DESCRIPTOR_URI, REPOSITORY_URI, "a" * 64, "B" * 64),
    ),
)
def test_unsafe_or_foreign_pins_fail_before_guard_aws_and_workdir(
    tmp_path: Path,
    descriptor_uri: str,
    repository_uri: str,
    descriptor_sha: str,
    repository_sha: str,
) -> None:
    args = [
        "--pinned-authority",
        "--descriptor-uri",
        descriptor_uri,
        "--descriptor-file-sha256",
        descriptor_sha,
        "--repo-tar-uri",
        repository_uri,
        "--repo-tar-file-sha256",
        repository_sha,
    ]

    result, calls, work = _run_strict(tmp_path, args)

    assert result.returncode == 64
    assert calls == []
    assert not work.exists()


@pytest.mark.parametrize(
    "environment",
    (
        {"AWS_PROFILE": "default"},
        {"AWS_PROFILE": "keep-gpu", "REGION": "us-east-1"},
    ),
)
def test_wrong_profile_or_region_fails_before_aws_and_workdir(
    tmp_path: Path,
    environment: dict[str, str],
) -> None:
    args = _base_pinned_args("a" * 64, "b" * 64)

    result, calls, work = _run_strict(
        tmp_path,
        args,
        environment=environment,
    )

    assert result.returncode == 64
    assert calls == []
    assert not work.exists()


def test_pinned_mode_reads_exact_descriptor_first_and_never_ssm(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "fixture-repo.tar.gz"
    _write_repository_tar(repository)
    repository_raw = repository.read_bytes()
    repository_sha = _sha_bytes(repository_raw)
    descriptor_raw = _descriptor_bytes(repository_sha256=repository_sha)
    descriptor_sha = _sha_bytes(descriptor_raw)

    result, calls, _work = _run_strict(
        tmp_path / "run",
        _base_pinned_args(descriptor_sha, repository_sha),
        descriptor_raw=descriptor_raw,
        repository_raw=repository_raw,
    )

    assert result.returncode != 0
    assert not any(call[:2] == ["ssm", "get-parameter"] for call in calls)
    object_reads = [call for call in calls if call[:2] == ["s3", "cp"]]
    assert [call[2] for call in object_reads] == [DESCRIPTOR_URI]


def test_strict_rehearsal_fetches_v2_manifest_and_every_role_by_pinned_version(
    tmp_path: Path,
) -> None:
    authority = _remote_chain_fixture(tmp_path / "authority")
    result, calls, work = _run_strict(
        tmp_path / "run",
        _base_pinned_args(
            str(authority["descriptor_sha"]),
            str(authority["repository_sha"]),
        ),
        descriptor_raw=authority["descriptor_raw"],
        repository_raw=authority["repository_raw"],
        environment={
            "FAKE_VERSIONED_OBJECTS": str(authority["object_map"]),
        },
    )

    assert result.returncode != 0
    inventory = authority["inventory"]
    assert isinstance(inventory, dict)
    assert {
        frozenset(item)
        for item in inventory["objects"]
    } == {
        frozenset(
            {"key", "size", "sha256", "kind", "safetensors", "run_scope"}
        ),
        frozenset(
            {
                "key",
                "size",
                "sha256",
                "kind",
                "safetensors",
                "run_scope",
                "version_id",
            }
        ),
    }
    assert "artifact inventory" not in result.stdout + result.stderr
    assert not any(call[:2] == ["ssm", "get-parameter"] for call in calls)
    head_calls = [
        call for call in calls if call[:2] == ["s3api", "head-object"]
    ]
    assert len(head_calls) == 1
    assert head_calls[0][head_calls[0].index("--key") + 1] == (
        authority["readiness_key"]
    )
    exact_gets = [
        call for call in calls if call[:2] == ["s3api", "get-object"]
    ]
    expected_versions = {
        str(authority["readiness_key"]): str(authority["readiness_version"]),
        str(authority["manifest_key"]): str(
            authority["readiness"]["bundle_manifest_version_id"]
        ),
        str(authority["readiness"]["artifact_audit_key"]): str(
            authority["readiness"]["staged_object_version_ids"][
                "artifact_audit"
            ]
        ),
    }
    manifest = json.loads(
        Path(
            json.loads(Path(authority["object_map"]).read_text())[
                str(authority["manifest_key"])
            ]["path"]
        ).read_bytes()
    )
    expected_versions.update(
        {
            str(item["key"]): str(
                authority["readiness"]["staged_object_version_ids"][
                    str(item["role"])
                ]
            )
            for item in manifest["files"]
        }
    )
    assert {
        call[call.index("--key") + 1]: call[
            call.index("--version-id") + 1
        ]
        for call in exact_gets
    } == expected_versions
    assert not (
        tmp_path / "run" / "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
    ).exists()
    assert not (
        work / "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
    ).exists()

@pytest.mark.parametrize(
    ("inventory_row_mutation", "expected_error"),
    (
        ({"foreign_field": "forbidden"}, "object schema mismatch"),
        ({"version_id": None}, "object authority mismatch"),
        ({"version_id": ""}, "object authority mismatch"),
        ({"version_id": "null"}, "object authority mismatch"),
        ({"version_id": "None"}, "object authority mismatch"),
        ({"version_id": "latest"}, "object authority mismatch"),
        (
            {"version_id": "training_config-version-2"},
            "object authority mismatch",
        ),
        ({"version_id": " leading-space"}, "object authority mismatch"),
        ({"version_id": "opaque-\u2603"}, "object authority mismatch"),
    ),
    ids=(
        "foreign-field",
        "json-null-version",
        "empty-version",
        "null-sentinel-version",
        "none-sentinel-version",
        "latest-sentinel-version",
        "staged-version-mismatch",
        "whitespace-version",
        "non-ascii-version",
    ),
)
def test_strict_rehearsal_rejects_inventory_schema_or_version_mutation(
    tmp_path: Path,
    inventory_row_mutation: dict[str, object],
    expected_error: str,
) -> None:
    authority = _remote_chain_fixture(
        tmp_path / "authority",
        inventory_row_mutation=inventory_row_mutation,
    )
    result, _calls, work = _run_strict(
        tmp_path / "run",
        _base_pinned_args(
            str(authority["descriptor_sha"]),
            str(authority["repository_sha"]),
        ),
        descriptor_raw=authority["descriptor_raw"],
        repository_raw=authority["repository_raw"],
        environment={
            "FAKE_VERSIONED_OBJECTS": str(authority["object_map"]),
        },
    )

    assert result.returncode != 0
    assert expected_error in result.stderr
    assert not (
        work / "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
    ).exists()


def test_strict_rehearsal_rejects_returned_version_mismatch_before_extraction(
    tmp_path: Path,
) -> None:
    authority = _remote_chain_fixture(tmp_path / "authority")
    result, calls, work = _run_strict(
        tmp_path / "run",
        _base_pinned_args(
            str(authority["descriptor_sha"]),
            str(authority["repository_sha"]),
        ),
        descriptor_raw=authority["descriptor_raw"],
        repository_raw=authority["repository_raw"],
        environment={
            "FAKE_VERSIONED_OBJECTS": str(authority["object_map"]),
            "FAKE_RETURNED_VERSION": "wrong-returned-version",
        },
    )

    assert result.returncode != 0
    exact_gets = [
        call for call in calls if call[:2] == ["s3api", "get-object"]
    ]
    assert len(exact_gets) == 1
    assert exact_gets[0][exact_gets[0].index("--key") + 1] == (
        authority["readiness_key"]
    )
    assert not any((work / "repo").iterdir())
    assert not Path(
        tmp_path / "run" / "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
    ).exists()


@pytest.mark.parametrize(
    "mutation",
    ("v1", "mixed", "bad_manifest", "missing_pin", "fractional_time"),
)
def test_strict_rehearsal_rejects_invalid_v2_chain_without_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    authority = _remote_chain_fixture(tmp_path / "authority")
    object_map_path = Path(authority["object_map"])
    object_map = json.loads(object_map_path.read_text())
    if mutation in {"v1", "mixed", "bad_manifest", "fractional_time"}:
        readiness_path = Path(
            object_map[str(authority["readiness_key"])]["path"]
        )
        readiness = json.loads(readiness_path.read_bytes())
        if mutation == "v1":
            for field in (
                "bundle_manifest_key",
                "bundle_manifest_file_sha256",
                "bundle_manifest_version_id",
                "staged_object_version_ids",
            ):
                readiness.pop(field)
            readiness["schema_version"] = 1
            readiness["record_type"] = "glm52_staged_control_plane_ready_v1"
        elif mutation == "mixed":
            readiness["unexpected_v1_field"] = "forbidden"
        elif mutation == "bad_manifest":
            readiness["bundle_manifest_file_sha256"] = "f" * 64
        else:
            readiness["staged_at"] = "2026-07-26T12:00:00.123456Z"
        readiness["ready_body_sha256"] = _body_sha(
            readiness, "ready_body_sha256"
        )
        readiness_path.write_bytes(_canonical(readiness))
    else:
        object_map[str(authority["manifest_key"])]["version"] = (
            "different-current-version"
        )
        object_map_path.write_text(json.dumps(object_map, sort_keys=True))

    result, calls, work = _run_strict(
        tmp_path / "run",
        _base_pinned_args(
            str(authority["descriptor_sha"]),
            str(authority["repository_sha"]),
        ),
        descriptor_raw=authority["descriptor_raw"],
        repository_raw=authority["repository_raw"],
        environment={
            "FAKE_VERSIONED_OBJECTS": str(authority["object_map"]),
        },
    )

    assert result.returncode != 0
    assert any(call[:2] == ["s3api", "get-object"] for call in calls)
    assert not any((work / "repo").iterdir())
    assert not (
        tmp_path / "run" / "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
    ).exists()


@pytest.mark.parametrize(
    "unsafe",
    (
        "campaigns/run/watchdog\npayload.zip",
        "campaigns/run/watchdog\tpayload.zip",
        "campaigns/run/watchdog\x00payload.zip",
        "campaigns/run/watchdog\x7fpayload.zip",
        r"campaigns/run/watchdog\payload.zip",
    ),
)
def test_strict_rehearsal_rejects_unsafe_watchdog_key_before_role_get(
    tmp_path: Path,
    unsafe: str,
) -> None:
    authority = _remote_chain_fixture(tmp_path / "authority")
    object_map_path = Path(authority["object_map"])
    object_map = json.loads(object_map_path.read_text())
    original_manifest_key = str(authority["manifest_key"])
    manifest_path = Path(object_map[original_manifest_key]["path"])
    manifest = json.loads(manifest_path.read_bytes())
    watchdog = next(
        item for item in manifest["files"] if item["role"] == "watchdog"
    )
    watchdog["key"] = unsafe
    manifest["bundle_manifest_body_sha256"] = _body_sha(
        manifest,
        "bundle_manifest_body_sha256",
    )
    manifest_raw = _canonical(manifest)
    manifest_path.write_bytes(manifest_raw)
    new_manifest_key = (
        f"campaigns/{RUN_ID}/submissions/{SUBMISSION_ID}/bundle-manifests/"
        f"{manifest['bundle_manifest_body_sha256']}/bundle-manifest-v1.json"
    )
    object_map[new_manifest_key] = object_map.pop(original_manifest_key)
    readiness_path = Path(
        object_map[str(authority["readiness_key"])]["path"]
    )
    readiness = json.loads(readiness_path.read_bytes())
    readiness.update(
        {
            "bundle_manifest_key": new_manifest_key,
            "bundle_manifest_file_sha256": _sha_bytes(manifest_raw),
            "bundle_manifest_body_sha256": manifest[
                "bundle_manifest_body_sha256"
            ],
        }
    )
    readiness["ready_body_sha256"] = _body_sha(
        readiness,
        "ready_body_sha256",
    )
    readiness_path.write_bytes(_canonical(readiness))
    object_map_path.write_text(json.dumps(object_map, sort_keys=True))

    result, calls, work = _run_strict(
        tmp_path / "run",
        _base_pinned_args(
            str(authority["descriptor_sha"]),
            str(authority["repository_sha"]),
        ),
        descriptor_raw=authority["descriptor_raw"],
        repository_raw=authority["repository_raw"],
        environment={"FAKE_VERSIONED_OBJECTS": str(object_map_path)},
    )

    assert result.returncode != 0
    assert not any(
        call[:2] == ["s3api", "get-object"]
        and call[call.index("--key") + 1] == unsafe
        for call in calls
    )
    assert not any((work / "repo").iterdir())
    assert not (
        tmp_path / "run" / "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
    ).exists()


def test_descriptor_hash_drift_stops_before_repository_or_dependent_reads(
    tmp_path: Path,
) -> None:
    descriptor_raw = _descriptor_bytes(repository_sha256="b" * 64)

    result, calls, _work = _run_strict(
        tmp_path,
        _base_pinned_args("f" * 64, "b" * 64),
        descriptor_raw=descriptor_raw,
    )

    assert result.returncode != 0
    object_reads = [call for call in calls if call[:2] == ["s3", "cp"]]
    assert [call[2] for call in object_reads] == [DESCRIPTOR_URI]


def test_descriptor_repository_identity_drift_stops_before_repository_read(
    tmp_path: Path,
) -> None:
    descriptor_raw = _descriptor_bytes(
        repository_sha256="b" * 64,
        repository_key=f"campaigns/{RUN_ID}/repository/foreign.tar.gz",
    )

    result, calls, _work = _run_strict(
        tmp_path,
        _base_pinned_args(_sha_bytes(descriptor_raw), "b" * 64),
        descriptor_raw=descriptor_raw,
    )

    assert result.returncode != 0
    object_reads = [call for call in calls if call[:2] == ["s3", "cp"]]
    assert [call[2] for call in object_reads] == [DESCRIPTOR_URI]


def test_invalid_campaign_field_stops_before_repository_read(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "fixture-repo.tar.gz"
    _write_repository_tar(repository)
    repository_raw = repository.read_bytes()
    descriptor = json.loads(
        _descriptor_bytes(repository_sha256=_sha_bytes(repository_raw))
    )
    descriptor["region"] = "us-east-1"
    descriptor.pop("descriptor_body_sha256")
    descriptor["descriptor_body_sha256"] = _sha_bytes(
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    )
    descriptor_raw = _canonical(descriptor)

    result, calls, _work = _run_strict(
        tmp_path / "run",
        _base_pinned_args(
            _sha_bytes(descriptor_raw),
            _sha_bytes(repository_raw),
        ),
        descriptor_raw=descriptor_raw,
        repository_raw=repository_raw,
    )

    assert result.returncode != 0
    object_reads = [call for call in calls if call[:2] == ["s3", "cp"]]
    assert [call[2] for call in object_reads] == [DESCRIPTOR_URI]


def test_repository_hash_drift_stops_before_extraction_and_dependent_reads(
    tmp_path: Path,
) -> None:
    authority = _remote_chain_fixture(tmp_path / "authority")
    object_map_path = Path(authority["object_map"])
    object_map = json.loads(object_map_path.read_text())
    repository_row = object_map[REPOSITORY_KEY]
    wrong_repository = tmp_path / "wrong-repository.tar.gz"
    wrong_repository.write_bytes(b"wrong repository bytes")
    repository_row["path"] = str(wrong_repository)
    object_map_path.write_text(json.dumps(object_map, sort_keys=True))

    result, calls, work = _run_strict(
        tmp_path / "run",
        _base_pinned_args(
            str(authority["descriptor_sha"]),
            str(authority["repository_sha"]),
        ),
        descriptor_raw=authority["descriptor_raw"],
        repository_raw=authority["repository_raw"],
        environment={
            "FAKE_VERSIONED_OBJECTS": str(authority["object_map"]),
        },
    )

    assert result.returncode != 0
    object_reads = [call for call in calls if call[:2] == ["s3", "cp"]]
    assert [call[2] for call in object_reads] == [DESCRIPTOR_URI]
    exact_reads = [
        call for call in calls if call[:2] == ["s3api", "get-object"]
    ]
    assert any(
        call[call.index("--key") + 1] == REPOSITORY_KEY
        for call in exact_reads
    )
    assert not any((work / "repo").iterdir())


def test_legacy_ssm_read_requires_exact_explicit_flag(tmp_path: Path) -> None:
    descriptor_raw = _descriptor_bytes(repository_sha256="b" * 64)
    exact, calls, _work = _run_strict(
        tmp_path / "exact",
        ["--legacy-active-parameter", "/keep-glm52/campaign/active"],
        descriptor_raw=descriptor_raw,
    )

    assert exact.returncode != 0
    assert calls[0][:2] == ["sts", "get-caller-identity"]
    assert calls[1][:2] == ["ssm", "get-parameter"]
    assert calls[1][calls[1].index("--name") + 1] == ("/keep-glm52/campaign/active")

    rejected, rejected_calls, rejected_work = _run_strict(
        tmp_path / "rejected",
        ["--legacy-active-parameter", "/foreign/active"],
        descriptor_raw=descriptor_raw,
        environment={"CAMPAIGN_DESCRIPTOR_PARAMETER": "/keep-glm52/campaign/active"},
    )
    assert rejected.returncode == 64
    assert rejected_calls == []
    assert not rejected_work.exists()


def _copy_wrapper_harness(tmp_path: Path) -> tuple[Path, Path]:
    scripts = tmp_path / "aws/glm52-gpu/scripts"
    scripts.mkdir(parents=True)
    wrapper = scripts / WRAPPER.name
    shutil.copyfile(WRAPPER, wrapper)
    wrapper.chmod(0o755)

    module_source = ROOT / "src/mlx_vq/quality/glm52_sky_campaign.py"
    module_target = tmp_path / "src/mlx_vq/quality/glm52_sky_campaign.py"
    module_target.parent.mkdir(parents=True)
    shutil.copyfile(module_source, module_target)
    native_source = ROOT / "src/glm52_enforcement/glm52_sky_campaign.py"
    native_target = tmp_path / "src/glm52_enforcement/glm52_sky_campaign.py"
    native_target.parent.mkdir(parents=True)
    shutil.copyfile(native_source, native_target)

    guard = scripts / "assert_rnd_aws_account.sh"
    guard.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' '[\"guard\"]' >> \"$FAKE_STRICT_LOG\"\n"
    )
    guard.chmod(0o755)

    strict = scripts / "rehearse_staged_control_plane.sh"
    strict.write_text(
        """#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
with Path(os.environ["FAKE_STRICT_LOG"]).open("a") as handle:
    handle.write(json.dumps(args) + "\\n")
if args == ["--validate-evidence", os.environ["REHEARSAL_EVIDENCE_OUTPUT"]]:
    raise SystemExit(0)
if not args or args[0] != "--pinned-authority":
    raise SystemExit(79)
descriptor_path = Path(os.environ["FAKE_DESCRIPTOR"])
descriptor = json.loads(descriptor_path.read_bytes())
evidence = {
    "schema_version": 2,
    "record_type": "glm52_staged_control_plane_rehearsal_v2",
    "status": "passed_before_cuda_h100_boundary",
    "run_id": descriptor["run_id"],
    "campaign_identity_sha256": descriptor["campaign_identity_sha256"],
    "descriptor_key": descriptor["campaign_descriptor_key"],
    "descriptor_file_sha256": hashlib.sha256(
        descriptor_path.read_bytes()
    ).hexdigest(),
    "descriptor_body_sha256": descriptor["descriptor_body_sha256"],
    "repo_tar_sha256": descriptor["repo_tar_sha256"],
}
Path(os.environ["REHEARSAL_EVIDENCE_OUTPUT"]).write_bytes(
    json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode() + b"\\n"
)
"""
    )
    strict.chmod(0o755)
    return wrapper, strict


def test_wrapper_delegates_all_exact_unmodified_pins(tmp_path: Path) -> None:
    repository = tmp_path / "remote-repo.tar.gz"
    _write_repository_tar(repository, wrapper_complete=True)
    repository_sha = _sha_bytes(repository.read_bytes())
    descriptor = tmp_path / "remote-descriptor.json"
    descriptor.write_bytes(_descriptor_bytes(repository_sha256=repository_sha))
    descriptor_sha = _sha_bytes(descriptor.read_bytes())
    wrapper, _strict = _copy_wrapper_harness(tmp_path / "harness")
    fake_bin, aws_log = _write_fake_aws(tmp_path / "fake")
    strict_log = tmp_path / "strict-calls.jsonl"
    evidence = tmp_path / "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "AWS_PROFILE": "keep-gpu",
        "REGION": "us-west-2",
        "DESCRIPTOR_URI": DESCRIPTOR_URI,
        "EXPECTED_DESCRIPTOR_SHA256": descriptor_sha,
        "REPO_TAR_URI": REPOSITORY_URI,
        "EXPECTED_REPO_TAR_SHA256": repository_sha,
        "REHEARSAL_EVIDENCE_OUTPUT": str(evidence),
        "FAKE_AWS_LOG": str(aws_log),
        "FAKE_DESCRIPTOR": str(descriptor),
        "FAKE_REPOSITORY": str(repository),
        "FAKE_DESCRIPTOR_URI": DESCRIPTOR_URI,
        "FAKE_REPOSITORY_URI": REPOSITORY_URI,
        "FAKE_STRICT_LOG": str(strict_log),
    }

    result = subprocess.run(
        ["bash", str(wrapper)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in strict_log.read_text().splitlines()]
    assert calls[1] == [
        "--pinned-authority",
        "--descriptor-uri",
        DESCRIPTOR_URI,
        "--descriptor-file-sha256",
        descriptor_sha,
        "--repo-tar-uri",
        REPOSITORY_URI,
        "--repo-tar-file-sha256",
        repository_sha,
    ]
    assert calls[2] == ["--validate-evidence", str(evidence)]


def test_wrapper_descriptor_hash_drift_stops_before_repository_read(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "remote-repo.tar.gz"
    _write_repository_tar(repository, wrapper_complete=True)
    repository_sha = _sha_bytes(repository.read_bytes())
    descriptor = tmp_path / "remote-descriptor.json"
    descriptor.write_bytes(_descriptor_bytes(repository_sha256=repository_sha))
    wrapper, _strict = _copy_wrapper_harness(tmp_path / "harness")
    fake_bin, aws_log = _write_fake_aws(tmp_path / "fake")
    strict_log = tmp_path / "strict-calls.jsonl"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "AWS_PROFILE": "keep-gpu",
        "REGION": "us-west-2",
        "DESCRIPTOR_URI": DESCRIPTOR_URI,
        "EXPECTED_DESCRIPTOR_SHA256": "f" * 64,
        "REPO_TAR_URI": REPOSITORY_URI,
        "EXPECTED_REPO_TAR_SHA256": repository_sha,
        "REHEARSAL_EVIDENCE_OUTPUT": str(
            tmp_path / "GLM52_STAGED_CONTROL_PLANE_REHEARSAL.json"
        ),
        "FAKE_AWS_LOG": str(aws_log),
        "FAKE_DESCRIPTOR": str(descriptor),
        "FAKE_REPOSITORY": str(repository),
        "FAKE_DESCRIPTOR_URI": DESCRIPTOR_URI,
        "FAKE_REPOSITORY_URI": REPOSITORY_URI,
        "FAKE_STRICT_LOG": str(strict_log),
    }

    result = subprocess.run(
        ["bash", str(wrapper)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    calls = [json.loads(line) for line in aws_log.read_text().splitlines()]
    object_reads = [call for call in calls if call[:2] == ["s3", "cp"]]
    assert [call[2] for call in object_reads] == [DESCRIPTOR_URI]


@pytest.mark.parametrize(
    "args",
    (
        ["--write-evidence"],
        ["--validate-evidence", "missing-evidence.json"],
    ),
)
def test_evidence_utility_modes_never_call_aws(
    tmp_path: Path,
    args: list[str],
) -> None:
    result, calls, _work = _run_strict(tmp_path, args)

    assert result.returncode != 0
    assert calls == []


def test_finalization_utility_mode_never_calls_aws(tmp_path: Path) -> None:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' 'SKYPILOT-BOOTSTRAP-REHEARSAL-OK'\n"
    )
    bootstrap.chmod(0o755)
    result, calls, _work = _run_strict(
        tmp_path,
        [
            "--exercise-finalization",
            "--bootstrap-command",
            str(bootstrap),
            "--",
            "--output",
            str(tmp_path / "evidence.json"),
        ],
    )

    assert result.returncode != 0
    assert calls == []
