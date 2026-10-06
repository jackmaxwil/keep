"""Behavioral tests for immutable post-seed Sky bundle authority reuse."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "aws/glm52-gpu/scripts"
BUNDLE_BUILDER = SCRIPTS / "build_sky_campaign_bundle.sh"
APPROVAL_BUILDER = SCRIPTS / "build_gpu_spend_approval.py"
RUN_ID = "glm52-sky-20260726"
QUALIFICATION_CACHE_SHA256 = "9" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _clean_process_env() -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        **({"TMPDIR": os.environ["TMPDIR"]} if os.environ.get("TMPDIR") else {}),
    }


def _authority_fixture(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    approval_source = tmp_path / "prior" / "GPU_SPEND_APPROVAL.json"
    approval_source.parent.mkdir()
    approval = subprocess.run(
        [
            "python3",
            str(APPROVAL_BUILDER),
            "--ingested-at",
            "2026-07-24T01:05:00Z",
            "--output",
            str(approval_source),
        ],
        cwd=ROOT,
        env=_clean_process_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert approval.returncode == 0, approval.stderr

    repo_source = tmp_path / "prior" / "repo.tar.gz"
    repo_source.write_bytes(b"\x1f\x8bimmutable-seed-repository\x00with-exact-bytes\n")
    objects = tmp_path / "object-authorities.json"
    objects.write_text("[]\n")
    output = tmp_path / "post-seed-bundle"
    env = {
        **_clean_process_env(),
        "RUN_ID": RUN_ID,
        "BUCKET": "keep-glm52-us-west-2-246813579024",
        "MUST_START_BY": "2026-07-27T01:05:00Z",
        "IMAGE_ID": "ami-0123456789abcdef0",
        "OBJECT_AUTHORITIES_JSON": str(objects),
        "SOURCE_SNAPSHOT_PREFIX": "source-snapshot/",
        "SOURCE_SNAPSHOT_SHA256": "2" * 64,
        "NON_VQ_PREFIX": "non-vq-package/",
        "NON_VQ_PACKAGE_SHA256": "3" * 64,
        "TEICH_PACK_KEY": "teich-pack/pack.json",
        "TEICH_PACK_SHA256": "4" * 64,
        "FROZEN_PROMPT_PACK_KEY": "quality/frozen.json",
        "FROZEN_PROMPT_PACK_SHA256": "5" * 64,
        "TRAINING_BASELINE_PREFIX": "training-baseline/",
        "TRAINING_BASELINE_SHA256": "6" * 64,
        "QUALIFICATION_CACHE_PREFIX": (
            f"qualification-cache/seeds/{RUN_ID}/{QUALIFICATION_CACHE_SHA256}/"
        ),
        "QUALIFICATION_CACHE_MANIFEST_SHA256": QUALIFICATION_CACHE_SHA256,
        "SUBMISSION_ID": "post-seed",
        "OUTPUT_DIR": str(output),
        "APPROVAL_SOURCE": str(approval_source),
        "EXPECTED_APPROVAL_SHA256": _sha256(approval_source),
        "REPO_TAR_SOURCE": str(repo_source),
        "EXPECTED_REPO_TAR_SHA256": _sha256(repo_source),
    }
    return env, approval_source, repo_source, output


def _run_bundle(
    env: dict[str, str],
    *,
    remove: tuple[str, ...] = (),
    update: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_env = {**env, **(update or {})}
    for name in remove:
        process_env.pop(name, None)
    return subprocess.run(
        [str(BUNDLE_BUILDER)],
        cwd=ROOT,
        env=process_env,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_no_copied_authority_or_temp(output: Path, name: str) -> None:
    assert not (output / name).exists()
    if output.exists() and output.is_dir():
        assert not tuple(output.glob(f".{name}.*.tmp"))


def test_reused_authorities_remain_byte_identical_and_bind_the_bundle(
    tmp_path: Path,
) -> None:
    env, approval_source, repo_source, output = _authority_fixture(tmp_path)

    result = _run_bundle(env)

    assert result.returncode == 0, result.stderr
    approval_output = output / "GPU_SPEND_APPROVAL.json"
    repo_output = output / "repo.tar.gz"
    assert approval_output.read_bytes() == approval_source.read_bytes()
    assert repo_output.read_bytes() == repo_source.read_bytes()

    approval_sha = _sha256(approval_source)
    repo_sha = _sha256(repo_source)
    approval_key = (
        f"campaigns/{RUN_ID}/authorities/GPU_SPEND_APPROVAL-{approval_sha}.json"
    )
    repo_key = f"campaigns/{RUN_ID}/repository/keep-{repo_sha}.tar.gz"
    descriptor = json.loads((output / "campaign-descriptor-v2.json").read_bytes())
    assert descriptor["approval_sha256"] == approval_sha
    assert descriptor["repo_tar_sha256"] == repo_sha
    assert descriptor["approval_key"] == approval_key
    assert descriptor["repo_tar_key"] == repo_key
    descriptor_body = dict(descriptor)
    descriptor_body_sha256 = descriptor_body.pop("descriptor_body_sha256")
    assert descriptor_body_sha256 == _canonical_sha256(descriptor_body)

    manifest = json.loads((output / "bundle-manifest-v1.json").read_bytes())
    staged_files = {item["role"]: item for item in manifest["files"]}
    assert staged_files["approval"] == {
        "key": approval_key,
        "local_name": "GPU_SPEND_APPROVAL.json",
        "role": "approval",
        "sha256": approval_sha,
        "size": approval_source.stat().st_size,
        "stage_order": 20,
    }
    assert staged_files["repository_tar"] == {
        "key": repo_key,
        "local_name": "repo.tar.gz",
        "role": "repository_tar",
        "sha256": repo_sha,
        "size": repo_source.stat().st_size,
        "stage_order": 10,
    }
    assert manifest["descriptor_body_sha256"] == descriptor["descriptor_body_sha256"]
    manifest_body = dict(manifest)
    manifest_body_sha256 = manifest_body.pop("bundle_manifest_body_sha256")
    assert manifest_body_sha256 == _canonical_sha256(manifest_body)

    inventory = json.loads((output / "artifact-inventory-v1.json").read_bytes())
    inventory_objects = {item["kind"]: item for item in inventory["objects"]}
    assert inventory_objects["approval"]["key"] == approval_key
    assert inventory_objects["approval"]["sha256"] == approval_sha
    assert inventory_objects["approval"]["size"] == approval_source.stat().st_size
    assert inventory_objects["repository_tar"]["key"] == repo_key
    assert inventory_objects["repository_tar"]["sha256"] == repo_sha
    assert inventory_objects["repository_tar"]["size"] == repo_source.stat().st_size
    inventory_sha = _sha256(output / "artifact-inventory-v1.json")
    assert descriptor["artifacts"]["artifact_inventory_sha256"] == inventory_sha
    assert (
        descriptor["artifacts"]["artifact_inventory_key"]
        == f"campaigns/{RUN_ID}/inventories/artifact-inventory-{inventory_sha}.json"
    )


def test_reused_approval_with_unknown_field_is_rejected_not_normalized(
    tmp_path: Path,
) -> None:
    env, approval_source, _, output = _authority_fixture(tmp_path)
    approval = json.loads(approval_source.read_bytes())
    approval["unreviewed_authority"] = True
    approval_source.write_text(json.dumps(approval, sort_keys=True) + "\n")
    env["EXPECTED_APPROVAL_SHA256"] = _sha256(approval_source)

    result = _run_bundle(env)

    assert result.returncode != 0
    assert "GPU spend approval schema mismatch" in result.stderr
    assert (
        output / "GPU_SPEND_APPROVAL.json"
    ).read_bytes() == approval_source.read_bytes()
    assert not (output / "campaign-descriptor-v2.json").exists()
    assert not (output / "bundle-manifest-v1.json").exists()


def test_repository_reuse_can_pair_with_the_unchanged_approval_build_path(
    tmp_path: Path,
) -> None:
    env, approval_source, repo_source, output = _authority_fixture(tmp_path)

    result = _run_bundle(
        env,
        remove=("APPROVAL_SOURCE", "EXPECTED_APPROVAL_SHA256"),
        update={"APPROVAL_INGESTED_AT": "2026-07-24T01:05:00Z"},
    )

    assert result.returncode == 0, result.stderr
    assert (output / "repo.tar.gz").read_bytes() == repo_source.read_bytes()
    assert (
        output / "GPU_SPEND_APPROVAL.json"
    ).read_bytes() == approval_source.read_bytes()


@pytest.mark.parametrize(
    ("authority", "expected_name", "mismatch"),
    (
        ("approval", "EXPECTED_APPROVAL_SHA256", "approval source SHA-256"),
        ("repository tar", "EXPECTED_REPO_TAR_SHA256", "repository tar source SHA-256"),
    ),
)
def test_reused_authority_drift_is_rejected_without_publishing_drifted_authority(
    tmp_path: Path,
    authority: str,
    expected_name: str,
    mismatch: str,
) -> None:
    env, approval_source, repo_source, output = _authority_fixture(tmp_path)
    source = approval_source if authority == "approval" else repo_source
    source.write_bytes(source.read_bytes() + b"drift")
    original = source.read_bytes()

    result = _run_bundle(env)

    assert result.returncode != 0
    assert mismatch in result.stderr
    assert source.read_bytes() == original
    output_name = (
        "GPU_SPEND_APPROVAL.json" if authority == "approval" else "repo.tar.gz"
    )
    _assert_no_copied_authority_or_temp(output, output_name)
    assert env[expected_name] != _sha256(source)


@pytest.mark.parametrize(
    "rebuild_input",
    ("APPROVAL_INGESTED_AT", "SLACK_PERMALINK"),
)
def test_approval_reuse_rejects_conflicting_rebuild_inputs(
    tmp_path: Path,
    rebuild_input: str,
) -> None:
    env, approval_source, _, output = _authority_fixture(tmp_path)

    result = _run_bundle(
        env,
        update={
            rebuild_input: (
                "2026-07-24T01:05:00Z"
                if rebuild_input == "APPROVAL_INGESTED_AT"
                else "https://example.slack.com/archives/C123/p456"
            )
        },
    )

    assert result.returncode != 0
    assert "approval reuse conflicts with approval rebuild inputs" in result.stderr
    assert approval_source.is_file()
    assert not output.exists()


@pytest.mark.parametrize(
    ("remove", "message"),
    (
        (
            ("EXPECTED_APPROVAL_SHA256",),
            "APPROVAL_SOURCE requires EXPECTED_APPROVAL_SHA256",
        ),
        (
            ("APPROVAL_SOURCE",),
            "EXPECTED_APPROVAL_SHA256 requires APPROVAL_SOURCE",
        ),
        (
            ("EXPECTED_REPO_TAR_SHA256",),
            "REPO_TAR_SOURCE requires EXPECTED_REPO_TAR_SHA256",
        ),
        (
            ("REPO_TAR_SOURCE",),
            "EXPECTED_REPO_TAR_SHA256 requires REPO_TAR_SOURCE",
        ),
    ),
)
def test_partial_reuse_configuration_is_rejected_before_output(
    tmp_path: Path,
    remove: tuple[str, ...],
    message: str,
) -> None:
    env, _, _, output = _authority_fixture(tmp_path)

    result = _run_bundle(env, remove=remove)

    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("EXPECTED_APPROVAL_SHA256", "A" * 64),
        ("EXPECTED_APPROVAL_SHA256", "a" * 63),
        ("EXPECTED_REPO_TAR_SHA256", "g" * 64),
    ),
)
def test_noncanonical_or_unknown_sha_authority_is_rejected(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    env, _, _, output = _authority_fixture(tmp_path)

    result = _run_bundle(env, update={name: value})

    assert result.returncode != 0
    assert f"{name} must be a lowercase SHA-256" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("source_name", ("APPROVAL_SOURCE", "REPO_TAR_SOURCE"))
def test_output_source_alias_is_rejected_without_overwriting_source(
    tmp_path: Path,
    source_name: str,
) -> None:
    env, _, _, _ = _authority_fixture(tmp_path)
    source = Path(env[source_name])
    original = source.read_bytes()

    result = _run_bundle(env, update={"OUTPUT_DIR": str(source)})

    assert result.returncode != 0
    assert "OUTPUT_DIR must not alias an authority source" in result.stderr
    assert source.read_bytes() == original
    assert source.is_file()


@pytest.mark.parametrize("source_name", ("APPROVAL_SOURCE", "REPO_TAR_SOURCE"))
def test_reused_authority_source_must_not_be_a_symlink(
    tmp_path: Path,
    source_name: str,
) -> None:
    env, _, _, output = _authority_fixture(tmp_path)
    source = Path(env[source_name])
    link = source.with_name(f"{source.name}.link")
    link.symlink_to(source.name)

    result = _run_bundle(
        env,
        update={
            source_name: str(link),
            (
                "EXPECTED_APPROVAL_SHA256"
                if source_name == "APPROVAL_SOURCE"
                else "EXPECTED_REPO_TAR_SHA256"
            ): _sha256(source),
        },
    )

    assert result.returncode != 0
    assert "authority source must be a regular non-symlink file" in result.stderr
    assert not output.exists()
