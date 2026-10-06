"""Task 13 stage-scoped controller authority issuer contracts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import stat
import subprocess

import pytest

from glm52_enforcement.canonical import canonical_json_bytes, canonical_sha256
from glm52_enforcement.task13_controller_authority import (
    AUTHORITY_THREAT_MODEL,
    OPENSSH_SIGNING_NAMESPACE,
    ControllerAuthoritySigner,
    ControllerAuthorityIssuerError,
    issue_controller_execution_authority,
    issue_controller_operation_capability,
    verify_controller_execution_authority,
    verify_controller_operation_capability,
)
from glm52_task13_signer_support import (
    PRODUCTION_PINNED_SIGNER_FINGERPRINT,
    PRODUCTION_PINNED_SIGNER_PUBLIC_KEY,
    ephemeral_controller_authority_signer,
    patched_python_script_command,
)
from glm52_enforcement.task13_campaign_package import (
    canonical_campaign_package_bytes,
)
from test_glm52_task13_campaign_runner import (
    FiniteServices,
    MemoryJournal,
    _package,
)


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "aws/glm52-gpu/scripts/"
    "issue_glm52_task13_controller_authority.py"
)
COORDINATOR = (
    ROOT
    / "aws/glm52-gpu/scripts/"
    "glm52_task13_production_coordinator.py"
)
APPROVAL = (
    ROOT
    / "docs/superpowers/approvals/"
    "2026-07-28-glm52-campaign-owner-approvals.md"
)
COORDINATOR_SHA = hashlib.sha256(COORDINATOR.read_bytes()).hexdigest()
APPROVAL_SHA = hashlib.sha256(APPROVAL.read_bytes()).hexdigest()
FULL_RUN_WORK, SIGNING_PRIVATE_KEY, SIGNER = (
    ephemeral_controller_authority_signer()
)


def test_v2_boundary_is_pinned_and_does_not_claim_same_uid_protection() -> None:
    """Same-UID malicious code is explicitly outside this authority boundary."""

    assert AUTHORITY_THREAT_MODEL == (
        "Protects against accidental or direct protocol bypass and authority "
        "object tampering; malicious code running as the signing-key owner "
        "is outside this boundary."
    )
    assert OPENSSH_SIGNING_NAMESPACE == (
        "keep-glm52-task13-controller-authority-v2"
    )
    assert PRODUCTION_PINNED_SIGNER_PUBLIC_KEY == (
        "ssh-ed25519 "
        "AAAAC3NzaC1lZDI1NTE5AAAAIDGJndv7GTsFzrh/fGs82pm/W6MmUvOsnPlP28IKugex "
        "keep-glm52-task13-controller-v3"
    )
    assert PRODUCTION_PINNED_SIGNER_FINGERPRINT == (
        "SHA256:8nCALZDwL4X59iIu+znZj5QBKJCUi/c5tjkMR39Q5mM"
    )


def test_issuer_binds_pinned_owner_approval_and_execution_custody() -> None:
    """Break caught: stage authority is detached from consent and code custody."""

    package = _package()
    custody = package["production_retry_plan"]["execution_custody"]
    authority = issue_controller_execution_authority(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="launch",
        ttl_seconds=60,
        coordinator_executable_sha256=COORDINATOR_SHA,
        owner_approval_sha256=APPROVAL_SHA,
        now=1_900_000_000,
        random_bytes=lambda size: b"a" * size,
        signer=SIGNER,
    )

    assert authority.owner_approval_sha256 == APPROVAL_SHA
    assert authority.execution_custody_identity_sha256 == custody[
        "canonical_identity_sha256"
    ]
    assert authority.schema_version == 2
    assert authority.record_type == (
        "glm52_task13_controller_execution_authority_v2"
    )
    assert verify_controller_execution_authority(
        authority,
        now=1_900_000_001,
    ) is authority


def test_v2_signature_rejects_tampering_and_legacy_v1() -> None:
    """Break caught: a self-hash or edited grant is accepted as source authority."""

    package = _package()
    authority = issue_controller_execution_authority(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="launch",
        ttl_seconds=60,
        coordinator_executable_sha256=COORDINATOR_SHA,
        owner_approval_sha256=APPROVAL_SHA,
        now=1_900_000_000,
        random_bytes=lambda size: b"v" * size,
        signer=SIGNER,
    )
    tampered = asdict(authority)
    tampered["production_launch_approved"] = False
    with pytest.raises(
        ControllerAuthorityIssuerError,
        match="signature|identity",
    ):
        verify_controller_execution_authority(
            type(authority)(**tampered),
            now=1_900_000_001,
        )
    legacy = asdict(authority)
    legacy["schema_version"] = 1
    with pytest.raises(
        ControllerAuthorityIssuerError,
        match="legacy|schema",
    ):
        verify_controller_execution_authority(
            type(authority)(**legacy),
            now=1_900_000_001,
        )


def test_operation_capability_binds_request_operation_journal_and_ttl() -> None:
    """Break caught: a valid stage signature is replayed for a different request."""

    package = _package()
    authority = issue_controller_execution_authority(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="launch",
        ttl_seconds=120,
        coordinator_executable_sha256=COORDINATOR_SHA,
        owner_approval_sha256=APPROVAL_SHA,
        now=1_900_000_000,
        random_bytes=lambda size: b"s" * size,
        signer=SIGNER,
    )
    journal_ids = ["1" * 64, "2" * 64]
    capability = issue_controller_operation_capability(
        authority=authority,
        action="execute",
        operation_kind="guarded-launch",
        operation_id="launch:task10-workflow",
        request_identity_sha256="3" * 64,
        journal_store_path="/private/tmp/campaign-journal.jsonl",
        journal_store_identity_sha256="4" * 64,
        journal_record_identities=journal_ids,
        journal_state="POSSIBLY_SENT",
        prior_execute_capability_identity_sha256=None,
        ttl_seconds=60,
        now=1_900_000_001,
        random_bytes=lambda size: b"o" * size,
        signer=SIGNER,
    )
    assert capability.schema_version == 2
    assert capability.action == "execute"
    assert capability.operation_kind == "guarded-launch"
    assert capability.operation_id == "launch:task10-workflow"
    assert capability.request_identity_sha256 == "3" * 64
    assert capability.journal_record_identities == journal_ids
    assert capability.expires_at_epoch_seconds == 1_900_000_061
    assert verify_controller_operation_capability(
        capability,
        authority=authority,
        now=1_900_000_002,
    ) is capability

    edited = asdict(capability)
    edited["operation_id"] = "launch:foreign"
    with pytest.raises(
        ControllerAuthorityIssuerError,
        match="signature|identity",
    ):
        verify_controller_operation_capability(
            type(capability)(**edited),
            authority=authority,
            now=1_900_000_002,
        )


def test_reconcile_capability_requires_original_execute_custody() -> None:
    package = _package()
    authority = issue_controller_execution_authority(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="launch",
        ttl_seconds=120,
        coordinator_executable_sha256=COORDINATOR_SHA,
        owner_approval_sha256=APPROVAL_SHA,
        now=1_900_000_000,
        random_bytes=lambda size: b"r" * size,
        signer=SIGNER,
    )
    with pytest.raises(
        ControllerAuthorityIssuerError,
        match="prior execute",
    ):
        issue_controller_operation_capability(
            authority=authority,
            action="reconcile",
            operation_kind="guarded-launch",
            operation_id="launch:task10-workflow",
            request_identity_sha256="3" * 64,
            journal_store_path="/private/tmp/campaign-journal.jsonl",
            journal_store_identity_sha256="4" * 64,
            journal_record_identities=["1" * 64, "2" * 64],
            journal_state="POSSIBLY_SENT",
            prior_execute_capability_identity_sha256=None,
            ttl_seconds=60,
            now=1_900_000_001,
            random_bytes=lambda size: b"q" * size,
            signer=SIGNER,
        )


def test_signer_refuses_foreign_private_key_even_when_mode_is_0600(
    tmp_path: Path,
) -> None:
    """Break caught: the issuer accepts any caller-selected signing root."""

    tmp_path.chmod(0o700)
    foreign = (tmp_path / "controller-authority-ed25519").resolve()
    generated = subprocess.run(
        [
            "/usr/bin/ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            "foreign",
            "-f",
            str(foreign),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr
    foreign.chmod(0o600)
    foreign.with_suffix(".pub").chmod(0o600)
    with pytest.raises(
        ControllerAuthorityIssuerError,
        match="public key",
    ):
        ControllerAuthoritySigner(
            private_key_path=foreign,
            full_run_work=tmp_path.resolve(),
        )




@pytest.mark.parametrize(
    ("stage", "expected"),
    (
        ("deploy-disabled", (False, False, False, False)),
        ("qualification-cache-seed", (False, True, False, False)),
        ("h100-qualification", (False, False, True, False)),
        ("launch", (False, False, False, True)),
    ),
)
def test_issuer_grants_exactly_one_stage_family_and_self_hashes(
    stage: str,
    expected: tuple[bool, bool, bool, bool],
) -> None:
    package = _package()
    authority = issue_controller_execution_authority(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage=stage,
        ttl_seconds=60,
        coordinator_executable_sha256=COORDINATOR_SHA,
        owner_approval_sha256=APPROVAL_SHA,
        now=1_900_000_000,
        random_bytes=lambda size: bytes([len(stage)]) * size,
        signer=SIGNER,
    )
    assert (
        authority.change_set_execution_approved,
        authority.qualification_cache_seed_approved,
        authority.h100_qualification_approved,
        authority.production_launch_approved,
    ) == expected
    assert authority.expires_at_epoch_seconds == 1_900_000_060
    body = asdict(authority)
    body.pop("sshsig_base64")
    identity = body.pop("canonical_identity_sha256")
    assert identity == canonical_sha256(body)


@pytest.mark.parametrize("ttl", (59, 901, True))
def test_issuer_rejects_ttl_outside_closed_window(ttl: object) -> None:
    package = _package()
    with pytest.raises(ControllerAuthorityIssuerError, match="TTL"):
        issue_controller_execution_authority(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="launch",
            ttl_seconds=ttl,
            coordinator_executable_sha256=COORDINATOR_SHA,
            owner_approval_sha256=APPROVAL_SHA,
            signer=SIGNER,
        )


def test_issuer_rejects_prequalification_launch_and_artifact_drift() -> None:
    package = _package("PREQUALIFICATION")
    with pytest.raises(
        ControllerAuthorityIssuerError,
        match="prequalification",
    ):
        issue_controller_execution_authority(
            package=package,
            reviewed_artifacts=package["reviewed_artifacts"],
            stage="launch",
            ttl_seconds=60,
            coordinator_executable_sha256=COORDINATOR_SHA,
            owner_approval_sha256=APPROVAL_SHA,
            signer=SIGNER,
        )
    with pytest.raises(
        ControllerAuthorityIssuerError,
        match="reviewed artifacts",
    ):
        issue_controller_execution_authority(
            package=package,
            reviewed_artifacts=[],
            stage="h100-qualification",
            ttl_seconds=60,
            coordinator_executable_sha256=COORDINATOR_SHA,
            owner_approval_sha256=APPROVAL_SHA,
            signer=SIGNER,
        )


def test_issued_authority_is_accepted_by_runner_for_only_its_stage() -> None:
    from glm52_enforcement.task13_campaign_runner import run_campaign_stage

    package = _package()
    services = FiniteServices()
    authority = issue_controller_execution_authority(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="deploy-disabled",
        ttl_seconds=900,
        coordinator_executable_sha256=COORDINATOR_SHA,
        owner_approval_sha256=APPROVAL_SHA,
        signer=SIGNER,
    )
    result = run_campaign_stage(
        package=package,
        reviewed_artifacts=package["reviewed_artifacts"],
        stage="deploy-disabled",
        services=services,
        journal=MemoryJournal(),
        authority=authority,
    )
    assert result["status"] == "STAGE_COMMITTED"


def _run_cli(
    tmp_path: Path,
    *,
    output_name: str,
    approval_sha: str = APPROVAL_SHA,
    coordinator_sha: str = COORDINATOR_SHA,
) -> subprocess.CompletedProcess[str]:
    package = _package()
    package_path = (tmp_path / "package.json").resolve()
    artifacts_path = (tmp_path / "reviewed-artifacts.json").resolve()
    if not package_path.exists():
        package_path.write_bytes(canonical_campaign_package_bytes(package))
        artifacts_path.write_bytes(
            canonical_json_bytes(package["reviewed_artifacts"]) + b"\n"
        )
    arguments = [
        "--package",
        str(package_path),
        "--reviewed-artifacts",
        str(artifacts_path),
        "--stage",
        "launch",
        "--ttl-seconds",
        "60",
        "--expected-owner-approval-sha256",
        approval_sha,
        "--expected-coordinator-sha256",
        coordinator_sha,
        "--full-run-work",
        str(FULL_RUN_WORK),
        "--signing-private-key",
        str(SIGNING_PRIVATE_KEY),
        "--output",
        str((tmp_path / output_name).resolve()),
    ]
    return subprocess.run(
        patched_python_script_command(
            public_key_path=SIGNING_PRIVATE_KEY.with_suffix(".pub"),
            script_path=SCRIPT,
            arguments=arguments,
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_exact_reads_pins_inputs_writes_0600_and_uses_fresh_nonce(
    tmp_path: Path,
) -> None:
    first = _run_cli(tmp_path, output_name="authority-a.json")
    second = _run_cli(tmp_path, output_name="authority-b.json")
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    path_a = tmp_path / "authority-a.json"
    path_b = tmp_path / "authority-b.json"
    assert stat.S_IMODE(path_a.stat().st_mode) == 0o600
    original_a = path_a.read_bytes()
    authority_a = json.loads(path_a.read_bytes())
    authority_b = json.loads(path_b.read_bytes())
    assert (
        authority_a["controller_nonce_sha256"]
        != authority_b["controller_nonce_sha256"]
    )
    assert authority_a["coordinator_executable_sha256"] == COORDINATOR_SHA
    body = dict(authority_a)
    body.pop("sshsig_base64")
    identity = body.pop("canonical_identity_sha256")
    assert identity == canonical_sha256(body)

    overwrite = _run_cli(tmp_path, output_name="authority-a.json")
    assert overwrite.returncode != 0
    assert path_a.read_bytes() == original_a


@pytest.mark.parametrize(
    ("approval_sha", "coordinator_sha", "message"),
    (
        ("0" * 64, COORDINATOR_SHA, "owner approval identity drifted"),
        (APPROVAL_SHA, "0" * 64, "coordinator executable identity drifted"),
    ),
)
def test_cli_rejects_approval_or_executable_drift(
    tmp_path: Path,
    approval_sha: str,
    coordinator_sha: str,
    message: str,
) -> None:
    result = _run_cli(
        tmp_path,
        output_name="authority.json",
        approval_sha=approval_sha,
        coordinator_sha=coordinator_sha,
    )
    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / "authority.json").exists()
