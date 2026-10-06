from __future__ import annotations

import hashlib
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_vq.recovery_campaign.verification import (
    AVAILABLE_VERIFICATION_PROFILES,
    VerificationProfile,
    VerificationProfileError,
    get_verification_authority,
    get_verification_profile,
    profile_fingerprint,
    render_verification_result,
    run_verification_profile,
)


EXPECTED_NAMES = (
    "glm52-recovery-mixed-rate",
    "glm52-recovery-loader",
    "glm52-recovery-candidate",
    "glm52-recovery-campaign",
)
PROTECTED = (
    ".keep-heavy-job.lock",
    "runs",
    "glm52-community-wow-section-handoff-20260710.md",
    "glm52-recovery-campaign-handoff-20260710.md",
)


def _write_review(root: Path, profile: VerificationProfile) -> None:
    authority = get_verification_authority(profile, root)
    record = root / profile.required_review_record
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps(
            {
                "decision": "approved",
                "profile": profile.name,
                "profile_sha256": profile_fingerprint(profile),
                "owned_content_sha256": authority.owned_content_sha256,
                "index_sha256": authority.index_sha256,
                "head_commit": authority.head_commit,
                "reviewer": profile.reviewer_authority,
                "schema_version": 1,
            },
            sort_keys=True,
        )
        + "\n"
    )


def _write_repo(root: Path, profile: VerificationProfile, *, review: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for relative in (
        *profile.owned_paths,
        *profile.runtime_paths,
        *profile.protected_paths,
    ):
        path = root / relative
        if relative == "runs":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture for {relative}\n")
    for argv in profile.cli_help_checks:
        if len(argv) == 3 and argv[0] == "python" and argv[1].endswith(".py"):
            path = root / argv[1]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        elif len(argv) == 4 and argv[:2] == ("python", "-m"):
            module = argv[2]
            candidates = (
                root / "src" / Path(*module.split(".")).with_suffix(".py"),
                root / "src" / Path(*module.split(".")) / "__main__.py",
            )
            candidates[0].parent.mkdir(parents=True, exist_ok=True)
            candidates[0].touch(exist_ok=True)
    if review:
        _write_review(root, profile)


def _profile(**changes: object) -> VerificationProfile:
    base = VerificationProfile(
        name="fixture-profile",
        owned_paths=("src/tool.py", "tests/test_tool.py"),
        pytest_modules=("tests/test_tool.py",),
        pytest_exclusions=(),
        compile_targets=("src/tool.py",),
        cli_help_checks=(("python", "src/tool.py", "--help"),),
        runtime_paths=(),
        protected_paths=PROTECTED,
        metal_allowed=False,
        heavy_lock_allowed=False,
        required_review_record=".superpowers/sdd/fixture-profile-review.json",
        reviewer_authority="independent-adversarial-review-v1",
    )
    return replace(base, **changes)


class RecordingRunner:
    def __init__(self, exit_codes: tuple[int, ...] = ()) -> None:
        self.calls: list[tuple[tuple[str, ...], Path]] = []
        self.exit_codes = list(exit_codes)

    def __call__(self, argv: tuple[str, ...], *, cwd: Path):
        self.calls.append((argv, cwd))
        exit_code = self.exit_codes.pop(0) if self.exit_codes else 0
        return SimpleNamespace(
            returncode=exit_code,
            stdout=f"stdout-{len(self.calls)}",
            stderr=f"stderr-{len(self.calls)}",
        )


def test_exact_profile_names_and_declarations_are_immutable() -> None:
    assert AVAILABLE_VERIFICATION_PROFILES == EXPECTED_NAMES

    profiles = tuple(get_verification_profile(name) for name in EXPECTED_NAMES)
    assert tuple(profile.name for profile in profiles) == EXPECTED_NAMES
    assert all(profile.protected_paths == PROTECTED for profile in profiles)
    assert all(profile.metal_allowed is False for profile in profiles)
    assert all(profile.heavy_lock_allowed is False for profile in profiles)
    assert all(profile.owned_paths for profile in profiles)
    assert all(profile.pytest_modules for profile in profiles)
    assert all(profile.compile_targets for profile in profiles)
    assert all(profile.cli_help_checks for profile in profiles)
    assert [profile.pytest_exclusions for profile in profiles] == [
        (),
        (),
        (),
        ("test_concurrent_creators_and_writers_form_one_chain_repeatedly",),
    ]
    assert all(profile.required_review_record.endswith("-review.json") for profile in profiles)
    with pytest.raises(FrozenInstanceError):
        profiles[0].name = "changed"  # type: ignore[misc]


def test_profile_declarations_are_grounded_in_the_repository() -> None:
    root = Path(__file__).parents[1]
    for name in EXPECTED_NAMES:
        profile = get_verification_profile(name)
        for relative in (
            *profile.owned_paths,
            *profile.pytest_modules,
            *profile.compile_targets,
            *profile.runtime_paths,
            *profile.protected_paths,
        ):
            assert (root / relative).exists(), (name, relative)


def test_runtime_support_declaration_must_exist_and_must_not_be_symlinked(
    tmp_path: Path,
) -> None:
    profile = _profile(runtime_paths=("support/data.json",))
    _write_repo(tmp_path, profile)
    runtime = tmp_path / "support/data.json"
    runtime.unlink()
    with pytest.raises(VerificationProfileError, match="does not exist"):
        run_verification_profile(
            profile,
            tmp_path,
            command_runner=RecordingRunner(),
            staged_path_provider=lambda _root: (),
        )

    runtime.symlink_to(tmp_path / "src/tool.py")
    with pytest.raises(VerificationProfileError, match="symlink"):
        run_verification_profile(
            profile,
            tmp_path,
            command_runner=RecordingRunner(),
            staged_path_provider=lambda _root: (),
        )


def test_runtime_support_change_after_copy_blocks_post_authentication(
    tmp_path: Path,
) -> None:
    profile = _profile(runtime_paths=("support/data.json",))
    _write_repo(tmp_path, profile)
    runtime = tmp_path / "support/data.json"

    class RuntimeChangingRunner(RecordingRunner):
        def __call__(self, argv: tuple[str, ...], *, cwd: Path):
            outcome = super().__call__(argv, cwd=cwd)
            if len(self.calls) == 1:
                runtime.write_text("changed after snapshot\n")
            return outcome

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=RuntimeChangingRunner(),
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert result.commit_ready is False
    authentication = [check for check in result.checks if check.kind == "authentication"]
    assert authentication[-1].status == "blocked"
    assert "declared paths changed" in (authentication[-1].reason or "")


def test_runtime_support_change_stales_review_authority(tmp_path: Path) -> None:
    profile = _profile(runtime_paths=("support/data.json",))
    _write_repo(tmp_path, profile)
    (tmp_path / "support/data.json").write_text("changed before verification\n")

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=RecordingRunner(),
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    review = next(check for check in result.checks if check.kind == "review")
    assert review.status == "blocked"
    assert "authority" in (review.reason or "")


def test_builtin_profiles_pass_strict_declaration_authentication() -> None:
    root = Path(__file__).parents[1]
    for name in EXPECTED_NAMES:
        runner = RecordingRunner()
        result = run_verification_profile(
            name,
            root,
            command_runner=runner,
            staged_path_provider=lambda _root: (),
            heavy_lock_state="held-known",
            metal_available=False,
        )
        assert runner.calls, name
        assert result.status == "blocked", "review record is intentionally absent"


def test_unknown_name_reports_exact_available_inventory() -> None:
    with pytest.raises(
        VerificationProfileError,
        match=(
            r"^unknown verification profile 'missing'; available: "
            + ", ".join(EXPECTED_NAMES)
            + r"$"
        ),
    ):
        get_verification_profile("missing")


def test_runner_uses_direct_argv_and_captures_every_command(tmp_path: Path) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    runner = RecordingRunner(exit_codes=(1, 0, 2))

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=runner,
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
        python_executable="/verified/python",
    )

    assert [call[0] for call in runner.calls] == [
        (
            "/verified/python",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_tool.py",
        ),
        ("/verified/python", "-m", "compileall", "-q", "src/tool.py"),
        ("/verified/python", "src/tool.py", "--help"),
    ]
    assert len({call[1] for call in runner.calls}) == 1
    execution_root = runner.calls[0][1]
    assert execution_root != tmp_path
    assert "keep-verification-" in execution_root.as_posix()
    command_checks = [check for check in result.checks if check.kind == "command"]
    assert [check.status for check in command_checks] == ["failed", "passed", "failed"]
    assert [check.exit_code for check in command_checks] == [1, 0, 2]
    assert [check.stdout for check in command_checks] == ["stdout-1", "stdout-2", "stdout-3"]
    assert [check.stderr for check in command_checks] == ["stderr-1", "stderr-2", "stderr-3"]
    assert result.commit_ready is False


@pytest.mark.parametrize(
    ("staged", "message"),
    (
        (("README.md",), "outside ownership fence"),
        ((".keep-heavy-job.lock",), "protected path"),
        (("runs/output.json",), "protected path"),
        (("../escape.py",), "unsafe staged path"),
    ),
)
def test_staged_scope_violations_block_commands(
    tmp_path: Path, staged: tuple[str, ...], message: str
) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    runner = RecordingRunner()

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=runner,
        staged_path_provider=lambda _root: staged,
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert runner.calls == []
    assert result.commit_ready is False
    assert message in "\n".join(result.scope_violations)
    assert any(check.status == "blocked" for check in result.checks)


def test_post_command_staged_scope_change_fails_closed(tmp_path: Path) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    inventories = iter(((), ("README.md",)))

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=RecordingRunner(),
        staged_path_provider=lambda _root: next(inventories),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert result.commit_ready is False
    assert "outside ownership fence" in "\n".join(result.scope_violations)


def test_default_staged_inventory_includes_protected_rename_source(tmp_path: Path) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "fixture@example.com"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Fixture"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    subprocess.run(
        ("git", "mv", ".keep-heavy-job.lock", "renamed-heavy-job.lock"),
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / ".keep-heavy-job.lock").write_text("replacement\n")
    subprocess.run(("git", "add", "-A"), cwd=tmp_path, check=True)
    runner = RecordingRunner()

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=runner,
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert runner.calls == []
    assert ".keep-heavy-job.lock" in result.staged_paths
    assert "staged protected path: .keep-heavy-job.lock" in result.scope_violations


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("owned_paths", ("/absolute.py",), "absolute"),
        ("owned_paths", ("../escape.py",), "parent traversal"),
        ("owned_paths", ("src/tool.py", "src/tool.py"), "duplicate"),
        ("pytest_modules", ("tests/missing.py",), "does not exist"),
        ("compile_targets", ("src/missing.py",), "does not exist"),
        ("cli_help_checks", (("python", "src/missing.py", "--help"),), "does not exist"),
    ),
)
def test_unsafe_duplicate_or_nonexistent_declarations_are_configuration_errors(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    profile = _profile(**{field: value})
    _write_repo(tmp_path, _profile(), review=False)

    with pytest.raises(VerificationProfileError, match=match):
        run_verification_profile(
            profile,
            tmp_path,
            command_runner=RecordingRunner(),
            staged_path_provider=lambda _root: (),
        )


def test_symlinked_declared_file_and_symlinked_repo_root_are_rejected(tmp_path: Path) -> None:
    profile = _profile()
    root = tmp_path / "repo"
    _write_repo(root, profile)
    target = root / "actual.py"
    target.write_text("pass\n")
    (root / "src/tool.py").unlink()
    (root / "src/tool.py").symlink_to(target)

    with pytest.raises(VerificationProfileError, match="symlink"):
        run_verification_profile(
            profile,
            root,
            command_runner=RecordingRunner(),
            staged_path_provider=lambda _root: (),
        )

    link = tmp_path / "repo-link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(VerificationProfileError, match="repository root.*symlink"):
        run_verification_profile(
            profile,
            link,
            command_runner=RecordingRunner(),
            staged_path_provider=lambda _root: (),
        )


def test_symlinked_intermediate_declaration_ancestor_is_rejected(tmp_path: Path) -> None:
    profile = _profile()
    root = tmp_path / "repo"
    _write_repo(root, profile)
    real_src = root / "real-src"
    real_src.mkdir()
    (real_src / "tool.py").write_text("pass\n")
    (root / "src/tool.py").unlink()
    (root / "src").rmdir()
    (root / "src").symlink_to(real_src, target_is_directory=True)

    with pytest.raises(VerificationProfileError, match="symlink"):
        run_verification_profile(
            profile,
            root,
            command_runner=RecordingRunner(),
            staged_path_provider=lambda _root: (),
        )


def test_intermediate_ancestor_swap_during_open_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    real_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "src" and dir_fd is not None and not swapped:
            swapped = True
            (tmp_path / "src").rename(tmp_path / "original-src")
            (tmp_path / "src").symlink_to(tmp_path / "original-src", target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr("mlx_vq.recovery_campaign.verification.os.open", swapping_open)
    with pytest.raises(VerificationProfileError, match="symlink|changed|authenticate"):
        run_verification_profile(
            profile,
            tmp_path,
            command_runner=RecordingRunner(),
            staged_path_provider=lambda _root: (),
        )


def test_post_command_reauthentication_detects_owned_file_replacement(
    tmp_path: Path,
) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)

    class ReplacingRunner(RecordingRunner):
        def __call__(self, argv: tuple[str, ...], *, cwd: Path):
            outcome = super().__call__(argv, cwd=cwd)
            if len(self.calls) == 1:
                target = cwd / "src/tool.py"
                replacement = cwd / "src/replacement.py"
                replacement.write_text("replacement\n")
                replacement.replace(target)
            return outcome

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=ReplacingRunner(),
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert result.commit_ready is False
    authentication = [check for check in result.checks if check.kind == "authentication"]
    assert authentication[-1].status == "blocked"
    assert "changed during verification" in (authentication[-1].reason or "")


def test_known_held_lock_and_forbidden_metal_are_policy_skips_not_failures(
    tmp_path: Path,
) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=RecordingRunner(),
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    policy = [check for check in result.checks if check.kind == "policy"]
    assert [(check.name, check.status) for check in policy] == [
        ("metal", "skipped-by-declared-policy"),
        ("heavy-lock", "skipped-by-declared-policy"),
    ]
    assert result.commit_ready is True


@pytest.mark.parametrize(
    ("changes", "kwargs", "reason"),
    (
        ({"metal_allowed": True}, {"metal_available": False}, "Metal unavailable"),
        (
            {"heavy_lock_allowed": True},
            {"heavy_lock_state": "held-known"},
            "heavy lock held",
        ),
        (
            {"heavy_lock_allowed": True},
            {"heavy_lock_state": "ambiguous"},
            "heavy lock ambiguous",
        ),
    ),
)
def test_required_unavailable_resources_block_commands(
    tmp_path: Path, changes: dict[str, object], kwargs: dict[str, object], reason: str
) -> None:
    profile = _profile(**changes)
    _write_repo(tmp_path, profile)
    runner = RecordingRunner()

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=runner,
        staged_path_provider=lambda _root: (),
        **kwargs,
    )

    assert runner.calls == []
    assert result.commit_ready is False
    assert reason in "\n".join(check.reason or "" for check in result.checks)


def test_missing_review_is_honest_absence_and_does_not_invalidate_profile(
    tmp_path: Path,
) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile, review=False)
    runner = RecordingRunner()

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=runner,
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert len(runner.calls) == 3
    assert result.commit_ready is False
    review = next(check for check in result.checks if check.kind == "review")
    assert review.status == "blocked"
    assert review.reason == "review record absent"
    assert review.evidence_sha256 is None


@pytest.mark.parametrize("decision", ("rejected", "pending"))
def test_nonapproved_review_keeps_readiness_false(tmp_path: Path, decision: str) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    record = tmp_path / profile.required_review_record
    payload = json.loads(record.read_text())
    payload["decision"] = decision
    record.write_text(json.dumps(payload, sort_keys=True) + "\n")

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=RecordingRunner(),
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    review = next(check for check in result.checks if check.kind == "review")
    assert review.status == "blocked"
    assert decision in (review.reason or "")
    assert review.evidence_sha256 == hashlib.sha256(record.read_bytes()).hexdigest()
    assert result.commit_ready is False


def test_review_must_bind_canonical_profile_fingerprint(tmp_path: Path) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    record = tmp_path / profile.required_review_record
    payload = json.loads(record.read_text())
    payload["profile_sha256"] = "0" * 64
    record.write_text(json.dumps(payload, sort_keys=True) + "\n")

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=RecordingRunner(),
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    review = next(check for check in result.checks if check.kind == "review")
    assert review.status == "blocked"
    assert "fingerprint" in (review.reason or "")
    assert result.commit_ready is False


def test_review_approval_becomes_stale_when_owned_content_changes(tmp_path: Path) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    (tmp_path / "src/tool.py").write_text("changed after approval\n")

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=RecordingRunner(),
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    review = next(check for check in result.checks if check.kind == "review")
    assert review.status == "blocked"
    assert "authority" in (review.reason or "")
    assert result.commit_ready is False


def test_review_requires_the_declared_reviewer_authority(tmp_path: Path) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    record = tmp_path / profile.required_review_record
    payload = json.loads(record.read_text())
    payload["reviewer"] = "self-approved"
    record.write_text(json.dumps(payload, sort_keys=True) + "\n")

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=RecordingRunner(),
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    review = next(check for check in result.checks if check.kind == "review")
    assert review.status == "blocked"
    assert "reviewer authority" in (review.reason or "")


def test_commands_execute_from_private_snapshot_not_transient_worktree(
    tmp_path: Path,
) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    observed: list[str] = []

    class TransientReplacementRunner(RecordingRunner):
        def __call__(self, argv: tuple[str, ...], *, cwd: Path):
            target = tmp_path / "src/tool.py"
            original = target.read_bytes()
            target.write_text("malicious transient replacement\n")
            try:
                observed.append((cwd / "src/tool.py").read_text())
            finally:
                target.write_bytes(original)
            return super().__call__(argv, cwd=cwd)

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=TransientReplacementRunner(),
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert observed == ["fixture for src/tool.py\n"] * 3
    assert result.commit_ready is True


def test_staged_owned_deletion_with_retained_worktree_file_blocks(tmp_path: Path) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "fixture@example.com"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Fixture"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    _write_review(tmp_path, profile)
    subprocess.run(("git", "rm", "--cached", "src/tool.py"), cwd=tmp_path, check=True)
    runner = RecordingRunner()

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=runner,
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert runner.calls == []
    assert result.commit_ready is False
    assert any("index" in (check.reason or "") for check in result.checks)


def test_staged_owned_blob_different_from_worktree_invalidates_review(tmp_path: Path) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "fixture@example.com"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Fixture"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    _write_review(tmp_path, profile)
    target = tmp_path / "src/tool.py"
    approved = target.read_bytes()
    target.write_text("malicious staged bytes\n")
    subprocess.run(("git", "add", "src/tool.py"), cwd=tmp_path, check=True)
    target.write_bytes(approved)

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=RecordingRunner(),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert result.commit_ready is False
    review = next(check for check in result.checks if check.kind == "review")
    assert review.status == "blocked"
    assert "authority" in (review.reason or "")


def test_head_change_during_commands_blocks_readiness(tmp_path: Path) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "fixture@example.com"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Fixture"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    _write_review(tmp_path, profile)

    class HeadChangingRunner(RecordingRunner):
        def __call__(self, argv: tuple[str, ...], *, cwd: Path):
            outcome = super().__call__(argv, cwd=cwd)
            if len(self.calls) == 1:
                subprocess.run(
                    ("git", "commit", "--allow-empty", "-qm", "head changed"),
                    cwd=tmp_path,
                    check=True,
                )
            return outcome

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=HeadChangingRunner(),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert result.commit_ready is False
    authentication = [check for check in result.checks if check.kind == "authentication"]
    assert authentication[-1].status == "blocked"
    assert "HEAD changed" in (authentication[-1].reason or "")


def test_hdiutil_retries_one_silent_create_failure(monkeypatch) -> None:
    verification_module = __import__(
        "mlx_vq.recovery_campaign.verification", fromlist=["_run_hdiutil"]
    )
    outcomes = iter(
        (
            SimpleNamespace(returncode=1, stderr="", stdout=""),
            SimpleNamespace(returncode=0, stderr="", stdout=""),
        )
    )
    calls: list[tuple[str, ...]] = []

    def run(argv, **_kwargs):
        calls.append(tuple(argv))
        return next(outcomes)

    monkeypatch.setattr(verification_module.sys, "platform", "darwin")
    monkeypatch.setattr(verification_module.shutil, "which", lambda _name: "/usr/bin/hdiutil")
    monkeypatch.setattr(verification_module.subprocess, "run", run)

    verification_module._run_hdiutil(("create", "-quiet", "snapshot.dmg"))

    assert calls == [
        ("hdiutil", "create", "-quiet", "snapshot.dmg"),
        ("hdiutil", "create", "-quiet", "snapshot.dmg"),
    ]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS read-only image contract")
def test_real_commands_cannot_replace_private_snapshot_bytes(tmp_path: Path) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile, review=False)
    (tmp_path / "src/tool.py").write_text(
        "import argparse\nargparse.ArgumentParser().parse_args()\n"
    )
    (tmp_path / "tests/test_tool.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n\n"
        "def test_snapshot_is_os_read_only():\n"
        "    target = Path(__file__).parents[1] / 'src/tool.py'\n"
        "    with pytest.raises(OSError):\n"
        "        target.write_text('substituted bytes\\n')\n"
    )
    _write_review(tmp_path, profile)

    result = run_verification_profile(
        profile,
        tmp_path,
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert result.commit_ready is True
    assert all(
        check.status == "passed"
        for check in result.checks
        if check.kind == "command"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS read-only image contract")
def test_unowned_tracked_export_substitution_blocks_before_commands(
    tmp_path: Path, monkeypatch
) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile, review=False)
    (tmp_path / "src/tool.py").write_text(
        "import argparse\nargparse.ArgumentParser().parse_args()\n"
    )
    (tmp_path / "src/helper.py").write_text("VALUE = 'trusted'\n")
    (tmp_path / "tests/test_tool.py").write_text(
        "import helper\n\n"
        "def test_unowned_helper():\n"
        "    assert helper.VALUE == 'trusted'\n"
    )
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "fixture@example.com"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Fixture"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    _write_review(tmp_path, profile)
    marker = tmp_path / "substituted-helper-executed"
    verification_module = __import__(
        "mlx_vq.recovery_campaign.verification", fromlist=["_run_hdiutil"]
    )
    original_hdiutil = verification_module._run_hdiutil
    substituted = False

    def substituting_hdiutil(argv: tuple[str, ...]) -> None:
        nonlocal substituted
        if argv and argv[0] == "create" and not substituted:
            substituted = True
            source = Path(argv[argv.index("-srcfolder") + 1])
            (source / "src/helper.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed')\n"
                "VALUE = 'trusted'\n"
            )
        original_hdiutil(argv)

    monkeypatch.setattr(
        "mlx_vq.recovery_campaign.verification._run_hdiutil",
        substituting_hdiutil,
    )

    result = run_verification_profile(
        profile,
        tmp_path,
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert marker.exists() is False
    assert result.commit_ready is False
    snapshot = next(check for check in result.checks if check.name == "execution-snapshot")
    assert snapshot.status == "blocked"
    assert "index blob" in (snapshot.reason or "")


def test_full_index_export_authenticates_symlink_and_exact_leaf_inventory(
    tmp_path: Path,
) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    (tmp_path / "src/unowned-link.py").symlink_to("tool.py")
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "fixture@example.com"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Fixture"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    verification_module = __import__(
        "mlx_vq.recovery_campaign.verification",
        fromlist=["_materialize_execution_snapshot"],
    )

    with verification_module._materialize_execution_snapshot(
        profile, tmp_path
    ) as snapshot:
        verification_module._authenticate_full_index_export(
            profile, snapshot, snapshot.root
        )
        link = snapshot.root / "src/unowned-link.py"
        link.unlink()
        link.symlink_to("different.py")
        with pytest.raises(VerificationProfileError, match="index blob mismatch"):
            verification_module._authenticate_full_index_export(
                profile, snapshot, snapshot.root
            )

    with verification_module._materialize_execution_snapshot(
        profile, tmp_path
    ) as snapshot:
        (snapshot.root / "src/unexpected.py").write_text("unexpected\n")
        with pytest.raises(VerificationProfileError, match="extra=.*unexpected.py"):
            verification_module._authenticate_full_index_export(
                profile, snapshot, snapshot.root
            )

    with verification_module._materialize_execution_snapshot(
        profile, tmp_path
    ) as snapshot:
        (snapshot.root / "src/unowned-link.py").unlink()
        with pytest.raises(VerificationProfileError, match="missing|does not exist"):
            verification_module._authenticate_full_index_export(
                profile, snapshot, snapshot.root
            )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS descriptor index contract")
def test_private_index_path_cannot_be_replaced_before_git_export(
    tmp_path: Path, monkeypatch
) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile, review=False)
    (tmp_path / "src/tool.py").write_text(
        "import argparse\nargparse.ArgumentParser().parse_args()\n"
    )
    helper = tmp_path / "src/helper.py"
    helper.write_text("VALUE = 'trusted'\n")
    (tmp_path / "tests/test_tool.py").write_text(
        "import helper\n\n"
        "def test_index_helper():\n"
        "    assert helper.VALUE == 'trusted'\n"
    )
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "fixture@example.com"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Fixture"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    _write_review(tmp_path, profile)
    source_index = tmp_path / ".git/index"
    original_index = source_index.read_bytes()
    marker = tmp_path / "substituted-index-executed"
    helper.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "VALUE = 'trusted'\n"
    )
    subprocess.run(("git", "add", "src/helper.py"), cwd=tmp_path, check=True)
    tampered_index = source_index.read_bytes()
    source_index.write_bytes(original_index)
    helper.write_text("VALUE = 'trusted'\n")

    verification_module = __import__(
        "mlx_vq.recovery_campaign.verification", fromlist=["_run_git"]
    )
    original_run_git = verification_module._run_git
    attempted = False
    replacement_blocked = False

    def replacing_private_index(
        root: Path, argv: tuple[str, ...], **kwargs
    ):
        nonlocal attempted, replacement_blocked
        environment = kwargs.get("env")
        if argv and argv[0] == "ls-files" and environment and not attempted:
            attempted = True
            private_index = Path(environment["GIT_INDEX_FILE"])
            replacement = private_index.with_name("replacement-index")
            try:
                replacement.write_bytes(tampered_index)
                replacement.replace(private_index)
            except OSError:
                replacement_blocked = True
                replacement.unlink(missing_ok=True)
        return original_run_git(root, argv, **kwargs)

    monkeypatch.setattr(
        "mlx_vq.recovery_campaign.verification._run_git",
        replacing_private_index,
    )

    result = run_verification_profile(
        profile,
        tmp_path,
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert attempted is True
    assert replacement_blocked is True
    assert marker.exists() is False
    assert result.commit_ready is True


def test_anonymous_index_mutation_by_git_operation_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "fixture@example.com"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Fixture"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    index_bytes = (tmp_path / ".git/index").read_bytes()
    verification_module = __import__(
        "mlx_vq.recovery_campaign.verification", fromlist=["_run_git"]
    )
    original_run_git = verification_module._run_git

    def mutating_git(root: Path, argv: tuple[str, ...], **kwargs):
        output = original_run_git(root, argv, **kwargs)
        descriptor = kwargs["pass_fds"][0]
        os.pwrite(descriptor, b"X", 0)
        return output

    monkeypatch.setattr(
        "mlx_vq.recovery_campaign.verification._run_git", mutating_git
    )
    with tempfile.TemporaryFile() as anonymous_index:
        anonymous_index.write(index_bytes)
        anonymous_index.flush()
        with pytest.raises(VerificationProfileError, match="mutated"):
            verification_module._run_git_with_anonymous_index(
                tmp_path,
                ("ls-files", "--stage", "-z"),
                descriptor=anonymous_index.fileno(),
                expected_bytes=index_bytes,
            )


def test_staged_owned_symlink_blocks_private_snapshot(tmp_path: Path) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "fixture@example.com"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Fixture"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    _write_review(tmp_path, profile)
    target = tmp_path / "src/tool.py"
    original = target.read_bytes()
    target.unlink()
    target.symlink_to("../tests/test_tool.py")
    subprocess.run(("git", "add", "src/tool.py"), cwd=tmp_path, check=True)
    target.unlink()
    target.write_bytes(original)
    runner = RecordingRunner()

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=runner,
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert runner.calls == []
    assert result.commit_ready is False
    snapshot = next(check for check in result.checks if check.name == "execution-snapshot")
    assert snapshot.status == "blocked"
    assert "symlink" in (snapshot.reason or "")


def test_exported_snapshot_bytes_must_match_captured_index_blob(
    tmp_path: Path, monkeypatch
) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "fixture@example.com"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Fixture"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    _write_review(tmp_path, profile)
    verification_module = __import__(
        "mlx_vq.recovery_campaign.verification", fromlist=["_run_git"]
    )
    original_run_git = verification_module._run_git

    def substituting_export(root: Path, argv: tuple[str, ...], **kwargs):
        output = original_run_git(root, argv, **kwargs)
        if argv and argv[0] == "checkout-index":
            prefix = next(
                item.removeprefix("--prefix=")
                for item in argv
                if item.startswith("--prefix=")
            )
            (Path(prefix) / "src/tool.py").write_text("substituted export\n")
        return output

    monkeypatch.setattr(
        "mlx_vq.recovery_campaign.verification._run_git", substituting_export
    )
    runner = RecordingRunner()

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=runner,
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert runner.calls == []
    assert result.commit_ready is False
    snapshot = next(check for check in result.checks if check.name == "execution-snapshot")
    assert "index blob" in (snapshot.reason or "")


def test_absolute_cli_module_declaration_is_profile_error(tmp_path: Path) -> None:
    profile = _profile(
        cli_help_checks=(("python", "-m", "/absolute/module", "--help"),)
    )
    _write_repo(tmp_path, _profile())

    with pytest.raises(VerificationProfileError, match="CLI-help module"):
        run_verification_profile(
            profile,
            tmp_path,
            command_runner=RecordingRunner(),
            staged_path_provider=lambda _root: (),
        )


def test_heavy_allowed_profile_retains_lock_through_commands(tmp_path: Path) -> None:
    profile = _profile(heavy_lock_allowed=True)
    _write_repo(tmp_path, profile)
    competing_acquisitions: list[bool] = []

    class CompetingRunner(RecordingRunner):
        def __call__(self, argv: tuple[str, ...], *, cwd: Path):
            descriptor = os.open(tmp_path / ".keep-heavy-job.lock", os.O_RDWR)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    competing_acquisitions.append(False)
                else:
                    competing_acquisitions.append(True)
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
            return super().__call__(argv, cwd=cwd)

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=CompetingRunner(),
        staged_path_provider=lambda _root: (),
        metal_available=False,
    )

    assert competing_acquisitions == [False, False, False]
    assert result.commit_ready is True


def test_private_snapshot_never_manufactures_a_free_heavy_lane(tmp_path: Path) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    snapshot_acquisitions: list[bool] = []

    class SnapshotLockRunner(RecordingRunner):
        def __call__(self, argv: tuple[str, ...], *, cwd: Path):
            descriptor = os.open(cwd / ".keep-heavy-job.lock", os.O_RDWR)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    snapshot_acquisitions.append(False)
                else:
                    snapshot_acquisitions.append(True)
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
            return super().__call__(argv, cwd=cwd)

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=SnapshotLockRunner(),
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert snapshot_acquisitions == [False, False, False]
    assert result.commit_ready is True


def test_heavy_lock_swap_restore_is_ambiguous_and_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    profile = _profile(heavy_lock_allowed=True)
    _write_repo(tmp_path, profile)
    lock_path = tmp_path / ".keep-heavy-job.lock"
    held_descriptor = os.open(lock_path, os.O_RDWR)
    fcntl.flock(held_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verification_module = __import__(
        "mlx_vq.recovery_campaign.verification", fromlist=["_open_heavy_lock"]
    )
    swapped = False

    def swapped_open(_root: Path, **_kwargs):
        nonlocal swapped
        assert not swapped
        swapped = True
        detached = tmp_path / "held-lock"
        lock_path.rename(detached)
        lock_path.write_bytes(b"")
        replacement_descriptor = os.open(lock_path, os.O_RDWR)
        replacement_stat = os.fstat(replacement_descriptor)
        lock_path.unlink()
        detached.rename(lock_path)
        return replacement_descriptor, (replacement_stat.st_dev, replacement_stat.st_ino)

    monkeypatch.setattr(
        "mlx_vq.recovery_campaign.verification._open_heavy_lock", swapped_open
    )
    runner = RecordingRunner()
    try:
        result = run_verification_profile(
            profile,
            tmp_path,
            command_runner=runner,
            staged_path_provider=lambda _root: (),
            metal_available=False,
        )
    finally:
        fcntl.flock(held_descriptor, fcntl.LOCK_UN)
        os.close(held_descriptor)

    assert runner.calls == []
    heavy = next(check for check in result.checks if check.name == "heavy-lock")
    assert heavy.status == "blocked"
    assert heavy.reason == "heavy lock ambiguous"


def test_review_parses_the_exact_authenticated_descriptor_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    record = tmp_path / profile.required_review_record
    rejected = json.loads(record.read_text())
    rejected["decision"] = "rejected"
    record.write_text(json.dumps(rejected, sort_keys=True) + "\n")
    verification_module = __import__(
        "mlx_vq.recovery_campaign.verification", fromlist=["_read_authenticated_path"]
    )
    original_read = verification_module._read_authenticated_path

    def swap_after_authentication(root_descriptor: int, relative, **kwargs):
        authenticated = original_read(root_descriptor, relative, **kwargs)
        if relative.as_posix() == profile.required_review_record:
            approved = dict(rejected)
            approved["decision"] = "approved"
            replacement = record.with_name("replacement.json")
            replacement.write_text(json.dumps(approved, sort_keys=True) + "\n")
            replacement.replace(record)
        return authenticated

    monkeypatch.setattr(
        "mlx_vq.recovery_campaign.verification._read_authenticated_path",
        swap_after_authentication,
    )

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=RecordingRunner(),
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    review = next(check for check in result.checks if check.kind == "review")
    assert review.status == "blocked"
    assert "rejected" in (review.reason or "")


def test_byte_identical_review_record_replacement_is_detected(tmp_path: Path) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    record = tmp_path / profile.required_review_record

    class ReplacingReviewRunner(RecordingRunner):
        def __call__(self, argv: tuple[str, ...], *, cwd: Path):
            outcome = super().__call__(argv, cwd=cwd)
            if len(self.calls) == 1:
                replacement = record.with_name("replacement.json")
                replacement.write_bytes(record.read_bytes())
                replacement.replace(record)
            return outcome

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=ReplacingReviewRunner(),
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    assert result.commit_ready is False
    authentication = [check for check in result.checks if check.kind == "authentication"]
    assert authentication[-1].status == "blocked"
    assert authentication[-1].reason == "review record changed during verification"


@pytest.mark.parametrize(
    ("payload", "reason"),
    (
        (
            '{"decision":"approved","extra":1,"profile":"fixture-profile",'
            '"profile_sha256":"' + "0" * 64 + '","reviewer":"r","schema_version":1}\n',
            "unknown fields",
        ),
        (
            '{"decision":"approved","decision":"rejected",'
            '"profile":"fixture-profile","profile_sha256":"' + "0" * 64
            + '","reviewer":"r","schema_version":1}\n',
            "duplicate field",
        ),
        (
            '{"decision":"approved","profile":"fixture-profile",'
            '"profile_sha256":"' + "0" * 64
            + '","reviewer":"r","schema_version":NaN}\n',
            "non-finite",
        ),
            (
                '{"decision":"approved","head_commit":"NO_GIT",'
                '"index_sha256":"' + "0" * 64 + '","owned_content_sha256":"'
                + "0" * 64 + '","profile":"fixture-profile",'
                '"profile_sha256":"' + "0" * 64
                + '","reviewer":"r","schema_version":true}\n',
            "schema_version",
        ),
    ),
)
def test_review_json_is_strict_and_exact(
    tmp_path: Path, payload: str, reason: str
) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    record = tmp_path / profile.required_review_record
    record.write_text(payload)

    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=RecordingRunner(),
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
    )

    review = next(check for check in result.checks if check.kind == "review")
    assert review.status == "blocked"
    assert reason in (review.reason or "")
    assert result.commit_ready is False


def test_json_and_terminal_rendering_are_deterministic(tmp_path: Path) -> None:
    profile = _profile()
    _write_repo(tmp_path, profile)
    result = run_verification_profile(
        profile,
        tmp_path,
        command_runner=RecordingRunner(),
        staged_path_provider=lambda _root: (),
        heavy_lock_state="held-known",
        metal_available=False,
        python_executable="python",
    )

    first = render_verification_result(result, as_json=True)
    second = render_verification_result(result, as_json=True)
    assert first == second
    assert first.endswith("\n")
    payload = json.loads(first)
    assert payload["profile"] == "fixture-profile"
    assert payload["commit_ready"] is True
    assert {check["status"] for check in payload["checks"]} == {
        "passed",
        "skipped-by-declared-policy",
    }
    terminal = render_verification_result(result, as_json=False)
    assert "Profile: fixture-profile" in terminal
    assert "Commit ready: yes" in terminal
