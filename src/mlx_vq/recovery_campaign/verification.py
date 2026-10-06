"""Strict, shell-free verification profiles for GLM-5.2 recovery work."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, Sequence


class VerificationProfileError(ValueError):
    """Raised when a verification profile or repository declaration is invalid."""


@dataclass(frozen=True)
class VerificationProfile:
    """One immutable verification ownership and policy declaration."""

    name: str
    owned_paths: tuple[str, ...]
    pytest_modules: tuple[str, ...]
    pytest_exclusions: tuple[str, ...]
    compile_targets: tuple[str, ...]
    cli_help_checks: tuple[tuple[str, ...], ...]
    runtime_paths: tuple[str, ...]
    protected_paths: tuple[str, ...]
    metal_allowed: bool
    heavy_lock_allowed: bool
    required_review_record: str
    reviewer_authority: str


@dataclass(frozen=True)
class VerificationCheckResult:
    """Machine-readable evidence for one authentication, policy, or command check."""

    name: str
    kind: str
    status: str
    argv: tuple[str, ...] = ()
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    reason: str | None = None
    evidence_sha256: str | None = None
    evidence_identity: str | None = None


@dataclass(frozen=True)
class VerificationRunResult:
    """Deterministic aggregate verification and commit-readiness result."""

    profile: str
    profile_sha256: str
    status: str
    commit_ready: bool
    staged_paths: tuple[str, ...]
    scope_violations: tuple[str, ...]
    checks: tuple[VerificationCheckResult, ...]


@dataclass(frozen=True)
class VerificationAuthority:
    """Canonical content, index, and commit identity reviewed for one profile."""

    profile_sha256: str
    owned_content_sha256: str
    index_sha256: str
    head_commit: str


_PROTECTED_PATHS = (
    ".keep-heavy-job.lock",
    "runs",
    "glm52-community-wow-section-handoff-20260710.md",
    "glm52-recovery-campaign-handoff-20260710.md",
)
_REVIEWER_AUTHORITY = "independent-adversarial-review-v1"
_GIT_EXECUTABLE = "/usr/bin/git"


def _declared_profile(
    name: str,
    *,
    owned_paths: tuple[str, ...],
    pytest_modules: tuple[str, ...],
    compile_targets: tuple[str, ...],
    cli_help_checks: tuple[tuple[str, ...], ...],
    runtime_paths: tuple[str, ...] = (),
    pytest_exclusions: tuple[str, ...] = (),
) -> VerificationProfile:
    return VerificationProfile(
        name=name,
        owned_paths=owned_paths,
        pytest_modules=pytest_modules,
        pytest_exclusions=pytest_exclusions,
        compile_targets=compile_targets,
        cli_help_checks=cli_help_checks,
        runtime_paths=runtime_paths,
        protected_paths=_PROTECTED_PATHS,
        metal_allowed=False,
        heavy_lock_allowed=False,
        required_review_record=f".superpowers/sdd/{name}-review.json",
        reviewer_authority=_REVIEWER_AUTHORITY,
    )


_PROFILES = (
    _declared_profile(
        "glm52-recovery-mixed-rate",
        owned_paths=(
            "benchmarks/run_glm52_recovery_wave1.py",
            "src/mlx_vq/convert/glm52_recovery_materialize.py",
            "src/mlx_vq/quant/rtn.py",
            "src/mlx_vq/validate/glm52_recovery_artifact.py",
            "tests/test_glm52_recovery_artifact_audit.py",
            "tests/test_glm52_recovery_wave1.py",
            "tests/test_rtn_quantization.py",
        ),
        pytest_modules=(
            "tests/test_glm52_recovery_artifact_audit.py",
            "tests/test_glm52_recovery_wave1.py",
            "tests/test_rtn_quantization.py",
        ),
        compile_targets=(
            "benchmarks/run_glm52_recovery_wave1.py",
            "src/mlx_vq/convert/glm52_recovery_materialize.py",
            "src/mlx_vq/quant/rtn.py",
            "src/mlx_vq/validate/glm52_recovery_artifact.py",
        ),
        cli_help_checks=(
            ("python", "benchmarks/run_glm52_recovery_wave1.py", "--help"),
            ("python", "-m", "mlx_vq.convert.glm52_recovery_materialize", "--help"),
        ),
    ),
    _declared_profile(
        "glm52-recovery-loader",
        owned_paths=(
            "src/mlx_vq/io/load.py",
            "src/mlx_vq/models/glm52_composite_loader.py",
            "src/mlx_vq/models/glm52_vq_adapter.py",
            "src/mlx_vq/validate/glm52_recovery_artifact.py",
            "tests/test_glm52_production_generation.py",
            "tests/test_glm52_recovery_artifact_audit.py",
            "tests/test_glm52_vq_adapter.py",
        ),
        pytest_modules=(
            "tests/test_glm52_production_generation.py",
            "tests/test_glm52_recovery_artifact_audit.py",
            "tests/test_glm52_vq_adapter.py",
        ),
        compile_targets=(
            "src/mlx_vq/io/load.py",
            "src/mlx_vq/models/glm52_composite_loader.py",
            "src/mlx_vq/models/glm52_vq_adapter.py",
            "src/mlx_vq/validate/glm52_recovery_artifact.py",
        ),
        cli_help_checks=(("python", "-m", "keep.cli", "build", "--help"),),
        runtime_paths=(
            "artifacts/quality/instruction_hf_dolly48_prompts.jsonl",
        ),
    ),
    _declared_profile(
        "glm52-recovery-candidate",
        owned_paths=(
            "src/mlx_vq/convert/glm52_recovery_materialize.py",
            "src/mlx_vq/io/load.py",
            "src/mlx_vq/models/glm52_composite_loader.py",
            "src/mlx_vq/models/glm52_vq_adapter.py",
            "src/mlx_vq/validate/glm52_recovery_artifact.py",
            "tests/test_glm52_composite_artifact_audit.py",
            "tests/test_glm52_production_generation.py",
            "tests/test_glm52_recovery_artifact_audit.py",
            "tests/test_glm52_recovery_wave1.py",
            "tests/test_glm52_vq_adapter.py",
        ),
        pytest_modules=(
            "tests/test_glm52_composite_artifact_audit.py",
            "tests/test_glm52_production_generation.py",
            "tests/test_glm52_recovery_artifact_audit.py",
            "tests/test_glm52_recovery_wave1.py",
            "tests/test_glm52_vq_adapter.py",
        ),
        compile_targets=(
            "src/mlx_vq/convert/glm52_recovery_materialize.py",
            "src/mlx_vq/io/load.py",
            "src/mlx_vq/models/glm52_composite_loader.py",
            "src/mlx_vq/models/glm52_vq_adapter.py",
            "src/mlx_vq/validate/glm52_recovery_artifact.py",
        ),
        cli_help_checks=(
            ("python", "-m", "mlx_vq.convert.glm52_recovery_materialize", "--help"),
            ("python", "-m", "keep.cli", "build", "--help"),
        ),
        runtime_paths=(
            "artifacts/quality/instruction_hf_dolly48_prompts.jsonl",
        ),
    ),
    _declared_profile(
        "glm52-recovery-campaign",
        owned_paths=(
            "recipes/glm52_recovery_campaign_v1_20260711.yaml",
            "src/mlx_vq/convert/glm52_recovery_materialize.py",
            "src/mlx_vq/recovery_campaign/__init__.py",
            "src/mlx_vq/recovery_campaign/cli.py",
            "src/mlx_vq/recovery_campaign/config.py",
            "src/mlx_vq/recovery_campaign/controller.py",
            "src/mlx_vq/recovery_campaign/launcher.py",
            "src/mlx_vq/recovery_campaign/ledger.py",
            "src/mlx_vq/recovery_campaign/models.py",
            "src/mlx_vq/recovery_campaign/observer.py",
            "src/mlx_vq/recovery_campaign/render.py",
            "src/mlx_vq/recovery_campaign/supervisor.py",
            "src/mlx_vq/recovery_campaign/verification.py",
            "tests/test_recovery_campaign_advance.py",
            "tests/test_recovery_campaign_cli.py",
            "tests/test_recovery_campaign_config.py",
            "tests/test_recovery_campaign_controller.py",
            "tests/test_recovery_campaign_hardening.py",
            "tests/test_recovery_campaign_ledger.py",
            "tests/test_recovery_campaign_observer.py",
            "tests/test_recovery_campaign_verification.py",
        ),
        pytest_modules=(
            "tests/test_recovery_campaign_advance.py",
            "tests/test_recovery_campaign_cli.py",
            "tests/test_recovery_campaign_config.py",
            "tests/test_recovery_campaign_controller.py",
            "tests/test_recovery_campaign_hardening.py",
            "tests/test_recovery_campaign_ledger.py",
            "tests/test_recovery_campaign_observer.py",
            "tests/test_recovery_campaign_verification.py",
        ),
        compile_targets=(
            "src/mlx_vq/convert/glm52_recovery_materialize.py",
            "src/mlx_vq/recovery_campaign/__init__.py",
            "src/mlx_vq/recovery_campaign/cli.py",
            "src/mlx_vq/recovery_campaign/config.py",
            "src/mlx_vq/recovery_campaign/controller.py",
            "src/mlx_vq/recovery_campaign/launcher.py",
            "src/mlx_vq/recovery_campaign/ledger.py",
            "src/mlx_vq/recovery_campaign/models.py",
            "src/mlx_vq/recovery_campaign/observer.py",
            "src/mlx_vq/recovery_campaign/render.py",
            "src/mlx_vq/recovery_campaign/supervisor.py",
            "src/mlx_vq/recovery_campaign/verification.py",
        ),
        cli_help_checks=(
            ("python", "-m", "keep.cli", "verify", "glm52-recovery-campaign", "--help"),
            (
                "python",
                "-m",
                "keep.cli",
                "recovery",
                "campaign",
                "verify",
                "glm52-recovery-campaign",
                "--help",
            ),
        ),
        runtime_paths=(
            "artifacts/quality/instruction_hf_dolly48_prompts.jsonl",
        ),
        pytest_exclusions=(
            "test_concurrent_creators_and_writers_form_one_chain_repeatedly",
        ),
    ),
)

AVAILABLE_VERIFICATION_PROFILES = tuple(profile.name for profile in _PROFILES)
_PROFILE_BY_NAME = {profile.name: profile for profile in _PROFILES}


def get_verification_profile(name: str) -> VerificationProfile:
    """Return one immutable named profile or reject it with the exact inventory."""

    try:
        return _PROFILE_BY_NAME[name]
    except KeyError as error:
        available = ", ".join(AVAILABLE_VERIFICATION_PROFILES)
        raise VerificationProfileError(
            f"unknown verification profile {name!r}; available: {available}"
        ) from error


def _profile_payload(profile: VerificationProfile) -> dict[str, object]:
    return {
        "cli_help_checks": [list(argv) for argv in profile.cli_help_checks],
        "compile_targets": list(profile.compile_targets),
        "heavy_lock_allowed": profile.heavy_lock_allowed,
        "metal_allowed": profile.metal_allowed,
        "name": profile.name,
        "owned_paths": list(profile.owned_paths),
        "protected_paths": list(profile.protected_paths),
        "pytest_exclusions": list(profile.pytest_exclusions),
        "pytest_modules": list(profile.pytest_modules),
        "required_review_record": profile.required_review_record,
        "reviewer_authority": profile.reviewer_authority,
        "runtime_paths": list(profile.runtime_paths),
    }


def profile_fingerprint(profile: VerificationProfile) -> str:
    """Hash every declared path, argv, and policy field canonically."""

    encoded = json.dumps(
        _profile_payload(profile),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _PathSnapshot:
    device: int
    inode: int
    mode: int
    size: int
    sha256: str | None


def _validate_relative_path(value: str, *, field: str) -> PurePosixPath:
    if type(value) is not str or not value:
        raise VerificationProfileError(f"{field} must contain nonempty strings")
    if "\x00" in value or "\\" in value:
        raise VerificationProfileError(f"{field} contains unsafe path {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise VerificationProfileError(f"{field} contains absolute path {value!r}")
    if value != path.as_posix() or any(part == ".." for part in path.parts):
        raise VerificationProfileError(f"{field} contains parent traversal {value!r}")
    if any(part in ("", ".") for part in path.parts):
        raise VerificationProfileError(f"{field} contains unsafe path {value!r}")
    return path


def _validate_unique(values: Sequence[str], *, field: str) -> None:
    if len(values) != len(set(values)):
        raise VerificationProfileError(f"{field} contains a duplicate declaration")


def _help_target(argv: tuple[str, ...]) -> str:
    if type(argv) is not tuple or not argv or any(type(arg) is not str or not arg for arg in argv):
        raise VerificationProfileError("cli_help_checks must contain nonempty argv tuples")
    if argv[0] != "python" or argv[-1] != "--help":
        raise VerificationProfileError(
            f"unsafe CLI-help declaration {argv!r}; expected direct python argv ending --help"
        )
    if len(argv) >= 4 and argv[1] == "-m":
        module = argv[2]
        if not module or any(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) is None
            for part in module.split(".")
        ):
            raise VerificationProfileError(f"unsafe CLI-help module {module!r}")
        return "module:" + module
    if len(argv) == 3 and argv[1].endswith(".py"):
        _validate_relative_path(argv[1], field="cli_help_checks")
        return argv[1]
    raise VerificationProfileError(f"unsafe CLI-help declaration {argv!r}")


def _module_target(root: Path, module: str) -> Path:
    relative = Path(*module.split("."))
    candidates = (
        root / "src" / relative.with_suffix(".py"),
        root / "src" / relative / "__main__.py",
    )
    for candidate in candidates:
        if candidate.exists() or candidate.is_symlink():
            return candidate
    raise VerificationProfileError(
        f"cli_help_checks module {module!r} does not exist under src"
    )


def _open_authenticated_root(root: Path) -> tuple[int, _PathSnapshot]:
    try:
        metadata = root.lstat()
    except OSError as error:
        raise VerificationProfileError(f"repository root is unavailable: {error}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise VerificationProfileError("repository root must not be a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise VerificationProfileError("repository root must be a directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError as error:
        raise VerificationProfileError(f"cannot authenticate repository root: {error}") from error
    current = os.fstat(descriptor)
    visible = root.lstat()
    if (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino) or (
        visible.st_dev,
        visible.st_ino,
    ) != (current.st_dev, current.st_ino):
        os.close(descriptor)
        raise VerificationProfileError("repository root changed during authentication")
    return descriptor, _PathSnapshot(
        current.st_dev, current.st_ino, current.st_mode, 0, None
    )


def _open_authenticated_parent(
    root_descriptor: int,
    relative: PurePosixPath,
    *,
    allow_missing: bool,
) -> tuple[int, str] | None:
    current = os.dup(root_descriptor)
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    for part in relative.parts[:-1]:
        try:
            before = os.stat(part, dir_fd=current, follow_symlinks=False)
        except FileNotFoundError:
            os.close(current)
            if allow_missing:
                return None
            raise VerificationProfileError(f"declared path {relative} does not exist")
        except OSError as error:
            os.close(current)
            raise VerificationProfileError(
                f"cannot inspect declared path {relative}: {error}"
            ) from error
        if stat.S_ISLNK(before.st_mode):
            os.close(current)
            raise VerificationProfileError(
                f"declared path {relative} has symlink component {part!r}"
            )
        if not stat.S_ISDIR(before.st_mode):
            os.close(current)
            raise VerificationProfileError(
                f"declared path {relative} has non-directory ancestor {part!r}"
            )
        try:
            child = os.open(part, directory_flags, dir_fd=current)
        except OSError as error:
            os.close(current)
            raise VerificationProfileError(
                f"cannot authenticate declared path {relative} ancestor {part!r}: {error}"
            ) from error
        try:
            opened = os.fstat(child)
            visible = os.stat(part, dir_fd=current, follow_symlinks=False)
        except OSError as error:
            os.close(child)
            os.close(current)
            raise VerificationProfileError(
                f"cannot reauthenticate declared path {relative} ancestor {part!r}: {error}"
            ) from error
        identity = (opened.st_dev, opened.st_ino)
        if identity != (before.st_dev, before.st_ino) or identity != (
            visible.st_dev,
            visible.st_ino,
        ):
            os.close(child)
            os.close(current)
            raise VerificationProfileError(
                f"declared path {relative} ancestor {part!r} changed during authentication"
            )
        os.close(current)
        current = child
    return current, relative.parts[-1]


@dataclass(frozen=True)
class _AuthenticatedPath:
    snapshot: _PathSnapshot
    content: bytes | None


def _read_authenticated_path(
    root_descriptor: int,
    relative: PurePosixPath,
    *,
    allow_missing: bool = False,
) -> _AuthenticatedPath | None:
    parent_and_leaf = _open_authenticated_parent(
        root_descriptor, relative, allow_missing=allow_missing
    )
    if parent_and_leaf is None:
        return None
    parent, leaf = parent_and_leaf
    try:
        try:
            before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            if allow_missing:
                return None
            raise VerificationProfileError(f"declared path {relative} does not exist")
        if stat.S_ISLNK(before.st_mode):
            raise VerificationProfileError(f"declared path {relative} is a symlink")
        is_directory = stat.S_ISDIR(before.st_mode)
        if not is_directory and not stat.S_ISREG(before.st_mode):
            raise VerificationProfileError(
                f"declared path {relative} is not a regular file or directory"
            )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        if is_directory:
            flags |= getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(leaf, flags, dir_fd=parent)
        except OSError as error:
            raise VerificationProfileError(
                f"cannot authenticate declared path {relative}: {error}"
            ) from error
        try:
            opened = os.fstat(descriptor)
            visible = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            identity = (opened.st_dev, opened.st_ino)
            if identity != (before.st_dev, before.st_ino) or identity != (
                visible.st_dev,
                visible.st_ino,
            ):
                raise VerificationProfileError(
                    f"declared path {relative} changed during authentication"
                )
            if is_directory:
                return _AuthenticatedPath(
                    _PathSnapshot(opened.st_dev, opened.st_ino, opened.st_mode, 0, None),
                    None,
                )
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
                digest.update(block)
            after = os.fstat(descriptor)
            visible_after = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) or (after.st_dev, after.st_ino) != (
                visible_after.st_dev,
                visible_after.st_ino,
            ):
                raise VerificationProfileError(
                    f"declared path {relative} changed while hashing"
                )
            return _AuthenticatedPath(
                _PathSnapshot(
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_size,
                    digest.hexdigest(),
                ),
                b"".join(chunks),
            )
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _declaration_paths(profile: VerificationProfile, root: Path) -> tuple[str, ...]:
    for field, values in (
        ("owned_paths", profile.owned_paths),
        ("pytest_modules", profile.pytest_modules),
        ("pytest_exclusions", profile.pytest_exclusions),
        ("compile_targets", profile.compile_targets),
        ("runtime_paths", profile.runtime_paths),
        ("protected_paths", profile.protected_paths),
    ):
        if type(values) is not tuple:
            raise VerificationProfileError(f"{field} must be an immutable tuple")
        _validate_unique(values, field=field)
        for value in values:
            _validate_relative_path(value, field=field)
    if type(profile.metal_allowed) is not bool or type(profile.heavy_lock_allowed) is not bool:
        raise VerificationProfileError("profile policy declarations must be booleans")
    if type(profile.reviewer_authority) is not str or not profile.reviewer_authority:
        raise VerificationProfileError("reviewer_authority must be a nonempty string")
    for exclusion in profile.pytest_exclusions:
        if re.fullmatch(r"test_[A-Za-z0-9_]+", exclusion) is None:
            raise VerificationProfileError(
                f"unsafe pytest exclusion declaration {exclusion!r}"
            )
    _validate_relative_path(profile.required_review_record, field="required_review_record")
    if not profile.pytest_modules or not profile.compile_targets or not profile.cli_help_checks:
        raise VerificationProfileError(
            "pytest, compile, and CLI-help declarations must be nonempty"
        )
    owned = set(profile.owned_paths)
    for value in (*profile.pytest_modules, *profile.compile_targets):
        if value not in owned:
            candidate = root.joinpath(*PurePosixPath(value).parts)
            if not candidate.exists() and not candidate.is_symlink():
                raise VerificationProfileError(f"declared path {value!r} does not exist")
            raise VerificationProfileError(f"declared check path {value!r} is outside owned_paths")
    if len(profile.cli_help_checks) != len(set(profile.cli_help_checks)):
        raise VerificationProfileError("cli_help_checks contains a duplicate declaration")
    help_paths: list[str] = []
    for argv in profile.cli_help_checks:
        target = _help_target(argv)
        if target.startswith("module:"):
            module_path = _module_target(root, target.removeprefix("module:"))
            help_paths.append(module_path.relative_to(root).as_posix())
        else:
            help_paths.append(target)
    return tuple(
        dict.fromkeys(
            (
                *profile.owned_paths,
                *profile.pytest_modules,
                *profile.compile_targets,
                *profile.runtime_paths,
                *profile.protected_paths,
                *help_paths,
            )
        )
    )


def _authenticate_declarations(
    profile: VerificationProfile, root: Path
) -> tuple[_PathSnapshot, Mapping[str, _PathSnapshot]]:
    root_descriptor, root_snapshot = _open_authenticated_root(root)
    try:
        paths = _declaration_paths(profile, root)
        snapshots: dict[str, _PathSnapshot] = {}
        for relative in paths:
            authenticated = _read_authenticated_path(
                root_descriptor, PurePosixPath(relative)
            )
            assert authenticated is not None
            snapshots[relative] = authenticated.snapshot
        return root_snapshot, snapshots
    finally:
        os.close(root_descriptor)


@dataclass(frozen=True)
class _ExecutionSnapshot:
    root: Path
    authority: VerificationAuthority
    source_index_path: Path | None
    source_index_sha256: str
    index_entries: tuple[tuple[str, str, str], ...] = ()
    object_format: str = "sha1"
    read_only: bool = False


def _run_git(
    root: Path,
    argv: tuple[str, ...],
    *,
    env: Mapping[str, str] | None = None,
    pass_fds: tuple[int, ...] = (),
) -> bytes:
    git_path = Path(_GIT_EXECUTABLE)
    metadata = git_path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise VerificationProfileError(
            f"pinned Git executable is not a regular file: {_GIT_EXECUTABLE}"
        )
    completed = subprocess.run(
        (_GIT_EXECUTABLE, *argv),
        cwd=root,
        check=False,
        capture_output=True,
        text=False,
        shell=False,
        env=dict(env) if env is not None else None,
        pass_fds=pass_fds,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise VerificationProfileError(
            f"git {' '.join(argv)} failed with exit {completed.returncode}: {detail}"
        )
    return completed.stdout


def _read_descriptor_bytes(descriptor: int) -> bytes:
    before = os.fstat(descriptor)
    chunks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        block = os.pread(descriptor, min(1024 * 1024, before.st_size - offset), offset)
        if not block:
            raise VerificationProfileError(
                "anonymous Git index ended before its authenticated size"
            )
        chunks.append(block)
        offset += len(block)
    after = os.fstat(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise VerificationProfileError(
            "anonymous Git index changed during descriptor authentication"
        )
    return b"".join(chunks)


def _run_git_with_anonymous_index(
    root: Path,
    argv: tuple[str, ...],
    *,
    descriptor: int,
    expected_bytes: bytes,
) -> bytes:
    if _read_descriptor_bytes(descriptor) != expected_bytes:
        raise VerificationProfileError(
            "anonymous Git index differs from authenticated source before Git operation"
        )
    os.lseek(descriptor, 0, os.SEEK_SET)
    environment = dict(os.environ)
    environment["GIT_INDEX_FILE"] = f"/dev/fd/{descriptor}"
    output = _run_git(
        root,
        argv,
        env=environment,
        pass_fds=(descriptor,),
    )
    if _read_descriptor_bytes(descriptor) != expected_bytes:
        raise VerificationProfileError(
            "Git operation mutated the anonymous authenticated index"
        )
    return output


def _git_index_path(root: Path) -> Path | None:
    completed = subprocess.run(
        (_GIT_EXECUTABLE, "rev-parse", "--git-path", "index"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if completed.returncode != 0:
        return None
    path = Path(completed.stdout.strip())
    return path if path.is_absolute() else root / path


def _read_external_regular(path: Path) -> tuple[bytes, str]:
    try:
        before = path.lstat()
    except OSError as error:
        raise VerificationProfileError(f"cannot inspect Git index {path}: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise VerificationProfileError("Git index must be a no-symlink regular file")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise VerificationProfileError(f"cannot open Git index {path}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    visible = path.lstat()
    identity = (opened.st_dev, opened.st_ino)
    if identity != (before.st_dev, before.st_ino) or identity != (
        after.st_dev,
        after.st_ino,
    ) or identity != (visible.st_dev, visible.st_ino) or (
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
    ) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise VerificationProfileError("Git index changed during authentication")
    raw = b"".join(chunks)
    return raw, hashlib.sha256(raw).hexdigest()


def _read_git_index(root: Path, path: Path) -> tuple[bytes, str]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return _read_external_regular(path)
    relative_posix = PurePosixPath(relative.as_posix())
    root_descriptor, _ = _open_authenticated_root(root)
    try:
        authenticated = _read_authenticated_path(root_descriptor, relative_posix)
    finally:
        os.close(root_descriptor)
    assert authenticated is not None and authenticated.content is not None
    return authenticated.content, authenticated.snapshot.sha256 or ""


def _parse_stage_zero_entries(raw: bytes) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, path_raw = record.split(b"\t", 1)
            mode_raw, blob_raw, stage_raw = metadata.split(b" ", 2)
            path = path_raw.decode("utf-8", "strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise VerificationProfileError("Git index emitted an invalid stage entry") from error
        if stage_raw != b"0":
            raise VerificationProfileError(f"Git index has unresolved stage for {path!r}")
        if path in entries:
            raise VerificationProfileError(f"Git index contains duplicate path {path!r}")
        entries[path] = (mode_raw.decode("ascii"), blob_raw.decode("ascii"))
    return entries


def _copy_protected_paths(profile: VerificationProfile, source: Path, target: Path) -> None:
    source_descriptor, _ = _open_authenticated_root(source)
    try:
        for relative_text in profile.protected_paths:
            relative = PurePosixPath(relative_text)
            authenticated = _read_authenticated_path(source_descriptor, relative)
            assert authenticated is not None
            destination = target.joinpath(*relative.parts)
            if authenticated.content is None:
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(authenticated.content)
            os.chmod(destination, stat.S_IMODE(authenticated.snapshot.mode))
    finally:
        os.close(source_descriptor)


def _copy_runtime_paths(profile: VerificationProfile, source: Path, target: Path) -> None:
    source_descriptor, _ = _open_authenticated_root(source)
    try:
        for relative_text in profile.runtime_paths:
            relative = PurePosixPath(relative_text)
            authenticated = _read_authenticated_path(source_descriptor, relative)
            assert authenticated is not None
            if authenticated.content is None:
                raise VerificationProfileError(
                    f"runtime support declaration {relative_text!r} must be a regular file"
                )
            destination = target.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(authenticated.content)
            os.chmod(destination, stat.S_IMODE(authenticated.snapshot.mode))
    finally:
        os.close(source_descriptor)


def _copy_declared_snapshot(profile: VerificationProfile, source: Path, target: Path) -> None:
    source_descriptor, _ = _open_authenticated_root(source)
    try:
        for relative_text in _declaration_paths(profile, source):
            relative = PurePosixPath(relative_text)
            authenticated = _read_authenticated_path(source_descriptor, relative)
            assert authenticated is not None
            destination = target.joinpath(*relative.parts)
            if authenticated.content is None:
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(authenticated.content)
            os.chmod(destination, stat.S_IMODE(authenticated.snapshot.mode))
    finally:
        os.close(source_descriptor)


def _owned_content_digest(
    profile: VerificationProfile,
    snapshot_root: Path,
    *,
    index_entries: Mapping[str, tuple[str, str]] | None,
    object_format: str,
) -> str:
    root_descriptor, _ = _open_authenticated_root(snapshot_root)
    rows: list[dict[str, object]] = []
    try:
        for role, relative_text in (
            *(("owned", path) for path in profile.owned_paths),
            *(("runtime", path) for path in profile.runtime_paths),
        ):
            authenticated = _read_authenticated_path(
                root_descriptor, PurePosixPath(relative_text)
            )
            assert authenticated is not None and authenticated.content is not None
            content = authenticated.content
            row: dict[str, object] = {
                "path": relative_text,
                "role": role,
                "sha256": authenticated.snapshot.sha256,
                "size": authenticated.snapshot.size,
            }
            if index_entries is not None and role == "owned":
                entry = index_entries.get(relative_text)
                if entry is None:
                    raise VerificationProfileError(
                        f"required owned declaration {relative_text!r} is absent from index"
                    )
                mode, blob = entry
                expected_mode = (
                    "100755"
                    if authenticated.snapshot.mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                    else "100644"
                )
                if mode != expected_mode:
                    raise VerificationProfileError(
                        f"index mode for {relative_text!r} is {mode}, expected {expected_mode}"
                    )
                blob_hasher = hashlib.new(object_format)
                blob_hasher.update(f"blob {len(content)}\0".encode("ascii"))
                blob_hasher.update(content)
                if blob_hasher.hexdigest() != blob:
                    raise VerificationProfileError(
                        f"exported bytes for {relative_text!r} do not match index blob"
                    )
            rows.append(row)
    finally:
        os.close(root_descriptor)
    encoded = json.dumps(
        rows, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_blob_oid(content: bytes, object_format: str) -> str:
    try:
        hasher = hashlib.new(object_format)
    except ValueError as error:
        raise VerificationProfileError(
            f"unsupported Git object format {object_format!r}"
        ) from error
    hasher.update(f"blob {len(content)}\0".encode("ascii"))
    hasher.update(content)
    return hasher.hexdigest()


def _read_exported_index_bytes(
    root_descriptor: int,
    relative: PurePosixPath,
    mode: str,
) -> bytes:
    if mode in ("100644", "100755"):
        authenticated = _read_authenticated_path(root_descriptor, relative)
        assert authenticated is not None and authenticated.content is not None
        executable = bool(
            authenticated.snapshot.mode
            & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        )
        expected_executable = mode == "100755"
        if executable != expected_executable:
            raise VerificationProfileError(
                f"mounted mode for {relative} does not match index mode {mode}"
            )
        return authenticated.content
    if mode == "120000":
        parent_and_leaf = _open_authenticated_parent(
            root_descriptor, relative, allow_missing=False
        )
        assert parent_and_leaf is not None
        parent, leaf = parent_and_leaf
        try:
            before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISLNK(before.st_mode):
                raise VerificationProfileError(
                    f"mounted path {relative} is not the index-declared symlink"
                )
            target = os.readlink(leaf, dir_fd=parent)
            after = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise VerificationProfileError(
                    f"mounted symlink {relative} changed during authentication"
                )
            return os.fsencode(target)
        except FileNotFoundError as error:
            raise VerificationProfileError(
                f"required index path {relative} is missing from mounted export"
            ) from error
        finally:
            os.close(parent)
    raise VerificationProfileError(
        f"unsupported Git index mode {mode!r} for {relative}"
    )


def _mounted_leaf_paths(root: Path) -> set[str]:
    leaves: set[str] = set()
    pending: list[tuple[Path, PurePosixPath]] = [(root, PurePosixPath())]
    while pending:
        directory, relative_directory = pending.pop()
        try:
            entries = tuple(os.scandir(directory))
        except OSError as error:
            raise VerificationProfileError(
                f"cannot enumerate mounted export directory {relative_directory}: {error}"
            ) from error
        for entry in entries:
            relative = relative_directory / entry.name
            if entry.is_symlink() or entry.is_file(follow_symlinks=False):
                rendered = relative.as_posix()
                if rendered in leaves:
                    raise VerificationProfileError(
                        f"mounted export contains duplicate path {rendered!r}"
                    )
                leaves.add(rendered)
            elif entry.is_dir(follow_symlinks=False):
                pending.append((Path(entry.path), relative))
            else:
                raise VerificationProfileError(
                    f"mounted export contains unsupported file type at {relative}"
                )
    return leaves


def _authenticate_full_index_export(
    profile: VerificationProfile,
    snapshot: _ExecutionSnapshot,
    mounted_root: Path,
) -> None:
    if not snapshot.index_entries:
        return
    root_descriptor, _ = _open_authenticated_root(mounted_root)
    try:
        for path, mode, expected_blob in snapshot.index_entries:
            relative = _validate_relative_path(path, field="Git index entries")
            content = _read_exported_index_bytes(root_descriptor, relative, mode)
            actual_blob = _git_blob_oid(content, snapshot.object_format)
            if actual_blob != expected_blob:
                raise VerificationProfileError(
                    f"mounted index blob mismatch for {path!r}: "
                    f"expected {expected_blob}, got {actual_blob}"
                )
    finally:
        os.close(root_descriptor)

    allowed_leaves = {path for path, _mode, _blob in snapshot.index_entries}
    allowed_leaves.update(profile.runtime_paths)
    for relative in profile.protected_paths:
        metadata = (mounted_root / relative).lstat()
        if stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            allowed_leaves.add(relative)
    mounted_leaves = _mounted_leaf_paths(mounted_root)
    missing = sorted(allowed_leaves - mounted_leaves)
    extra = sorted(mounted_leaves - allowed_leaves)
    if missing or extra:
        raise VerificationProfileError(
            "mounted export leaf inventory mismatch: "
            f"missing={missing}, extra={extra}"
        )


@contextmanager
def _materialize_execution_snapshot(
    profile: VerificationProfile,
    root: Path,
    *,
    readonly: bool = False,
) -> Iterable[_ExecutionSnapshot]:
    with tempfile.TemporaryDirectory(prefix="keep-verification-") as temporary:
        snapshot_root = Path(temporary) / "repo"
        snapshot_root.mkdir()
        index_path = _git_index_path(root)
        if index_path is None:
            _copy_declared_snapshot(profile, root, snapshot_root)
            _, snapshots = _authenticate_declarations(profile, snapshot_root)
            synthetic = json.dumps(
                {
                    path: {
                        "mode": stat.S_IMODE(snapshot.mode),
                        "sha256": snapshot.sha256,
                        "size": snapshot.size,
                    }
                    for path, snapshot in sorted(snapshots.items())
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            index_sha256 = hashlib.sha256(synthetic).hexdigest()
            owned_sha256 = _owned_content_digest(
                profile,
                snapshot_root,
                index_entries=None,
                object_format="sha1",
            )
            authority = VerificationAuthority(
                profile_fingerprint(profile), owned_sha256, index_sha256, "NO_GIT"
            )
            snapshot = _ExecutionSnapshot(
                root=snapshot_root,
                authority=authority,
                source_index_path=None,
                source_index_sha256=index_sha256,
            )
            if readonly:
                with _readonly_execution_view(profile, snapshot) as mounted_root:
                    yield _ExecutionSnapshot(
                        root=mounted_root,
                        authority=authority,
                        source_index_path=None,
                        source_index_sha256=index_sha256,
                        read_only=True,
                    )
            else:
                yield snapshot
            return

        index_bytes, index_sha256 = _read_git_index(root, index_path)
        head = _run_git(root, ("rev-parse", "--verify", "HEAD")).decode().strip()
        object_format = _run_git(root, ("rev-parse", "--show-object-format")).decode().strip()
        with tempfile.TemporaryFile(prefix="keep-verification-index-") as private_index:
            private_index.write(index_bytes)
            private_index.flush()
            os.fsync(private_index.fileno())
            os.fchmod(private_index.fileno(), 0o400)
            stage_raw = _run_git_with_anonymous_index(
                root,
                ("ls-files", "--stage", "-z"),
                descriptor=private_index.fileno(),
                expected_bytes=index_bytes,
            )
            entries = _parse_stage_zero_entries(stage_raw)
            _run_git_with_anonymous_index(
                root,
                (
                    "checkout-index",
                    "--all",
                    "--force",
                    f"--prefix={snapshot_root.as_posix()}/",
                ),
                descriptor=private_index.fileno(),
                expected_bytes=index_bytes,
            )
        _, post_export_index_sha256 = _read_git_index(root, index_path)
        if post_export_index_sha256 != index_sha256:
            raise VerificationProfileError("Git index changed during snapshot export")
        _copy_protected_paths(profile, root, snapshot_root)
        _copy_runtime_paths(profile, root, snapshot_root)
        _authenticate_declarations(profile, snapshot_root)
        owned_sha256 = _owned_content_digest(
            profile,
            snapshot_root,
            index_entries=entries,
            object_format=object_format,
        )
        authority = VerificationAuthority(
            profile_fingerprint(profile), owned_sha256, index_sha256, head
        )
        captured_entries = tuple(
            (path, mode, blob)
            for path, (mode, blob) in sorted(entries.items())
        )
        snapshot = _ExecutionSnapshot(
            root=snapshot_root,
            authority=authority,
            source_index_path=index_path,
            source_index_sha256=index_sha256,
            index_entries=captured_entries,
            object_format=object_format,
        )
        if readonly:
            with _readonly_execution_view(profile, snapshot) as mounted_root:
                yield _ExecutionSnapshot(
                    root=mounted_root,
                    authority=authority,
                    source_index_path=index_path,
                    source_index_sha256=index_sha256,
                    index_entries=captured_entries,
                    object_format=object_format,
                    read_only=True,
                )
        else:
            yield snapshot


def get_verification_authority(
    profile_or_name: VerificationProfile | str, repo_root: str | Path
) -> VerificationAuthority:
    """Return the exact private-snapshot authority an approval must bind."""

    profile = (
        get_verification_profile(profile_or_name)
        if isinstance(profile_or_name, str)
        else profile_or_name
    )
    with _materialize_execution_snapshot(profile, Path(repo_root)) as snapshot:
        return snapshot.authority


def _run_hdiutil(argv: tuple[str, ...]) -> None:
    if sys.platform != "darwin" or shutil.which("hdiutil") is None:
        raise VerificationProfileError(
            "OS-enforced read-only execution snapshots require macOS hdiutil"
        )
    attempts = 2 if argv and argv[0] == "create" else 1
    for attempt in range(attempts):
        completed = subprocess.run(
            ("hdiutil", *argv),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        if completed.returncode == 0:
            return
        detail = completed.stderr.strip() or completed.stdout.strip()
        if attempt == 0 and attempts == 2 and not detail:
            continue
        raise VerificationProfileError(
            f"hdiutil {' '.join(argv)} failed with exit {completed.returncode}: {detail}"
        )


@contextmanager
def _readonly_execution_view(
    profile: VerificationProfile,
    snapshot: _ExecutionSnapshot,
) -> Iterable[Path]:
    """Seal authenticated bytes into an OS-enforced read-only command cwd."""

    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "-", profile.name)
    mount = Path(tempfile.gettempdir()) / (
        f"keep-verification-mount-{safe_name}-"
        f"{snapshot.authority.index_sha256[:16]}"
    )
    mutex_path = mount.with_name(mount.name + ".lock")
    mutex_descriptor = os.open(mutex_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(mutex_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise VerificationProfileError(
                f"read-only execution mount is already active: {mount}"
            ) from error
        if mount.exists():
            try:
                mount.rmdir()
            except OSError as error:
                raise VerificationProfileError(
                    f"read-only execution mount is not clean: {mount}"
                ) from error
        mount.mkdir()
    except Exception:
        os.close(mutex_descriptor)
        raise
    with tempfile.TemporaryDirectory(prefix="keep-verification-ro-") as temporary:
        temporary_path = Path(temporary)
        image = temporary_path / "snapshot.dmg"
        try:
            _run_hdiutil(
                (
                    "create",
                    "-quiet",
                    "-srcfolder",
                    str(snapshot.root),
                    "-fs",
                    "HFS+",
                    "-format",
                    "UDRO",
                    "-ov",
                    str(image),
                )
            )
            attached = False
            try:
                _run_hdiutil(
                    (
                        "attach",
                        "-quiet",
                        "-readonly",
                        "-nobrowse",
                        "-mountpoint",
                        str(mount),
                        str(image),
                    )
                )
                attached = True
                image.unlink()
                _authenticate_declarations(profile, mount)
                mounted_content_sha256 = _owned_content_digest(
                    profile,
                    mount,
                    index_entries=None,
                    object_format="sha1",
                )
                if mounted_content_sha256 != snapshot.authority.owned_content_sha256:
                    raise VerificationProfileError(
                        "read-only execution image does not match authenticated content"
                    )
                _authenticate_full_index_export(profile, snapshot, mount)
                yield mount
            finally:
                if attached:
                    try:
                        _run_hdiutil(("detach", "-quiet", str(mount)))
                    except VerificationProfileError:
                        _run_hdiutil(("detach", "-quiet", "-force", str(mount)))
        finally:
            try:
                mount.rmdir()
            finally:
                fcntl.flock(mutex_descriptor, fcntl.LOCK_UN)
                os.close(mutex_descriptor)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


def _review_check(
    profile: VerificationProfile,
    root: Path,
    authority: VerificationAuthority,
) -> VerificationCheckResult:
    relative = _validate_relative_path(
        profile.required_review_record, field="required_review_record"
    )
    root_descriptor, _ = _open_authenticated_root(root)
    try:
        authenticated = _read_authenticated_path(
            root_descriptor, relative, allow_missing=True
        )
    except VerificationProfileError as error:
        return VerificationCheckResult(
            name="adversarial-review",
            kind="review",
            status="blocked",
            reason=str(error),
        )
    finally:
        os.close(root_descriptor)
    if authenticated is None:
        return VerificationCheckResult(
            name="adversarial-review",
            kind="review",
            status="blocked",
            reason="review record absent",
        )
    raw = authenticated.content
    assert raw is not None
    identity = (
        f"device={authenticated.snapshot.device}|inode={authenticated.snapshot.inode}|"
        f"size={authenticated.snapshot.size}"
    )
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
        if type(payload) is not dict:
            raise ValueError("review record must be a JSON object")
        expected_keys = {
            "decision",
            "head_commit",
            "index_sha256",
            "owned_content_sha256",
            "profile",
            "profile_sha256",
            "reviewer",
            "schema_version",
        }
        unknown = sorted(set(payload) - expected_keys)
        missing = sorted(expected_keys - set(payload))
        if unknown:
            raise ValueError(f"unknown fields: {', '.join(unknown)}")
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("schema_version must be integer 1")
        for key in (
            "decision",
            "head_commit",
            "index_sha256",
            "owned_content_sha256",
            "profile",
            "profile_sha256",
            "reviewer",
        ):
            if type(payload[key]) is not str or not payload[key]:
                raise ValueError(f"{key} must be a nonempty string")
        if payload["profile"] != profile.name:
            raise ValueError("review profile does not match declaration")
        if payload["profile_sha256"] != profile_fingerprint(profile):
            raise ValueError("review profile fingerprint does not match declaration")
        if payload["reviewer"] != profile.reviewer_authority:
            raise ValueError("reviewer authority does not match declaration")
        if (
            payload["owned_content_sha256"] != authority.owned_content_sha256
            or payload["index_sha256"] != authority.index_sha256
            or payload["head_commit"] != authority.head_commit
        ):
            raise ValueError("review authority does not match authenticated execution")
        decision = str(payload["decision"])
        if decision != "approved":
            raise ValueError(f"review decision is {decision!r}, not approved")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        return VerificationCheckResult(
            name="adversarial-review",
            kind="review",
            status="blocked",
            reason=str(error),
            evidence_sha256=authenticated.snapshot.sha256,
            evidence_identity=identity,
        )
    return VerificationCheckResult(
        name="adversarial-review",
        kind="review",
        status="passed",
        evidence_sha256=authenticated.snapshot.sha256,
        evidence_identity=identity,
    )


def _default_staged_path_provider(root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        (
            _GIT_EXECUTABLE,
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--no-renames",
        ),
        cwd=root,
        check=False,
        capture_output=True,
        text=False,
        shell=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise VerificationProfileError(f"cannot inspect staged paths: {detail}")
    return tuple(
        part.decode("utf-8", "strict")
        for part in completed.stdout.split(b"\0")
        if part
    )


def _scope_violations(profile: VerificationProfile, staged_paths: Iterable[str]) -> tuple[str, ...]:
    owned = set(profile.owned_paths)
    protected = tuple(PurePosixPath(value) for value in profile.protected_paths)
    violations: set[str] = set()
    for raw in staged_paths:
        try:
            staged = _validate_relative_path(raw, field="staged paths")
        except VerificationProfileError:
            violations.add(f"unsafe staged path: {raw!r}")
            continue
        for boundary in protected:
            if staged == boundary or boundary in staged.parents:
                violations.add(f"staged protected path: {raw}")
        if staged.as_posix() not in owned:
            violations.add(f"staged path outside ownership fence: {raw}")
    return tuple(sorted(violations))


@dataclass
class _HeavyLockLease:
    state: str
    descriptor: int | None = None
    identity: tuple[int, int] | None = None
    stable: bool = True


def _visible_heavy_lock_identity(root: Path) -> tuple[int, int] | None:
    root_descriptor, _ = _open_authenticated_root(root)
    try:
        parent_and_leaf = _open_authenticated_parent(
            root_descriptor,
            PurePosixPath(".keep-heavy-job.lock"),
            allow_missing=True,
        )
        if parent_and_leaf is None:
            return None
        parent, leaf = parent_and_leaf
        try:
            metadata = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return None
        finally:
            os.close(parent)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise VerificationProfileError("heavy lock is not a no-symlink regular file")
        return metadata.st_dev, metadata.st_ino
    finally:
        os.close(root_descriptor)


def _open_heavy_lock(
    root: Path, *, writable: bool = True
) -> tuple[int, tuple[int, int]]:
    root_descriptor, _ = _open_authenticated_root(root)
    try:
        parent_and_leaf = _open_authenticated_parent(
            root_descriptor,
            PurePosixPath(".keep-heavy-job.lock"),
            allow_missing=False,
        )
        assert parent_and_leaf is not None
        parent, leaf = parent_and_leaf
        try:
            before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise VerificationProfileError(
                    "heavy lock is not a no-symlink regular file"
                )
            descriptor = os.open(
                leaf,
                (os.O_RDWR if writable else os.O_RDONLY)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent,
            )
            opened = os.fstat(descriptor)
            visible = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            identity = (opened.st_dev, opened.st_ino)
            if identity != (before.st_dev, before.st_ino) or identity != (
                visible.st_dev,
                visible.st_ino,
            ):
                os.close(descriptor)
                raise VerificationProfileError(
                    "heavy lock changed during descriptor authentication"
                )
            return descriptor, identity
        finally:
            os.close(parent)
    finally:
        os.close(root_descriptor)


def _acquire_heavy_lock_lease(
    root: Path,
    *,
    retain: bool,
    injected_state: str | None,
    writable: bool = True,
) -> _HeavyLockLease:
    if injected_state is not None:
        return _HeavyLockLease(injected_state)
    try:
        descriptor, identity = _open_heavy_lock(root, writable=writable)
    except (OSError, VerificationProfileError):
        return _HeavyLockLease("ambiguous")
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return _HeavyLockLease("held-known")
        except OSError as error:
            os.close(descriptor)
            if error.errno in (errno.EACCES, errno.EAGAIN):
                return _HeavyLockLease("held-known")
            return _HeavyLockLease("ambiguous")
        if _visible_heavy_lock_identity(root) != identity:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            return _HeavyLockLease("ambiguous")
        if retain:
            return _HeavyLockLease("free", descriptor, identity)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        return _HeavyLockLease("free")
    except Exception:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        raise


def _release_heavy_lock_lease(root: Path, lease: _HeavyLockLease) -> None:
    if lease.descriptor is None:
        return
    try:
        try:
            lease.stable = _visible_heavy_lock_identity(root) == lease.identity
        except (OSError, VerificationProfileError):
            lease.stable = False
        fcntl.flock(lease.descriptor, fcntl.LOCK_UN)
    finally:
        os.close(lease.descriptor)
        lease.descriptor = None


def _policy_checks(
    profile: VerificationProfile,
    *,
    heavy_lock_state: str,
    metal_available: bool | None,
) -> tuple[VerificationCheckResult, VerificationCheckResult]:
    if profile.metal_allowed:
        if metal_available is True:
            metal = VerificationCheckResult("metal", "policy", "passed")
        else:
            metal = VerificationCheckResult(
                "metal", "policy", "blocked", reason="Metal unavailable"
            )
    else:
        metal = VerificationCheckResult(
            "metal",
            "policy",
            "skipped-by-declared-policy",
            reason="profile forbids Metal; only pytest, compile, and help argv are permitted",
        )
    if profile.heavy_lock_allowed:
        if heavy_lock_state == "free":
            heavy = VerificationCheckResult("heavy-lock", "policy", "passed")
        elif heavy_lock_state == "ambiguous":
            heavy = VerificationCheckResult(
                "heavy-lock", "policy", "blocked", reason="heavy lock ambiguous"
            )
        else:
            heavy = VerificationCheckResult(
                "heavy-lock", "policy", "blocked", reason="heavy lock held"
            )
    else:
        heavy = VerificationCheckResult(
            "heavy-lock",
            "policy",
            "skipped-by-declared-policy",
            reason=(
                "profile forbids heavy work; only pytest, compile, and help argv are "
                f"permitted (observed lock: {heavy_lock_state})"
            ),
        )
    return metal, heavy


def _default_command_runner(
    argv: tuple[str, ...], *, cwd: Path, import_root: Path | None = None
):
    environment = dict(os.environ)
    snapshot_pythonpath = str((import_root or cwd) / "src")
    if environment.get("PYTHONPATH"):
        snapshot_pythonpath += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = snapshot_pythonpath
    pycache_root = Path(tempfile.gettempdir()) / "keep-verification-pycache"
    pycache_root.mkdir(parents=True, exist_ok=True)
    environment["PYTHONPYCACHEPREFIX"] = str(pycache_root)
    return subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        env=environment,
    )


def _commands(profile: VerificationProfile, python_executable: str) -> tuple[tuple[str, ...], ...]:
    pytest_policy: tuple[str, ...] = ()
    if profile.pytest_exclusions:
        pytest_policy = (
            "-k",
            " and ".join(
                f"not {exclusion}" for exclusion in profile.pytest_exclusions
            ),
        )
    commands: list[tuple[str, ...]] = [
        (
            python_executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *pytest_policy,
            *profile.pytest_modules,
        ),
        (python_executable, "-m", "compileall", "-q", *profile.compile_targets),
    ]
    commands.extend(
        (python_executable, *argv[1:]) for argv in profile.cli_help_checks
    )
    return tuple(commands)


def _command_name(index: int) -> str:
    return ("pytest", "compileall")[index] if index < 2 else f"cli-help-{index - 1}"


def _run_commands(
    profile: VerificationProfile,
    *,
    root: Path,
    command_runner: Callable[..., object],
    python_executable: str,
) -> tuple[VerificationCheckResult, ...]:
    results: list[VerificationCheckResult] = []
    for index, argv in enumerate(_commands(profile, python_executable)):
        try:
            outcome = command_runner(argv, cwd=root)
            exit_code = int(getattr(outcome, "returncode"))
            stdout = getattr(outcome, "stdout", "")
            stderr = getattr(outcome, "stderr", "")
            if type(stdout) is bytes:
                stdout = stdout.decode("utf-8", "replace")
            if type(stderr) is bytes:
                stderr = stderr.decode("utf-8", "replace")
            stdout = str(stdout)
            stderr = str(stderr)
            results.append(
                VerificationCheckResult(
                    name=_command_name(index),
                    kind="command",
                    status="passed" if exit_code == 0 else "failed",
                    argv=argv,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                )
            )
        except Exception as error:  # injected runners may model transport failure
            results.append(
                VerificationCheckResult(
                    name=_command_name(index),
                    kind="command",
                    status="failed",
                    argv=argv,
                    reason=f"command runner failed: {error}",
                )
            )
    return tuple(results)


def _blocked_commands(
    profile: VerificationProfile, *, python_executable: str, reason: str
) -> tuple[VerificationCheckResult, ...]:
    return tuple(
        VerificationCheckResult(
            name=_command_name(index),
            kind="command",
            status="blocked",
            argv=argv,
            reason=reason,
        )
        for index, argv in enumerate(_commands(profile, python_executable))
    )


def run_verification_profile(
    profile_or_name: VerificationProfile | str,
    repo_root: str | Path,
    *,
    command_runner: Callable[..., object] | None = None,
    staged_path_provider: Callable[[Path], Sequence[str]] | None = None,
    heavy_lock_state: str | None = None,
    metal_available: bool | None = None,
    python_executable: str | None = None,
    execution_snapshot_provider: Callable[
        [VerificationProfile, Path], object
    ]
    | None = None,
    readonly_snapshot: bool | None = None,
) -> VerificationRunResult:
    """Run one profile using authenticated declarations and direct argv only."""

    profile = (
        get_verification_profile(profile_or_name)
        if isinstance(profile_or_name, str)
        else profile_or_name
    )
    if type(profile) is not VerificationProfile:
        raise VerificationProfileError("profile must be a VerificationProfile or name")
    root = Path(repo_root)
    before_root, before_paths = _authenticate_declarations(profile, root)
    provider = staged_path_provider or _default_staged_path_provider
    try:
        before_staged = tuple(provider(root))
    except VerificationProfileError:
        raise
    except Exception as error:
        raise VerificationProfileError(f"cannot inspect staged paths: {error}") from error
    before_violations = _scope_violations(profile, before_staged)
    executable = python_executable or sys.executable
    snapshot_context = (
        execution_snapshot_provider(profile, root)
        if execution_snapshot_provider is not None
        else _materialize_execution_snapshot(
            profile,
            root,
            readonly=(command_runner is None)
            if readonly_snapshot is None
            else readonly_snapshot,
        )
    )
    try:
        snapshot_manager = snapshot_context
        with snapshot_manager as supplied_snapshot:  # type: ignore[attr-defined]
            if not isinstance(supplied_snapshot, _ExecutionSnapshot):
                raise VerificationProfileError(
                    "execution_snapshot_provider must yield an authenticated snapshot"
                )
            snapshot = supplied_snapshot
            review = _review_check(profile, root, snapshot.authority)
            snapshot_before_root, snapshot_before_paths = _authenticate_declarations(
                profile, snapshot.root
            )
            real_lease = _acquire_heavy_lock_lease(
                root,
                retain=profile.heavy_lock_allowed,
                injected_state=heavy_lock_state,
            )
            lock_state = real_lease.state
            if lock_state not in ("free", "held-known", "ambiguous"):
                raise VerificationProfileError(
                    f"unknown heavy lock state {lock_state!r}"
                )
            policy = _policy_checks(
                profile,
                heavy_lock_state=lock_state,
                metal_available=metal_available,
            )
            checks: list[VerificationCheckResult] = [
                VerificationCheckResult(
                    "declarations-before", "authentication", "passed"
                ),
                VerificationCheckResult(
                    "execution-snapshot", "authentication", "passed"
                ),
                *policy,
                review,
            ]
            pre_blockers = before_violations or any(
                check.status == "blocked" for check in policy
            )
            snapshot_lease: _HeavyLockLease | None = None
            try:
                if pre_blockers:
                    reason = (
                        "; ".join(before_violations)
                        if before_violations
                        else "required execution policy unavailable"
                    )
                    checks.extend(
                        _blocked_commands(
                            profile, python_executable=executable, reason=reason
                        )
                    )
                else:
                    snapshot_lease = _acquire_heavy_lock_lease(
                        snapshot.root,
                        retain=True,
                        injected_state=None,
                        writable=not snapshot.read_only,
                    )
                    if snapshot_lease.state != "free" or snapshot_lease.descriptor is None:
                        checks.extend(
                            _blocked_commands(
                                profile,
                                python_executable=executable,
                                reason="private snapshot heavy lock is unavailable",
                            )
                        )
                    else:
                        checks.extend(
                            _run_commands(
                                profile,
                                root=snapshot.root,
                                command_runner=command_runner
                                or (
                                    lambda argv, *, cwd: _default_command_runner(
                                        argv,
                                        cwd=cwd,
                                        import_root=snapshot.root,
                                    )
                                ),
                                python_executable=executable,
                            )
                        )
                post_reason: str | None = None
                try:
                    after_root, after_paths = _authenticate_declarations(profile, root)
                    if after_root != before_root or after_paths != before_paths:
                        post_reason = (
                            "repository or declared paths changed during verification"
                        )
                except VerificationProfileError as error:
                    post_reason = f"post-command authentication failed: {error}"
                try:
                    snapshot_after_root, snapshot_after_paths = (
                        _authenticate_declarations(profile, snapshot.root)
                    )
                    if (
                        snapshot_after_root != snapshot_before_root
                        or snapshot_after_paths != snapshot_before_paths
                    ):
                        post_reason = post_reason or (
                            "private execution snapshot changed during verification"
                        )
                except VerificationProfileError as error:
                    post_reason = post_reason or (
                        f"post-command snapshot authentication failed: {error}"
                    )
                if snapshot.source_index_path is not None:
                    _, current_index_sha256 = _read_git_index(
                        root, snapshot.source_index_path
                    )
                    if current_index_sha256 != snapshot.source_index_sha256:
                        post_reason = post_reason or (
                            "Git index changed during verification"
                        )
                    current_head = _run_git(
                        root, ("rev-parse", "--verify", "HEAD")
                    ).decode().strip()
                    if current_head != snapshot.authority.head_commit:
                        post_reason = post_reason or (
                            "Git HEAD changed during verification"
                        )
                try:
                    after_staged = tuple(provider(root))
                except Exception as error:
                    raise VerificationProfileError(
                        f"cannot inspect staged paths after commands: {error}"
                    ) from error
                after_violations = _scope_violations(profile, after_staged)
                if before_staged != after_staged:
                    post_reason = post_reason or (
                        "staged path inventory changed during verification"
                    )
                after_review = _review_check(profile, root, snapshot.authority)
                if after_review != review:
                    post_reason = post_reason or (
                        "review record changed during verification"
                    )
            finally:
                if snapshot_lease is not None:
                    _release_heavy_lock_lease(snapshot.root, snapshot_lease)
                    if not snapshot_lease.stable:
                        post_reason = post_reason or (
                            "private snapshot heavy lock authority changed"
                        )
                _release_heavy_lock_lease(root, real_lease)
                if not real_lease.stable:
                    post_reason = post_reason or "heavy lock authority changed"
            checks.append(
                VerificationCheckResult(
                    "declarations-after",
                    "authentication",
                    "blocked" if post_reason else "passed",
                    reason=post_reason,
                )
            )
    except VerificationProfileError as error:
        reason = f"authenticated index execution snapshot blocked: {error}"
        checks = [
            VerificationCheckResult(
                "declarations-before", "authentication", "passed"
            ),
            VerificationCheckResult(
                "execution-snapshot", "authentication", "blocked", reason=reason
            ),
            *_blocked_commands(
                profile, python_executable=executable, reason=reason
            ),
            VerificationCheckResult(
                "declarations-after", "authentication", "blocked", reason=reason
            ),
        ]
        after_staged = before_staged
        after_violations = before_violations
    scope_violations = tuple(sorted(set((*before_violations, *after_violations))))
    staged_paths = tuple(sorted(set((*before_staged, *after_staged))))
    statuses = {check.status for check in checks}
    commit_ready = not scope_violations and statuses <= {
        "passed",
        "skipped-by-declared-policy",
    }
    if "blocked" in statuses:
        status = "blocked"
    elif "failed" in statuses:
        status = "failed"
    else:
        status = "passed"
    return VerificationRunResult(
        profile=profile.name,
        profile_sha256=profile_fingerprint(profile),
        status=status,
        commit_ready=commit_ready,
        staged_paths=staged_paths,
        scope_violations=scope_violations,
        checks=tuple(checks),
    )


def _result_payload(result: VerificationRunResult) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    for check in result.checks:
        payload = asdict(check)
        payload["argv"] = list(check.argv)
        checks.append(payload)
    return {
        "checks": checks,
        "commit_ready": result.commit_ready,
        "profile": result.profile,
        "profile_sha256": result.profile_sha256,
        "scope_violations": list(result.scope_violations),
        "staged_paths": list(result.staged_paths),
        "status": result.status,
    }


def render_verification_result(
    result: VerificationRunResult, *, as_json: bool
) -> str:
    """Render deterministic JSON or concise terminal evidence."""

    if as_json:
        return json.dumps(
            _result_payload(result),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"
    lines = [
        f"Profile: {result.profile}",
        f"Status: {result.status}",
        f"Commit ready: {'yes' if result.commit_ready else 'no'}",
    ]
    for check in result.checks:
        line = f"- {check.name}: {check.status}"
        if check.exit_code is not None:
            line += f" (exit {check.exit_code})"
        if check.reason:
            line += f" — {check.reason}"
        lines.append(line)
    for violation in result.scope_violations:
        lines.append(f"- scope: {violation}")
    return "\n".join(lines) + "\n"


__all__ = [
    "AVAILABLE_VERIFICATION_PROFILES",
    "VerificationAuthority",
    "VerificationCheckResult",
    "VerificationProfile",
    "VerificationProfileError",
    "VerificationRunResult",
    "get_verification_profile",
    "get_verification_authority",
    "profile_fingerprint",
    "render_verification_result",
    "run_verification_profile",
]
