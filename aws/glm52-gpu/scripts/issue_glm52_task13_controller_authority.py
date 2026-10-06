#!/usr/bin/env python3
"""Issue one private, short-lived Task 13 controller execution authority."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
OWNER_APPROVAL = (
    REPO_ROOT
    / "docs/superpowers/approvals/"
    "2026-07-28-glm52-campaign-owner-approvals.md"
)
PRODUCTION_COORDINATOR = (
    REPO_ROOT
    / "aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py"
)
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task13_controller_authority import (  # noqa: E402
    FULL_RUN_WORK_POINTER,
    ControllerAuthoritySigner,
    ControllerAuthorityIssuerError,
    issue_controller_execution_authority,
    validate_owner_approval,
)


def _read_regular(path: Path, label: str) -> bytes:
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise ControllerAuthorityIssuerError(
            label + " must be an absolute regular non-symlink file"
        )
    return path.read_bytes()


def _read_canonical(path: Path, label: str) -> object:
    raw = _read_regular(path, label)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControllerAuthorityIssuerError(
            label + " is not JSON"
        ) from error
    if raw != canonical_json_bytes(value) + b"\n":
        raise ControllerAuthorityIssuerError(
            label + " is not canonical JSON plus LF"
        )
    return value


def _validated_coordinator_sha(expected_sha256: str) -> str:
    raw = _read_regular(PRODUCTION_COORDINATOR, "coordinator executable")
    mode = PRODUCTION_COORDINATOR.stat().st_mode
    if not stat.S_ISREG(mode) or mode & 0o111 == 0:
        raise ControllerAuthorityIssuerError(
            "coordinator executable is not executable"
        )
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise ControllerAuthorityIssuerError(
            "coordinator executable identity drifted"
        )
    return observed


def _write_new_private(path: Path, raw: bytes) -> None:
    if not path.is_absolute():
        raise ControllerAuthorityIssuerError(
            "output must be an absolute path"
        )
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ControllerAuthorityIssuerError(
            "output parent must be an existing real directory"
        )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _full_run_work(expected: Path) -> Path:
    raw = _read_regular(FULL_RUN_WORK_POINTER, "FULL_RUN_WORK pointer")
    pointer_info = FULL_RUN_WORK_POINTER.stat()
    if (
        stat.S_IMODE(pointer_info.st_mode) != 0o600
        or pointer_info.st_uid != os.getuid()
        or pointer_info.st_nlink != 1
    ):
        raise ControllerAuthorityIssuerError(
            "FULL_RUN_WORK pointer custody drifted"
        )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ControllerAuthorityIssuerError(
            "FULL_RUN_WORK pointer is not ASCII"
        ) from error
    if not text.endswith("\n") or "\n" in text[:-1]:
        raise ControllerAuthorityIssuerError(
            "FULL_RUN_WORK pointer is not one normalized path"
        )
    path = Path(text[:-1])
    if (
        not expected.is_absolute()
        or expected != path
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_dir()
        or path.resolve(strict=True) != path
        or stat.S_IMODE(path.stat().st_mode) != 0o700
        or path.stat().st_uid != os.getuid()
    ):
        raise ControllerAuthorityIssuerError(
            "FULL_RUN_WORK custody drifted"
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--reviewed-artifacts", required=True, type=Path)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "deploy-disabled",
            "collect-first-five",
            "collect-remaining",
            "finalize",
            "qualification-cache-seed",
            "h100-qualification",
            "launch",
            "terminal",
            "monitor",
        ),
    )
    parser.add_argument("--ttl-seconds", required=True, type=int)
    parser.add_argument(
        "--expected-owner-approval-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-coordinator-sha256",
        required=True,
    )
    parser.add_argument(
        "--signing-private-key",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--full-run-work",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        package = _read_canonical(args.package, "campaign package")
        artifacts = _read_canonical(
            args.reviewed_artifacts,
            "reviewed artifacts",
        )
        approval_raw = _read_regular(OWNER_APPROVAL, "owner approval")
        owner_approval_sha256 = validate_owner_approval(
            approval_raw,
            expected_sha256=args.expected_owner_approval_sha256,
        )
        coordinator_sha = _validated_coordinator_sha(
            args.expected_coordinator_sha256
        )
        full_run_work = _full_run_work(args.full_run_work)
        signer = ControllerAuthoritySigner(
            private_key_path=args.signing_private_key,
            full_run_work=full_run_work,
        )
        authority = issue_controller_execution_authority(
            package=package,
            reviewed_artifacts=artifacts,
            stage=args.stage,
            ttl_seconds=args.ttl_seconds,
            coordinator_executable_sha256=coordinator_sha,
            owner_approval_sha256=owner_approval_sha256,
            signer=signer,
        )
        _write_new_private(
            args.output,
            canonical_json_bytes(asdict(authority)) + b"\n",
        )
    except (
        ControllerAuthorityIssuerError,
        FileExistsError,
        OSError,
    ) as error:
        print(
            "Task 13 controller authority refused: %s" % error,
            file=sys.stderr,
        )
        return 64
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
