#!/usr/bin/env python3
"""Run an explicit Task 13 stage; default to pure package validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_COORDINATOR = (
    REPO_ROOT
    / "aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py"
)
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task13_campaign_runner import (  # noqa: E402
    CampaignJournalStores,
    CampaignRunnerError,
    ControllerExecutionAuthority,
    FileJournalStore,
    FinalizationCapture,
    SubprocessCoordinator,
    run_campaign_stage,
)
from glm52_enforcement.task13_controller_authority import (  # noqa: E402
    FULL_RUN_WORK_POINTER,
    ControllerAuthoritySigner,
)


def _read_canonical(path: Path) -> object:
    if not path.is_file() or path.is_symlink():
        raise CampaignRunnerError("input must be a regular non-symlink file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignRunnerError("input is not JSON") from error
    if raw != canonical_json_bytes(value) + b"\n":
        raise CampaignRunnerError("input is not canonical JSON plus LF")
    return value


def _authority(value: object) -> ControllerExecutionAuthority:
    fields = set(ControllerExecutionAuthority.__dataclass_fields__)
    if type(value) is not dict or set(value) != fields:
        raise CampaignRunnerError("controller authority fields drifted")
    try:
        return ControllerExecutionAuthority(**value)
    except TypeError as error:
        raise CampaignRunnerError("controller authority is invalid") from error


def _full_run_work(expected: Path) -> Path:
    if (
        not FULL_RUN_WORK_POINTER.is_file()
        or FULL_RUN_WORK_POINTER.is_symlink()
        or FULL_RUN_WORK_POINTER.resolve(strict=True)
        != FULL_RUN_WORK_POINTER
    ):
        raise CampaignRunnerError("FULL_RUN_WORK pointer drifted")
    info = FULL_RUN_WORK_POINTER.stat()
    raw = FULL_RUN_WORK_POINTER.read_bytes()
    if (
        stat.S_IMODE(info.st_mode) != 0o600
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
    ):
        raise CampaignRunnerError("FULL_RUN_WORK pointer custody drifted")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise CampaignRunnerError(
            "FULL_RUN_WORK pointer is not ASCII"
        ) from error
    if not text.endswith("\n") or "\n" in text[:-1]:
        raise CampaignRunnerError(
            "FULL_RUN_WORK pointer is not one normalized path"
        )
    work = Path(text[:-1])
    if (
        not expected.is_absolute()
        or expected != work
        or not work.is_absolute()
        or work.is_symlink()
        or not work.is_dir()
        or work.resolve(strict=True) != work
        or stat.S_IMODE(work.stat().st_mode) != 0o700
        or work.stat().st_uid != os.getuid()
    ):
        raise CampaignRunnerError("FULL_RUN_WORK custody drifted")
    return work


def _write_descriptor(descriptor: int, raw: bytes) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    remaining = memoryview(raw)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("capture output write made no progress")
        remaining = remaining[written:]
    os.fsync(descriptor)


class _FinalizationCaptureFiles:
    """Reserve and atomically retain three exact local evidence products."""

    def __init__(
        self,
        *,
        metadata_path: Path,
        payload_path: Path,
        gate_path: Path,
    ) -> None:
        self._paths = (
            Path(metadata_path),
            Path(payload_path),
            Path(gate_path),
        )
        self._descriptors: dict[Path, int] = {}
        self._created: list[Path] = []
        self._identities: dict[Path, tuple[int, int]] = {}
        self._persisted = False
        self._retain = False
        if len(set(self._paths)) != 3:
            raise CampaignRunnerError(
                "finalization capture outputs must be distinct"
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            for path in self._paths:
                if (
                    not path.is_absolute()
                    or path != path.resolve(strict=False)
                    or not path.parent.is_dir()
                    or path.parent.is_symlink()
                    or path.parent.resolve(strict=True) != path.parent
                ):
                    raise CampaignRunnerError(
                        "capture output must be absolute under a real directory"
                    )
                descriptor = os.open(path, flags, 0o600)
                self._created.append(path)
                self._descriptors[path] = descriptor
                created_info = os.fstat(descriptor)
                self._identities[path] = (
                    created_info.st_dev,
                    created_info.st_ino,
                )
                os.fchmod(descriptor, 0o600)
                info = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_uid != os.getuid()
                    or info.st_nlink != 1
                ):
                    raise CampaignRunnerError(
                        "capture output custody drifted"
                    )
        except BaseException as error:
            self.abort(primary_error=error)
            raise

    def __call__(self, capture: FinalizationCapture) -> None:
        if type(capture) is not FinalizationCapture or self._persisted:
            raise CampaignRunnerError("finalization capture delivery drifted")
        raw_values = (
            canonical_json_bytes(
                {
                    "StatusCode": capture.status_code,
                    "ExecutedVersion": capture.executed_version,
                }
            )
            + b"\n",
            capture.payload_bytes,
            capture.gate_bytes,
        )
        try:
            for path, raw in zip(self._paths, raw_values):
                _write_descriptor(self._descriptors[path], raw)
        except BaseException as error:
            self.abort(primary_error=error)
            raise
        self._persisted = True

    def commit(self) -> None:
        if not self._persisted:
            error = CampaignRunnerError(
                "finalization capture was not delivered"
            )
            self.abort(primary_error=error)
            raise error
        try:
            owns_all_paths = all(
                self._owns_path(path) for path in self._paths
            )
        except OSError as error:
            self.abort(primary_error=error)
            raise
        if not owns_all_paths:
            error = CampaignRunnerError(
                "capture output path custody drifted"
            )
            self.abort(primary_error=error)
            raise error
        close_error = self._close()
        if close_error is not None:
            self.abort(primary_error=close_error)
            raise close_error
        self._retain = True

    def _owns_path(self, path: Path) -> bool:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return False
        return (
            not path.is_symlink()
            and stat.S_ISREG(info.st_mode)
            and (info.st_dev, info.st_ino) == self._identities.get(path)
            and stat.S_IMODE(info.st_mode) == 0o600
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
        )


    def _close(self) -> OSError | None:
        close_error: OSError | None = None
        for descriptor in self._descriptors.values():
            try:
                os.close(descriptor)
            except OSError as error:
                if close_error is None:
                    close_error = error
        self._descriptors.clear()
        return close_error

    def abort(
        self,
        *,
        primary_error: BaseException | None = None,
    ) -> None:
        cleanup_error = self._close()
        if not self._retain:
            for path in self._created:
                try:
                    owns_path = self._owns_path(path)
                except OSError as error:
                    if cleanup_error is None:
                        cleanup_error = error
                    continue
                if not owns_path:
                    continue
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                except OSError as error:
                    if cleanup_error is None:
                        cleanup_error = error
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--reviewed-artifacts", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=(
            "validate",
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
        default="validate",
    )
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--full-run-work", type=Path)
    parser.add_argument("--signing-private-key", type=Path)
    parser.add_argument("--fence-journal", type=Path)
    parser.add_argument("--support-journal", type=Path)
    parser.add_argument("--campaign-journal", type=Path)
    parser.add_argument(
        "--finalize-invoke-metadata-output",
        type=Path,
    )
    parser.add_argument("--finalize-payload-output", type=Path)
    parser.add_argument("--gate-readback-output", type=Path)
    args = parser.parse_args(argv)
    capture_files: Optional[_FinalizationCaptureFiles] = None
    try:
        package = _read_canonical(args.package)
        artifacts = _read_canonical(args.reviewed_artifacts)
        capture_paths = (
            args.finalize_invoke_metadata_output,
            args.finalize_payload_output,
            args.gate_readback_output,
        )
        if args.stage == "finalize":
            if any(path is None for path in capture_paths):
                raise CampaignRunnerError(
                    "finalize requires all three local capture outputs"
                )
            assert all(path is not None for path in capture_paths)
            capture_files = _FinalizationCaptureFiles(
                metadata_path=args.finalize_invoke_metadata_output,
                payload_path=args.finalize_payload_output,
                gate_path=args.gate_readback_output,
            )
        elif any(path is not None for path in capture_paths):
            raise CampaignRunnerError(
                "finalization capture outputs are valid only for finalize"
            )
        if args.stage == "validate":
            if any(
                value is not None
                for value in (
                    args.authority,
                    args.full_run_work,
                    args.signing_private_key,
                    args.fence_journal,
                    args.support_journal,
                    args.campaign_journal,
                )
            ):
                raise CampaignRunnerError(
                    "validate stage refuses external controller arguments"
                )
            services = object()
            journal = object()
            authority = None
        else:
            if any(
                value is None
                for value in (
                    args.authority,
                    args.full_run_work,
                    args.signing_private_key,
                    args.fence_journal,
                    args.support_journal,
                    args.campaign_journal,
                )
            ):
                raise CampaignRunnerError(
                    "external stage requires authority, distinct fence, "
                    "support, and campaign journals, and "
                    "repository-owned SHA-pinned coordinator"
                )
            assert args.authority is not None
            assert args.full_run_work is not None
            assert args.signing_private_key is not None
            assert args.fence_journal is not None
            assert args.support_journal is not None
            assert args.campaign_journal is not None
            authority = _authority(_read_canonical(args.authority))
            journal = CampaignJournalStores(
                fence=FileJournalStore(args.fence_journal),
                support=FileJournalStore(args.support_journal),
                campaign=FileJournalStore(args.campaign_journal),
            )
            coordinator_sha256 = hashlib.sha256(
                PRODUCTION_COORDINATOR.read_bytes()
            ).hexdigest()
            signer = ControllerAuthoritySigner(
                private_key_path=args.signing_private_key,
                full_run_work=_full_run_work(args.full_run_work),
            )
            services = SubprocessCoordinator(
                PRODUCTION_COORDINATOR,
                coordinator_sha256,
                authority_signer=signer,
            )
        result = run_campaign_stage(
            package=package,
            reviewed_artifacts=artifacts,
            stage=args.stage,
            services=services,
            journal=journal,
            authority=authority,
            finalization_capture=capture_files,
        )
        output = canonical_json_bytes(result).decode("ascii")
        if capture_files is not None:
            capture_files.commit()
    except (CampaignRunnerError, OSError) as error:
        if capture_files is not None:
            capture_files.abort(primary_error=error)
        print("Task 13 runner refused: %s" % error, file=sys.stderr)
        return 64
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
