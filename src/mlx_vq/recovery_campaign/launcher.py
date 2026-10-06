"""Descriptor-authenticated supervisor launcher for recovery transitions."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import secrets
import select
import signal
import stat
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .models import LauncherRecord, Transition


_RFC3339_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)


@dataclass(frozen=True, slots=True)
class _DirectoryChain:
    fds: tuple[int, ...]
    names: tuple[str, ...]
    identities: tuple[tuple[int, int, int, int, int, int], ...]

    @property
    def final_fd(self) -> int:
        return self.fds[-1]

    def close(self) -> None:
        for descriptor in reversed(self.fds):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _timestamp(value: str | None) -> str:
    candidate = value
    if candidate is None:
        candidate = datetime.now(timezone.utc).isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z")
    if _RFC3339_UTC_RE.fullmatch(candidate) is None:
        raise ValueError("launcher timestamp must be canonical RFC3339 UTC with Z")
    try:
        parsed = datetime.fromisoformat(candidate[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("launcher timestamp must be a valid UTC instant") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("launcher timestamp must be UTC")
    return candidate


def command_sha256(argv: tuple[str, ...]) -> str:
    """Return the canonical identity for one exact argv tuple."""

    encoded = json.dumps(
        list(argv), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _directory_identity(
    descriptor: int,
) -> tuple[int, int, int, int, int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_nlink == 0:
        raise OSError("launch directory descriptor is not a linked real directory")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_or_create_log_directory(root: Path, relative: Path) -> _DirectoryChain:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fds = [os.open(root, flags)]
    names: list[str] = []
    try:
        for component in relative.parts:
            try:
                os.mkdir(component, mode=0o700, dir_fd=fds[-1])
            except FileExistsError:
                pass
            child = os.open(component, flags, dir_fd=fds[-1])
            _directory_identity(child)
            fds.append(child)
            names.append(component)
        return _DirectoryChain(
            fds=tuple(fds),
            names=tuple(names),
            identities=tuple(_directory_identity(fd) for fd in fds),
        )
    except BaseException:
        for descriptor in reversed(fds):
            os.close(descriptor)
        raise


def _verify_chain(chain: _DirectoryChain) -> None:
    if tuple(_directory_identity(fd) for fd in chain.fds) != chain.identities:
        raise OSError("launch directory identity changed")
    for index, name in enumerate(chain.names):
        visible = os.stat(name, dir_fd=chain.fds[index], follow_symlinks=False)
        if not stat.S_ISDIR(visible.st_mode) or (
            visible.st_dev,
            visible.st_ino,
        ) != chain.identities[index + 1][:2]:
            raise OSError("launch directory path was replaced or renamed")


def _refresh_chain(chain: _DirectoryChain) -> _DirectoryChain:
    refreshed = _DirectoryChain(
        fds=chain.fds,
        names=chain.names,
        identities=tuple(_directory_identity(fd) for fd in chain.fds),
    )
    _verify_chain(refreshed)
    return refreshed


def _verify_log(
    chain: _DirectoryChain,
    name: str,
    descriptor: int,
) -> None:
    opened = os.fstat(descriptor)
    visible = os.stat(name, dir_fd=chain.final_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink == 0
        or not stat.S_ISREG(visible.st_mode)
        or (opened.st_dev, opened.st_ino) != (visible.st_dev, visible.st_ino)
    ):
        raise OSError("launch log identity changed")


def _before_supervisor_launch(
    _chain: _DirectoryChain,
    _stdout_fd: int,
    _stderr_fd: int,
) -> None:
    """Test hook immediately before final descriptor revalidation."""


def _record(
    *,
    producer_pid: int,
    supervisor_pid: int,
    command_digest: str,
    started_at: str,
    stdout_relative: Path,
    stderr_relative: Path,
    terminal_relative: Path,
    transition: Transition,
    launch_token: str,
) -> LauncherRecord:
    return LauncherRecord(
        pid=producer_pid,
        supervisor_pid=supervisor_pid,
        command_sha256=command_digest,
        launch_token=launch_token,
        started_at=started_at,
        stdout_path=stdout_relative.as_posix(),
        stderr_path=stderr_relative.as_posix(),
        terminal_path=terminal_relative.as_posix(),
        experiment_name=transition.experiment_name,
        transition_name=transition.name,
    )


def launch_transition(
    transition: Transition,
    *,
    repo_root: str | Path,
    campaign: str,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    timestamp: str | None = None,
    controller_mutex_fd: int | None = None,
    terminal_context: dict[str, object] | None = None,
    launch_token: str | None = None,
) -> LauncherRecord:
    """Launch via a detached supervisor that owns wait and terminal evidence."""

    if transition.action != "resume" or not transition.argv:
        raise ValueError("only a declared resume transition may be launched")
    root = Path(repo_root)
    started_at = _timestamp(timestamp)
    command_digest = command_sha256(transition.argv)
    launch_token = launch_token or secrets.token_hex(32)
    if not re.fullmatch(r"[0-9a-f]{64}", launch_token):
        raise ValueError("launch_token must be a lowercase SHA-256-shaped identity")
    safe_stamp = started_at.replace(":", "-")
    relative_dir = Path("artifacts") / "quality" / f"{campaign}-launches"
    stem = f"{safe_stamp}-{transition.name}-{command_digest[:12]}"
    stdout_relative = relative_dir / f"{stem}.stdout.log"
    stderr_relative = relative_dir / f"{stem}.stderr.log"
    terminal_relative = relative_dir / f"{stem}.terminal.json"
    chain = _open_or_create_log_directory(root, relative_dir)
    file_flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    stdout_fd = stderr_fd = terminal_fd = -1
    try:
        stdout_fd = os.open(stdout_relative.name, file_flags, 0o600, dir_fd=chain.final_fd)
        stderr_fd = os.open(stderr_relative.name, file_flags, 0o600, dir_fd=chain.final_fd)
        terminal_fd = os.open(
            terminal_relative.name,
            file_flags,
            0o600,
            dir_fd=chain.final_fd,
        )
        chain = _refresh_chain(chain)
        _before_supervisor_launch(chain, stdout_fd, stderr_fd)
        _verify_chain(chain)
        _verify_log(chain, stdout_relative.name, stdout_fd)
        _verify_log(chain, stderr_relative.name, stderr_fd)
        _verify_log(chain, terminal_relative.name, terminal_fd)

        # Injectable factories remain synchronous for deterministic unit tests.
        if popen_factory is not subprocess.Popen:
            with os.fdopen(os.dup(stdout_fd), "wb") as stdout_file, os.fdopen(
                os.dup(stderr_fd), "wb"
            ) as stderr_file:
                process = popen_factory(
                    list(transition.argv),
                    cwd=root,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                    close_fds=True,
                )
            return _record(
                producer_pid=process.pid,
                supervisor_pid=process.pid,
                command_digest=command_digest,
                started_at=started_at,
                stdout_relative=stdout_relative,
                stderr_relative=stderr_relative,
                terminal_relative=terminal_relative,
                transition=transition,
                launch_token=launch_token,
            )

        control_name = f".{stem}.control-{secrets.token_hex(8)}.json"
        control_fd = os.open(
            control_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=chain.final_fd,
        )
        ready_read, ready_write = os.pipe()
        dummy_mutex_fd = -1
        staged_fds: list[int] = []
        try:
            os.unlink(control_name, dir_fd=chain.final_fd)
            chain = _refresh_chain(chain)
            control: dict[str, object] = {
                "schema_version": 1,
                "argv": list(transition.argv),
                "repo_root": str(root.absolute()),
                "campaign": campaign,
                "experiment": transition.experiment_name,
                "transition": transition.name,
                "command_sha256": command_digest,
                "launch_token": launch_token,
                "started_at": started_at,
                "stdout_path": stdout_relative.as_posix(),
                "stderr_path": stderr_relative.as_posix(),
                "terminal_path": terminal_relative.as_posix(),
                "chain_names": list(chain.names),
                "chain_identities": [list(identity) for identity in chain.identities],
                "terminal_context": terminal_context,
            }
            encoded = json.dumps(
                control,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if os.write(control_fd, encoded) != len(encoded):
                raise OSError("short supervisor control write")
            os.fsync(control_fd)
            os.lseek(control_fd, 0, os.SEEK_SET)
            mutex_fd = controller_mutex_fd
            if mutex_fd is None:
                dummy_mutex_fd = os.open(os.devnull, os.O_RDONLY)
                mutex_fd = dummy_mutex_fd
            sources = [
                control_fd,
                stdout_fd,
                stderr_fd,
                ready_write,
                terminal_fd,
                mutex_fd,
                *chain.fds,
            ]
            staged_fds = [
                fcntl.fcntl(
                    descriptor,
                    fcntl.F_DUPFD_CLOEXEC,
                    100 + index,
                )
                for index, descriptor in enumerate(sources)
            ]
            targets = [3, 4, 5, 6, 7, 8] + [
                20 + index for index in range(len(chain.fds))
            ]
            file_actions: list[tuple] = [
                (os.POSIX_SPAWN_DUP2, source, target)
                for source, target in zip(staged_fds, targets, strict=True)
            ]
            supervisor_argv = (
                sys.executable,
                "-m",
                "mlx_vq.recovery_campaign.supervisor",
            )
            supervisor_pid = os.posix_spawn(
                sys.executable,
                supervisor_argv,
                dict(os.environ),
                file_actions=file_actions,
                setsid=True,
            )
            os.close(ready_write)
            ready_write = -1
            readable, _writable, _errors = select.select(
                [ready_read],
                [],
                [],
                10.0,
            )
            if not readable:
                os.killpg(supervisor_pid, signal.SIGTERM)
                os.waitpid(supervisor_pid, 0)
                raise TimeoutError("recovery supervisor handshake timed out")
            with os.fdopen(ready_read, "rb") as ready:
                ready_read = -1
                response = ready.readline(4096).decode(
                    "utf-8", errors="replace"
                ).strip()
        finally:
            if ready_write >= 0:
                os.close(ready_write)
            if ready_read >= 0:
                os.close(ready_read)
            os.close(control_fd)
            if dummy_mutex_fd >= 0:
                os.close(dummy_mutex_fd)
            for staged_fd in staged_fds:
                os.close(staged_fd)
        if not response.startswith("OK "):
            os.waitpid(supervisor_pid, 0)
            raise OSError(f"recovery supervisor failed before launch: {response}")
        producer_pid = int(response.split(" ", 1)[1])
        return _record(
            producer_pid=producer_pid,
            supervisor_pid=supervisor_pid,
            command_digest=command_digest,
            started_at=started_at,
            stdout_relative=stdout_relative,
            stderr_relative=stderr_relative,
            terminal_relative=terminal_relative,
            transition=transition,
            launch_token=launch_token,
        )
    finally:
        chain.close()
        for descriptor in (stdout_fd, stderr_fd, terminal_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


__all__ = ["command_sha256", "launch_transition"]
