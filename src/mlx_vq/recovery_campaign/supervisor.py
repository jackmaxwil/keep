"""Single-threaded posix-spawn supervisor for recovery campaign producers."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import time
from pathlib import Path

from .config import load_campaign_config
from .ledger import LedgerError, LedgerUncertainCommitError, append_event, load_ledger
from .launcher import command_sha256
from .observer import observe_campaign


CONTROL_FD = 3
STDOUT_FD = 4
STDERR_FD = 5
READY_FD = 6
TERMINAL_FD = 7
MUTEX_FD = 8
CHAIN_FD_BASE = 20

_CONTROL_FIELDS = {
    "schema_version",
    "argv",
    "repo_root",
    "campaign",
    "experiment",
    "transition",
    "command_sha256",
    "launch_token",
    "started_at",
    "stdout_path",
    "stderr_path",
    "terminal_path",
    "chain_names",
    "chain_identities",
    "terminal_context",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_control() -> dict[str, object]:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(CONTROL_FD, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    raw = b"".join(chunks)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or set(value) != _CONTROL_FIELDS:
        raise ValueError("supervisor control payload fields mismatch")
    if value.get("schema_version") != 1 or _canonical(value) != raw:
        raise ValueError("supervisor control payload is not canonical schema v1")
    return value


def _directory_identity(fd: int) -> tuple[int, int, int, int, int, int]:
    metadata = os.fstat(fd)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_nlink == 0:
        raise OSError("supervisor directory fd is not a linked real directory")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _verify_paths(
    control: dict[str, object],
    *,
    strict_epochs: bool,
) -> None:
    names = control["chain_names"]
    identities = control["chain_identities"]
    if not isinstance(names, list) or not isinstance(identities, list):
        raise ValueError("supervisor chain metadata must be arrays")
    fds = [CHAIN_FD_BASE + index for index in range(len(identities))]
    actual = [list(_directory_identity(fd)) for fd in fds]
    if strict_epochs and actual != identities:
        raise OSError("supervisor directory chain identity changed")
    if not strict_epochs and any(
        actual_identity[:2] != expected_identity[:2]
        for actual_identity, expected_identity in zip(actual, identities, strict=True)
    ):
        raise OSError("supervisor directory chain inode changed")
    for index, name in enumerate(names):
        if not isinstance(name, str):
            raise ValueError("supervisor chain name must be text")
        visible = os.stat(name, dir_fd=fds[index], follow_symlinks=False)
        if not stat.S_ISDIR(visible.st_mode) or [visible.st_dev, visible.st_ino] != identities[index + 1][:2]:
            raise OSError("supervisor directory chain was renamed or replaced")
    final_fd = fds[-1]
    for field, descriptor in (
        ("stdout_path", STDOUT_FD),
        ("stderr_path", STDERR_FD),
        ("terminal_path", TERMINAL_FD),
    ):
        name = Path(str(control[field])).name
        opened = os.fstat(descriptor)
        visible = os.stat(name, dir_fd=final_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink == 0
            or not stat.S_ISREG(visible.st_mode)
            or (opened.st_dev, opened.st_ino) != (visible.st_dev, visible.st_ino)
        ):
            raise OSError(f"supervisor {field} identity changed")


def _append_reconciled(
    path: Path,
    *,
    events,
    event_kind: str,
    experiment: str,
    payload: dict[str, object],
    append=append_event,
    load_events=load_ledger,
):
    head = events[-1].event_sha256 if events else None
    count = len(events)
    try:
        return append(
            path,
            event_kind=event_kind,
            experiment=experiment,
            payload=payload,
            expected_head_sha256=head,
            expected_event_count=count,
        )
    except LedgerUncertainCommitError:
        refreshed = load_events(path)
        if len(refreshed) != count + 1:
            raise
        candidate = refreshed[-1]
        if (
            candidate.event_kind != event_kind
            or candidate.experiment != experiment
            or candidate.payload != payload
            or candidate.previous_event_sha256 != head
        ):
            raise
        return candidate


def record_terminal_outcome(
    control: dict[str, object],
    *,
    producer_pid: int,
    supervisor_pid: int,
    exit_code: int,
    config_loader=load_campaign_config,
    observer=observe_campaign,
    append=append_event,
    load_events=load_ledger,
) -> dict[str, object]:
    context = control["terminal_context"]
    if context is None:
        return {"classification": "finished"}
    if not isinstance(context, dict) or set(context) != {
        "campaign_config_path",
        "campaign_config_sha256",
        "ledger_path",
    }:
        raise ValueError("terminal_context fields mismatch")
    config = config_loader(str(context["campaign_config_path"]))
    if config.campaign_config_sha256 != context["campaign_config_sha256"]:
        raise ValueError("terminal monitor campaign fingerprint mismatch")
    if config.campaign != control["campaign"]:
        raise ValueError("terminal monitor campaign identity mismatch")
    transition = config.transition(str(control["transition"]))
    if transition.experiment_name != control["experiment"]:
        raise ValueError("terminal monitor experiment identity mismatch")
    if list(transition.argv) != control["argv"]:
        raise ValueError("terminal monitor argv differs from reloaded matrix")
    if command_sha256(transition.argv) != control["command_sha256"]:
        raise ValueError("terminal monitor command hash mismatch")
    repo_root = Path(str(control["repo_root"])).absolute()
    ledger_path = Path(str(context["ledger_path"])).absolute()
    canonical_ledger = (
        repo_root
        / "artifacts"
        / "quality"
        / f"{config.campaign}-ledger.jsonl"
    )
    if ledger_path != canonical_ledger:
        raise ValueError("terminal monitor ledger is not canonical for repo root")
    launched_fields = {
        "pid",
        "supervisor_pid",
        "command_sha256",
        "started_at",
        "stdout_path",
        "stderr_path",
        "terminal_path",
        "launch_token",
        "transition",
        "campaign_config_sha256",
    }
    deadline = time.monotonic() + 30.0
    events = ()
    while time.monotonic() < deadline:
        try:
            events = load_events(ledger_path)
        except LedgerError:
            time.sleep(0.05)
            continue
        matching = [
            event
            for event in events
            if event.event_kind == "transition_launched"
            and event.experiment == control["experiment"]
            and event.payload.get("launch_token") == control["launch_token"]
        ]
        if matching:
            if len(matching) != 1:
                raise ValueError("terminal monitor launch token is not unique")
            launched = matching[0]
            expected = {
                "pid": producer_pid,
                "supervisor_pid": supervisor_pid,
                "command_sha256": control["command_sha256"],
                "started_at": control["started_at"],
                "stdout_path": control["stdout_path"],
                "stderr_path": control["stderr_path"],
                "terminal_path": control["terminal_path"],
                "launch_token": control["launch_token"],
                "transition": control["transition"],
                "campaign_config_sha256": config.campaign_config_sha256,
            }
            if set(launched.payload) != launched_fields or launched.payload != expected:
                raise ValueError(
                    "terminal monitor transition_launched identity mismatch"
                )
            break
        time.sleep(0.05)
    else:
        raise ValueError("terminal monitor could not bind transition_launched")
    observation = observer(config, repo_root)
    snapshot = observation.snapshot(str(control["experiment"]))
    physical_complete = snapshot.physically_complete
    manifest_sha256 = snapshot.manifest_file_sha256 if physical_complete else None
    event_kind = (
        "transition_completed"
        if exit_code == 0 and physical_complete
        else "transition_finished"
    )
    payload: dict[str, object] = {
        "transition": control["transition"],
        "launch_token": control["launch_token"],
        "pid": producer_pid,
        "supervisor_pid": supervisor_pid,
        "command_sha256": control["command_sha256"],
        "exit_code": exit_code,
        "campaign_config_sha256": config.campaign_config_sha256,
        "physical_complete": physical_complete,
        "manifest_sha256": manifest_sha256,
        "terminal_path": control["terminal_path"],
    }
    terminal_event = _append_reconciled(
        ledger_path,
        events=events,
        event_kind=event_kind,
        experiment=str(control["experiment"]),
        payload=payload,
        append=append,
        load_events=load_events,
    )
    return {
        "classification": (
            "completed" if event_kind == "transition_completed" else "finished"
        ),
        "ledger_event_sha256": terminal_event.event_sha256,
        "physical_complete": physical_complete,
        "manifest_sha256": manifest_sha256,
    }


def _write_terminal(payload: dict[str, object]) -> None:
    encoded = _canonical(payload) + b"\n"
    os.lseek(TERMINAL_FD, 0, os.SEEK_SET)
    os.ftruncate(TERMINAL_FD, 0)
    if os.write(TERMINAL_FD, encoded) != len(encoded):
        raise OSError("short supervisor terminal write")
    os.fsync(TERMINAL_FD)


def main() -> int:
    control: dict[str, object] = {}
    handshake_sent = False
    try:
        control = _read_control()
        _verify_paths(control, strict_epochs=True)
        argv = control["argv"]
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise ValueError("supervisor argv must be a string array")
        process = subprocess.Popen(
            argv,
            cwd=str(control["repo_root"]),
            stdout=STDOUT_FD,
            stderr=STDERR_FD,
            close_fds=True,
        )
        os.write(READY_FD, f"OK {process.pid}\n".encode("ascii"))
        handshake_sent = True
        exit_code = process.wait()
        payload: dict[str, object] = {
            "schema_version": 1,
            "classification": "finished",
            "producer_pid": process.pid,
            "supervisor_pid": os.getpid(),
            "exit_code": exit_code,
            "command_sha256": control["command_sha256"],
            "launch_token": control["launch_token"],
        }
        try:
            _verify_paths(control, strict_epochs=False)
        except OSError as identity_error:
            payload.update(
                {"classification": "uncertain", "identity_error": str(identity_error)}
            )
        else:
            payload.update(
                record_terminal_outcome(
                    control,
                    producer_pid=process.pid,
                    supervisor_pid=os.getpid(),
                    exit_code=exit_code,
                )
            )
        _write_terminal(payload)
    except BaseException as error:
        if not handshake_sent:
            os.write(
                READY_FD,
                f"ERR {type(error).__name__}: {error}\n".encode(
                    "utf-8", errors="replace"
                ),
            )
        try:
            _write_terminal(
                {
                    "schema_version": 1,
                    "classification": "uncertain",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "command_sha256": control.get("command_sha256"),
                    "launch_token": control.get("launch_token"),
                }
            )
        except BaseException:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
