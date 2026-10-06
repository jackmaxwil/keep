#!/usr/bin/env python3
"""Fixed reboot-safe GLM-5.2 deadline guard.

The executable accepts one closed verb used by the authenticated bootstrap or
the two exact systemd units.  It has no caller-selectable path, unit, command,
signal, or environment override.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import NoReturn


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task10_worker import (  # noqa: E402
    CAMPAIGN_UNIT,
    DEADLINE_STATE_PATH,
    STOP_PATH,
    Task10WorkerError,
    WORKER_BOOTSTRAP_DESCRIPTOR_PATH,
    WORKER_INSTANCE_OBSERVATION_PATH,
    build_deadline_state,
    build_worker_runtime_authority,
    evaluate_deadline,
    render_deadline_timer_dropin,
    validate_deadline_state,
    worker_bootstrap_descriptor_from_mapping,
    worker_instance_observation_from_mapping,
)


STATE = Path(DEADLINE_STATE_PATH)
KEY = Path("/etc/keep-glm52/deadline-state.key")
DESCRIPTOR = Path(WORKER_BOOTSTRAP_DESCRIPTOR_PATH)
OBSERVATION = Path(WORKER_INSTANCE_OBSERVATION_PATH)
STOP = Path(STOP_PATH)
DROPIN = Path(
    "/etc/systemd/system/keep-glm52-deadline.timer.d/"
    "10-immutable-deadline.conf"
)
ALLOWED = frozenset({"initialize", "pre-start", "timer"})


def _fail(message: str) -> NoReturn:
    print("glm52 deadline guard: " + message, file=sys.stderr)
    raise SystemExit(70)


def _read_canonical(path: Path, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise Task10WorkerError(label + " is unreadable") from exc
    if (
        type(value) is not dict
        or raw != canonical_json_bytes(value) + b"\n"
    ):
        raise Task10WorkerError(label + " is not canonical JSON")
    return value


def _atomic_write(path: Path, raw: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".",
        dir=str(path.parent),
    )
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _key() -> bytes:
    try:
        value = KEY.read_bytes()
    except OSError as exc:
        raise Task10WorkerError("persistent deadline key is absent") from exc
    if len(value) != 32:
        raise Task10WorkerError("persistent deadline key is corrupt")
    return value


def _runtime_authority():
    descriptor_raw = _read_canonical(
        DESCRIPTOR,
        "worker bootstrap descriptor",
    )
    observation_raw = _read_canonical(
        OBSERVATION,
        "worker instance observation",
    )
    descriptor = worker_bootstrap_descriptor_from_mapping(descriptor_raw)
    observation = worker_instance_observation_from_mapping(observation_raw)
    raw = DESCRIPTOR.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = os.environ.get("GLM52_DESCRIPTOR_FILE_SHA256")
    if expected is not None and expected != digest:
        raise Task10WorkerError("authenticated descriptor identity drifted")
    return build_worker_runtime_authority(descriptor, observation), digest


def _typed_state(raw: dict[str, object]):
    from glm52_enforcement.task10_worker import DeadlineState

    try:
        return DeadlineState(**raw)
    except TypeError as exc:
        raise Task10WorkerError("persistent deadline state is malformed") from exc


def _write_state(state: object) -> None:
    _atomic_write(
        STATE,
        canonical_json_bytes(asdict(state)) + b"\n",
        0o600,
    )


def _initialize() -> None:
    if STATE.exists() or KEY.exists():
        raise Task10WorkerError("deadline initialization is one-shot")
    runtime, descriptor_sha = _runtime_authority()
    signing_key = os.urandom(32)
    _atomic_write(KEY, signing_key, 0o600)
    try:
        execution_deadline = datetime.strptime(
            runtime.descriptor.execution_deadline,
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=timezone.utc)
        state = build_deadline_state(
            descriptor_file_sha256=descriptor_sha,
            instance_id=runtime.observation.instance_id,
            allocation_ordinal=runtime.observation.allocation_ordinal,
            execution_deadline=execution_deadline,
            signing_key=signing_key,
        )
        _write_state(state)
        _atomic_write(DROPIN, render_deadline_timer_dropin(state), 0o644)
    except BaseException:
        try:
            KEY.unlink()
        except FileNotFoundError:
            pass
        raise


def _load():
    _runtime, descriptor_sha = _runtime_authority()
    state = _typed_state(_read_canonical(STATE, "persistent deadline state"))
    key = _key()
    return (
        validate_deadline_state(
            state,
            descriptor_file_sha256=descriptor_sha,
            signing_key=key,
        ),
        descriptor_sha,
        key,
    )


def _stop_marker() -> None:
    STOP.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        STOP,
        canonical_json_bytes(
            {
                "record_type": "glm52_worker_stop_edge_v1",
                "source": "persistent-deadline-guard",
            }
        )
        + b"\n",
        0o644,
    )


def _persist_edge(state: object, edge: str, key: bytes) -> object:
    updated = build_deadline_state(
        descriptor_file_sha256=state.descriptor_file_sha256,
        instance_id=state.instance_id,
        allocation_ordinal=state.allocation_ordinal,
        execution_deadline=datetime.strptime(
            state.execution_deadline,
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=timezone.utc),
        signing_key=key,
        last_completed_edge=edge,
    )
    _write_state(updated)
    return updated


def _evaluate(mode: str) -> None:
    state, descriptor_sha, key = _load()
    decision = evaluate_deadline(
        now=datetime.now(timezone.utc),
        state=state,
        descriptor_file_sha256=descriptor_sha,
        signing_key=key,
        stop_marker_present=STOP.exists(),
    )
    if decision.recreate_stop_marker:
        _stop_marker()
    if decision.persist_edge is not None:
        state = _persist_edge(state, decision.persist_edge, key)
    if mode == "pre-start":
        if not decision.allow_campaign_start:
            raise SystemExit(75)
        return
    if decision.systemctl_commands:
        if decision.systemctl_commands != (
            ("systemctl", "stop", CAMPAIGN_UNIT),
        ):
            raise Task10WorkerError("deadline command set drifted")
        result = subprocess.run(
            ["/usr/bin/systemctl", "stop", CAMPAIGN_UNIT],
            check=False,
            timeout=1201,
        )
        if result.returncode != 0:
            raise Task10WorkerError("fixed graceful stop did not complete")
        evidence = subprocess.run(
            [
                str(
                    REPO_ROOT
                    / "aws/glm52-gpu/scripts/"
                    "materialize_task10_graceful_stop.py"
                )
            ],
            check=False,
            timeout=30,
        )
        if evidence.returncode != 0:
            raise Task10WorkerError(
                "fixed graceful-stop evidence did not authenticate"
            )


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in ALLOWED:
        _fail("usage: glm52_deadline_guard.py initialize|pre-start|timer")
    try:
        if argv[0] == "initialize":
            _initialize()
        else:
            _evaluate(argv[0])
    except Task10WorkerError as exc:
        _fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
