#!/usr/bin/env python3
"""Materialize canonical local Task 10 graceful-stop evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import NoReturn


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task10_durable_s3 import (  # noqa: E402
    publish_or_adopt_exact,
)
from glm52_enforcement.task10_worker import (  # noqa: E402
    ACCOUNT_ID,
    CAMPAIGN_ROOT,
    CAMPAIGN_UNIT,
    DEADLINE_STATE_PATH,
    DESCRIPTOR_PATH,
    REGION,
    RUN_ID,
    STOP_PATH,
    UNIT_NAMES,
    WORKER_BOOTSTRAP_DESCRIPTOR_PATH,
    WORKER_INSTANCE_OBSERVATION_PATH,
    DeadlineState,
    build_graceful_stop_evidence,
    build_worker_runtime_authority,
    validate_deadline_state,
    validate_graceful_stop_evidence,
    worker_bootstrap_descriptor_from_mapping,
    worker_instance_observation_from_mapping,
)
from mlx_vq.quality.glm52_sky_campaign import (  # noqa: E402
    validate_sky_campaign_descriptor,
)


SYSTEMD = Path("/etc/systemd/system")
PRODUCTION_UNITS = REPO_ROOT / "aws/glm52-gpu/skypilot/production"
OUTPUT = Path(CAMPAIGN_ROOT) / "WORKER_GRACEFUL_STOP.json"
STOP = Path(STOP_PATH)
WRAPPER = Path(WORKER_BOOTSTRAP_DESCRIPTOR_PATH)
CAMPAIGN = Path(DESCRIPTOR_PATH)
STATE = Path(DEADLINE_STATE_PATH)
KEY = Path("/etc/keep-glm52/deadline-state.key")
OBSERVATION = Path(WORKER_INSTANCE_OBSERVATION_PATH)
CHECKPOINTS = (
    Path(CAMPAIGN_ROOT) / "training-checkpoints/latest.json",
    Path(CAMPAIGN_ROOT) / "teacher-checkpoints/latest.json",
)
LATEST = Path(CAMPAIGN_ROOT) / "ledger/latest.json"
TERMINAL = Path(CAMPAIGN_ROOT) / "CAMPAIGN_DRAINED.json"
SCRIPTS = (
    REPO_ROOT / "aws/glm52-gpu/scripts/collect_task10_ssm_graceful_stop.py",
    REPO_ROOT / "aws/glm52-gpu/scripts/glm52_deadline_guard.py",
    REPO_ROOT / "aws/glm52-gpu/scripts/materialize_task10_graceful_stop.py",
    REPO_ROOT / "aws/glm52-gpu/scripts/run_campaign.sh",
    REPO_ROOT
    / "aws/glm52-gpu/scripts/run_task10_production_campaign.py",
)


def _fail(message: str) -> NoReturn:
    print("Task 10 graceful-stop evidence: " + message, file=sys.stderr)
    raise SystemExit(70)


def _sha(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(label + " is absent or not a regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _show() -> dict[str, str]:
    result = subprocess.run(
        [
            "/usr/bin/systemctl",
            "show",
            CAMPAIGN_UNIT,
            "--property=ActiveState",
            "--property=SubState",
            "--property=Result",
            "--property=ExecMainCode",
            "--property=ExecMainStatus",
            "--property=MainPID",
            "--property=ControlGroup",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise ValueError("fixed campaign systemd readback failed")
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in values:
            raise ValueError("campaign systemd readback is malformed")
        values[key] = value
    return values


def _control_group_pids(readback: dict[str, str]) -> list[int]:
    main_pid = readback.get("MainPID")
    if main_pid != "0":
        raise ValueError("campaign main process remains present")
    control_group = readback.get("ControlGroup")
    if type(control_group) is not str or not control_group.startswith("/"):
        raise ValueError("campaign control group identity is absent")
    path = Path("/sys/fs/cgroup" + control_group) / "cgroup.procs"
    if not path.exists():
        return []
    raw = path.read_text().splitlines()
    try:
        pids = [int(value) for value in raw]
    except ValueError as exc:
        raise ValueError("campaign control group is malformed") from exc
    if any(value <= 0 for value in pids) or len(pids) != len(set(pids)):
        raise ValueError("campaign control group is malformed")
    return pids


def _hashes() -> tuple[dict[str, str], dict[str, str]]:
    units: dict[str, str] = {}
    for name in UNIT_NAMES:
        installed = SYSTEMD / name
        expected = PRODUCTION_UNITS / name
        installed_raw = installed.read_bytes()
        if installed_raw != expected.read_bytes():
            raise ValueError("installed unit bytes drifted")
        units[name] = hashlib.sha256(installed_raw).hexdigest()
    scripts = {
        path.name: _sha(path, path.name)
        for path in SCRIPTS
    }
    return units, scripts


def _exclusive(raw: bytes) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        OUTPUT,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            OUTPUT.unlink()
        except FileNotFoundError:
            pass
        raise


def _canonical(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(label + " is not canonical JSON")
    return value, raw


def _client() -> object:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=REGION,
        config=Config(
            retries={"mode": "standard", "total_max_attempts": 1},
            connect_timeout=2,
            read_timeout=15,
        ),
    )


def _runtime_authority() -> tuple[object, dict[str, object], DeadlineState]:
    wrapper_value, wrapper_raw = _canonical(
        WRAPPER,
        "worker bootstrap descriptor",
    )
    wrapper = worker_bootstrap_descriptor_from_mapping(wrapper_value)
    observation_value, _ = _canonical(
        OBSERVATION,
        "Task 9 worker instance observation",
    )
    observation = worker_instance_observation_from_mapping(
        observation_value
    )
    build_worker_runtime_authority(wrapper, observation)
    campaign_value, campaign_raw = _canonical(
        CAMPAIGN,
        "H.1c campaign descriptor",
    )
    campaign = validate_sky_campaign_descriptor(campaign_value)
    state_value, _ = _canonical(STATE, "persistent deadline state")
    signing_key = KEY.read_bytes()
    if len(signing_key) != 32:
        raise ValueError("persistent deadline key is corrupt")
    state = validate_deadline_state(
        DeadlineState(**state_value),
        descriptor_file_sha256=hashlib.sha256(wrapper_raw).hexdigest(),
        signing_key=signing_key,
    )
    if (
        hashlib.sha256(campaign_raw).hexdigest()
        != wrapper.base_descriptor_file_sha256
        or campaign["descriptor_body_sha256"]
        != wrapper.base_descriptor_body_sha256
        or campaign["campaign_identity_sha256"]
        != wrapper.campaign_identity_sha256
        or state.execution_deadline != wrapper.execution_deadline
        or state.instance_id != observation.instance_id
        or state.allocation_ordinal != observation.allocation_ordinal
    ):
        raise ValueError("graceful-stop runtime authority drifted")
    return wrapper, campaign, state


def _publish_or_adopt(
    evidence: dict[str, object],
    raw: bytes,
    *,
    client: object | None = None,
) -> str:
    wrapper, campaign, state = _runtime_authority()
    validate_graceful_stop_evidence(
        evidence,
        expected_unit_hashes=evidence["unit_file_sha256"],
        expected_script_hashes=evidence["script_file_sha256"],
    )
    if evidence["authority"] != "LOCAL_TIMER" or (
        evidence["ssm_command_id"] is not None
    ):
        raise ValueError("graceful-stop runtime authority drifted")
    bucket = str(campaign["bucket"])
    key = (
        f"campaigns/{RUN_ID}/submissions/production/generations/"
        f"{wrapper.generation:08d}/allocations/"
        f"{state.allocation_ordinal:08d}/WORKER_GRACEFUL_STOP.json"
    )
    metadata = {
        "glm52-account-id": ACCOUNT_ID,
        "glm52-activation-id": wrapper.activation_id,
        "glm52-body-sha256": str(
            evidence["graceful_stop_body_sha256"]
        ),
        "glm52-file-sha256": hashlib.sha256(raw).hexdigest(),
        "glm52-generation-text": wrapper.generation_text,
        "glm52-record-kind": "WORKER_GRACEFUL_STOP",
        "glm52-region": REGION,
        "glm52-run-id": RUN_ID,
    }
    return publish_or_adopt_exact(
        _client() if client is None else client,
        bucket=bucket,
        key=key,
        raw=raw,
        metadata=metadata,
    )


def _build_evidence(
    *,
    authority: str,
    ssm_command_id: str | None,
) -> dict[str, object]:
    _runtime_authority()
    readback = _show()
    unit_hashes, script_hashes = _hashes()
    checkpoint = next(
        (path for path in CHECKPOINTS if path.is_file()),
        None,
    )
    if checkpoint is None:
        raise ValueError("checkpoint latest evidence is absent")
    terminal_sha = (
        _sha(TERMINAL, "campaign terminal marker")
        if TERMINAL.is_file()
        else None
    )
    return dict(
        build_graceful_stop_evidence(
            unit_hashes=unit_hashes,
            script_hashes=script_hashes,
            active_state=readback.get("ActiveState", ""),
            sub_state=readback.get("SubState", ""),
            result=readback.get("Result", ""),
            exec_main_code=int(readback.get("ExecMainCode", "-1")),
            exec_main_status=int(readback.get("ExecMainStatus", "-1")),
            control_group_pids=_control_group_pids(readback),
            stop_file_identity_sha256=_sha(STOP, "stop file"),
            checkpoint_identity_sha256=_sha(
                checkpoint,
                "checkpoint latest",
            ),
            latest_marker_identity_sha256=_sha(
                LATEST,
                "campaign ledger latest",
            ),
            campaign_terminal_marker_identity_sha256=terminal_sha,
            ssm_command_id=ssm_command_id,
            authority=authority,
        )
    )


def main(argv: list[str]) -> int:
    if argv:
        _fail("this materializer accepts no arguments")
    if OUTPUT.exists():
        _fail("WORKER_GRACEFUL_STOP is immutable")
    try:
        evidence = _build_evidence(
            authority="LOCAL_TIMER",
            ssm_command_id=None,
        )
        raw = canonical_json_bytes(evidence) + b"\n"
        _publish_or_adopt(evidence, raw)
        _exclusive(raw)
    except (OSError, TypeError, ValueError) as exc:
        _fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
