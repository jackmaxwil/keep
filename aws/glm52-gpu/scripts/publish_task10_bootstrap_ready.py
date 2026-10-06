#!/usr/bin/env python3
"""Publish marker-last Task 10 bootstrap evidence from fixed local state."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import NoReturn


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.task10_worker import (  # noqa: E402
    ACCOUNT_ID,
    CAMPAIGN_ROOT,
    DEADLINE_DROPIN_PATH,
    DEADLINE_STATE_PATH,
    DESCRIPTOR_PATH,
    RUN_ID,
    UNIT_NAMES,
    WORKER_BOOTSTRAP_DESCRIPTOR_PATH,
    DeadlineState,
    build_bootstrap_ready,
    render_deadline_timer_dropin,
    validate_deadline_state,
    worker_bootstrap_descriptor_from_mapping,
)
from mlx_vq.quality.glm52_sky_campaign import (  # noqa: E402
    validate_sky_campaign_descriptor,
)


SYSTEMD = Path("/etc/systemd/system")
PRODUCTION_UNITS = (
    REPO_ROOT / "aws/glm52-gpu/skypilot/production"
)
MARKER = Path(CAMPAIGN_ROOT) / "BOOTSTRAP_READY.json"
STATE = Path(DEADLINE_STATE_PATH)
DESCRIPTOR = Path(DESCRIPTOR_PATH)
WORKER_DESCRIPTOR = Path(WORKER_BOOTSTRAP_DESCRIPTOR_PATH)
KEY = Path("/etc/keep-glm52/deadline-state.key")
DROPIN = Path(DEADLINE_DROPIN_PATH)
_VERSION = re.compile(r"^[A-Za-z0-9._~+/=-]{1,1024}$")


def _fail(message: str) -> NoReturn:
    print("task10 bootstrap evidence: " + message, file=sys.stderr)
    raise SystemExit(70)


def _canonical(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(label + " is unreadable") from exc
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(label + " is not canonical JSON")
    return value, raw


def _show(unit: str) -> dict[str, str]:
    result = subprocess.run(
        [
            "/usr/bin/systemctl",
            "show",
            unit,
            "--property=ActiveState",
            "--property=SubState",
            "--property=Type",
            "--property=NotifyAccess",
            "--property=FragmentPath",
            "--property=MainPID",
            "--property=ExecMainPID",
            "--property=NextElapseUSecRealtime",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "LANG": "C", "LC_ALL": "C"},
    )
    if result.returncode != 0:
        raise ValueError("fixed systemd readback failed")
    values = {}
    for line in result.stdout.splitlines():
        key, separator, item = line.partition("=")
        if not separator or key in values:
            raise ValueError("systemd readback is malformed")
        values[key] = item
    return values


def _canonical_next_elapse(value: object) -> str:
    if type(value) is not str or not value.endswith(" UTC"):
        raise ValueError("deadline timer next elapse is not exact UTC")
    try:
        parsed = datetime.strptime(
            value,
            "%a %Y-%m-%d %H:%M:%S UTC",
        )
    except ValueError as exc:
        raise ValueError(
            "deadline timer next elapse is malformed"
        ) from exc
    return parsed.replace(tzinfo=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _process_argv(pid: int) -> list[str]:
    raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    if not raw.endswith(b"\0"):
        raise ValueError("campaign main process argv is malformed")
    try:
        argv = [item.decode("utf-8") for item in raw[:-1].split(b"\0")]
    except UnicodeDecodeError as exc:
        raise ValueError("campaign main process argv is not UTF-8") from exc
    if not argv or any(not item for item in argv):
        raise ValueError("campaign main process argv is malformed")
    return argv


def _readback(deadline_state: DeadlineState) -> dict[str, object]:
    campaign = _show("keep-glm52-campaign.service")
    timer = _show("keep-glm52-deadline.timer")
    enabled = subprocess.run(
        ["/usr/bin/systemctl", "is-enabled", "keep-glm52-deadline.timer"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    main_pid_text = campaign.get("MainPID", "0")
    exec_main_pid_text = campaign.get("ExecMainPID", "0")
    if (
        not main_pid_text.isdigit()
        or not exec_main_pid_text.isdigit()
        or int(main_pid_text) <= 0
    ):
        raise ValueError("campaign main process identity is absent")
    main_pid = int(main_pid_text)
    expected_dropin = render_deadline_timer_dropin(deadline_state)
    observed_dropin = DROPIN.read_bytes()
    if observed_dropin != expected_dropin:
        raise ValueError("deadline timer drop-in bytes drifted")
    return {
        "campaign_active_state": campaign.get("ActiveState"),
        "campaign_sub_state": campaign.get("SubState"),
        "campaign_type": campaign.get("Type"),
        "campaign_notify_access": campaign.get("NotifyAccess"),
        "campaign_fragment_path": campaign.get("FragmentPath"),
        "campaign_main_pid": main_pid,
        "campaign_exec_main_pid": int(exec_main_pid_text),
        "campaign_main_argv": _process_argv(main_pid),
        "deadline_timer_active_state": timer.get("ActiveState"),
        "deadline_timer_sub_state": timer.get("SubState"),
        "deadline_timer_enabled_state": (
            enabled.stdout.strip() if enabled.returncode == 0 else ""
        ),
        "deadline_timer_next_elapse_usec_realtime": _canonical_next_elapse(
            timer.get("NextElapseUSecRealtime")
        ),
        "deadline_timer_dropin_path": str(DROPIN),
        "deadline_timer_dropin_sha256": hashlib.sha256(
            observed_dropin
        ).hexdigest(),
        "deadline_timer_on_calendar": [
            deadline_state.stop_assignment_at,
            deadline_state.graceful_stop_at,
        ],
    }


def _atomic_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
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
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _publish(
    *,
    marker: dict[str, object],
    raw: bytes,
    descriptor: dict[str, object],
    activation_id: str,
    generation: int,
    allocation_ordinal: int,
) -> None:
    import boto3
    from botocore.config import Config

    bucket = descriptor.get("bucket")
    campaign_identity = descriptor.get("campaign_identity_sha256")
    if type(bucket) is not str or type(campaign_identity) is not str:
        raise ValueError("descriptor publication identity is absent")
    key = (
        f"campaigns/{RUN_ID}/submissions/production/generations/"
        f"{generation:08d}/allocations/{allocation_ordinal:08d}/"
        "BOOTSTRAP_READY.json"
    )
    body_sha = str(marker["bootstrap_body_sha256"])
    file_sha = hashlib.sha256(raw).hexdigest()
    candidate_sha = hashlib.sha256(
        (
            campaign_identity
            + "\0"
            + body_sha
            + "\0"
            + file_sha
        ).encode("ascii")
    ).hexdigest()
    metadata = {
        "glm52-account-id": ACCOUNT_ID,
        "glm52-activation-id": activation_id,
        "glm52-body-sha256": body_sha,
        "glm52-candidate-identity-sha256": candidate_sha,
        "glm52-file-sha256": file_sha,
        "glm52-generation-text": f"{generation:08d}",
        "glm52-record-kind": "BOOTSTRAP_READY",
        "glm52-region": "us-west-2",
        "glm52-run-id": RUN_ID,
    }
    client = boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(
            retries={"mode": "standard", "total_max_attempts": 1},
            connect_timeout=2,
            read_timeout=15,
        ),
    )
    response = client.put_object(
        Bucket=bucket,
        Key=key,
        Body=raw,
        ContentType="application/json",
        Metadata=metadata,
        ChecksumAlgorithm="SHA256",
        ChecksumSHA256=base64.b64encode(
            hashlib.sha256(raw).digest()
        ).decode("ascii"),
        ExpectedBucketOwner=ACCOUNT_ID,
        IfNoneMatch="*",
    )
    metadata_response = response.get("ResponseMetadata")
    version_id = response.get("VersionId")
    if (
        type(metadata_response) is not dict
        or metadata_response.get("HTTPStatusCode") != 200
        or type(version_id) is not str
        or _VERSION.fullmatch(version_id) is None
        or version_id == "null"
    ):
        raise ValueError("bootstrap marker direct response is invalid")
    readback = client.get_object(
        Bucket=bucket,
        Key=key,
        VersionId=version_id,
        ExpectedBucketOwner=ACCOUNT_ID,
        ChecksumMode="ENABLED",
    )
    stream = readback.get("Body")
    expected_checksum = base64.b64encode(
        hashlib.sha256(raw).digest()
    ).decode("ascii")
    readback_metadata = readback.get("ResponseMetadata")
    if (
        readback.get("VersionId") != version_id
        or readback.get("ChecksumSHA256") != expected_checksum
        or type(readback_metadata) is not dict
        or readback_metadata.get("HTTPStatusCode") != 200
        or getattr(stream, "read", None) is None
        or stream.read() != raw
    ):
        raise ValueError("bootstrap marker exact-version readback drifted")


def main(argv: list[str]) -> int:
    if argv:
        _fail("this publisher accepts no arguments")
    if MARKER.exists():
        _fail("BOOTSTRAP_READY is immutable")
    try:
        descriptor, descriptor_raw = _canonical(
            DESCRIPTOR,
            "campaign descriptor",
        )
        worker_descriptor_raw, worker_descriptor_bytes = _canonical(
            WORKER_DESCRIPTOR,
            "worker bootstrap descriptor",
        )
        worker_descriptor = worker_bootstrap_descriptor_from_mapping(
            worker_descriptor_raw
        )
        campaign = validate_sky_campaign_descriptor(descriptor)
        if (
            hashlib.sha256(descriptor_raw).hexdigest()
            != worker_descriptor.base_descriptor_file_sha256
            or campaign["descriptor_body_sha256"]
            != worker_descriptor.base_descriptor_body_sha256
            or campaign["campaign_identity_sha256"]
            != worker_descriptor.campaign_identity_sha256
        ):
            raise ValueError(
                "accepted H.1c descriptor and worker wrapper drifted"
            )
        state_raw, _ = _canonical(STATE, "persistent deadline state")
        state = DeadlineState(**state_raw)
        signing_key = KEY.read_bytes()
        if len(signing_key) != 32:
            raise ValueError("persistent deadline key is corrupt")
        worker_descriptor_file_sha256 = hashlib.sha256(
            worker_descriptor_bytes
        ).hexdigest()
        state = validate_deadline_state(
            state,
            descriptor_file_sha256=worker_descriptor_file_sha256,
            signing_key=signing_key,
        )
        if state.execution_deadline != worker_descriptor.execution_deadline:
            raise ValueError(
                "deadline state and worker descriptor deadline drifted"
            )
        unit_hashes = {}
        for name in UNIT_NAMES:
            installed = (SYSTEMD / name).read_bytes()
            expected = (PRODUCTION_UNITS / name).read_bytes()
            if installed != expected:
                raise ValueError("installed unit bytes drifted")
            unit_hashes[name] = hashlib.sha256(installed).hexdigest()
        archive_hash = os.environ.get(
            "GLM52_REPOSITORY_ARCHIVE_FILE_SHA256",
            "",
        )
        archive_version = os.environ.get(
            "GLM52_REPOSITORY_ARCHIVE_VERSION_ID",
            "",
        )
        descriptor_version = worker_descriptor.base_descriptor_version_id
        generation = worker_descriptor.generation
        marker = build_bootstrap_ready(
            unit_hashes=unit_hashes,
            deadline_state=state,
            systemd_readback=_readback(state),
            descriptor_file_sha256=hashlib.sha256(descriptor_raw).hexdigest(),
            descriptor_version_id=descriptor_version,
            archive_file_sha256=archive_hash,
            archive_version_id=archive_version,
            instance_id=state.instance_id,
            allocation_ordinal=state.allocation_ordinal,
            marker_publish_sequence=9,
        )
        raw = canonical_json_bytes(marker) + b"\n"
        _publish(
            marker=marker,
            raw=raw,
            descriptor=descriptor,
            activation_id=worker_descriptor.activation_id,
            generation=generation,
            allocation_ordinal=state.allocation_ordinal,
        )
        _atomic_exclusive(MARKER, raw)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        _fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
