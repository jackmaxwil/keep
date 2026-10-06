"""Closed local bridge from Task 10 to the Task 13 durable launch route."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Mapping

from .canonical import canonical_json_bytes, canonical_sha256


ACCOUNT_ID = "246813579024"
REGION = "us-west-2"
PROFILE = "keep-gpu"
RUN_ID = "glm52-sky-20260724"
H100_BUCKET = "keep-glm52-models-246813579024-us-west-2"
RUNNER_RELATIVE_PATH = "aws/glm52-gpu/scripts/run_glm52_task13_campaign.py"
COORDINATOR_RELATIVE_PATH = (
    "aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py"
)


class Task13BridgeError(ValueError):
    """A Task 10 to Task 13 launch route was not pinned exactly."""


@dataclass(frozen=True)
class Task13LaunchBridge:
    schema_version: int
    record_type: str
    account_id: str
    region: str
    profile: str
    run_id: str
    activation_id: str
    package_path: str
    package_file_sha256: str
    package_body_sha256: str
    reviewed_artifacts_path: str
    reviewed_artifacts_file_sha256: str
    reviewed_artifacts_body_sha256: str
    controller_authority_path: str
    controller_authority_file_sha256: str
    controller_authority_body_sha256: str
    fence_journal_path: str
    support_journal_path: str
    campaign_journal_path: str
    runner_relative_path: str
    runner_file_sha256: str
    coordinator_relative_path: str
    coordinator_file_sha256: str
    h100_resume_ready_bucket: str
    h100_resume_ready_key: str
    h100_resume_ready_version_id: str
    h100_resume_ready_file_sha256: str
    h100_resume_ready_body_sha256: str
    task13_route_binding_identity_sha256: str
    canonical_identity_sha256: str


def _sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Task13BridgeError(label + " must be a lowercase SHA-256")
    return value


def _body(value: Task13LaunchBridge) -> dict[str, object]:
    result = asdict(value)
    result.pop("canonical_identity_sha256")
    return result


def task13_launch_bridge_from_mapping(value: object) -> Task13LaunchBridge:
    if type(value) is not dict:
        raise Task13BridgeError("Task 13 launch bridge must be an object")
    if set(value) != set(Task13LaunchBridge.__dataclass_fields__):
        raise Task13BridgeError("Task 13 launch bridge field set drifted")
    try:
        bridge = Task13LaunchBridge(**value)
    except TypeError as error:
        raise Task13BridgeError("Task 13 launch bridge is malformed") from error
    if (
        bridge.schema_version != 1
        or bridge.record_type != "glm52_task10_task13_launch_bridge_v1"
        or bridge.account_id != ACCOUNT_ID
        or bridge.region != REGION
        or bridge.profile != PROFILE
        or bridge.run_id != RUN_ID
        or not bridge.activation_id
        or bridge.runner_relative_path != RUNNER_RELATIVE_PATH
        or bridge.coordinator_relative_path != COORDINATOR_RELATIVE_PATH
        or bridge.h100_resume_ready_bucket != H100_BUCKET
        or not bridge.h100_resume_ready_key
        or not bridge.h100_resume_ready_version_id
    ):
        raise Task13BridgeError("Task 13 launch bridge scope drifted")
    for path in (
        bridge.package_path,
        bridge.reviewed_artifacts_path,
        bridge.controller_authority_path,
        bridge.fence_journal_path,
        bridge.support_journal_path,
        bridge.campaign_journal_path,
    ):
        if not os.path.isabs(path) or "\x00" in path:
            raise Task13BridgeError("Task 13 launch bridge path is not absolute")
    for field in (
        "package_file_sha256",
        "package_body_sha256",
        "reviewed_artifacts_file_sha256",
        "reviewed_artifacts_body_sha256",
        "controller_authority_file_sha256",
        "controller_authority_body_sha256",
        "runner_file_sha256",
        "coordinator_file_sha256",
        "h100_resume_ready_file_sha256",
        "h100_resume_ready_body_sha256",
        "task13_route_binding_identity_sha256",
    ):
        _sha(getattr(bridge, field), field)
    if bridge.canonical_identity_sha256 != canonical_sha256(_body(bridge)):
        raise Task13BridgeError("Task 13 launch bridge self-hash drifted")
    return bridge


def _read_pinned_canonical(
    path_value: str,
    *,
    file_sha256: str,
    body_sha256: str,
    label: str,
) -> bytes:
    path = Path(path_value)
    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise Task13BridgeError(label + " must be a regular non-symlink file")
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise Task13BridgeError(label + " is unreadable") from error
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or hashlib.sha256(raw).hexdigest() != file_sha256
    ):
        raise Task13BridgeError(label + " file identity drifted")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Task13BridgeError(label + " is not JSON") from error
    if raw != canonical_json_bytes(value) + b"\n" or canonical_sha256(value) != body_sha256:
        raise Task13BridgeError(label + " canonical body drifted")
    return raw


def _validate_journal(path_value: str) -> None:
    path = Path(path_value)
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise Task13BridgeError("Task 13 journal parent is not a real directory")
    if not path.exists():
        return
    try:
        info = path.lstat()
        raw = path.read_bytes()
    except OSError as error:
        raise Task13BridgeError("Task 13 journal is unreadable") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise Task13BridgeError("Task 13 journal must be a 0600 regular file")
    if raw and not raw.endswith(b"\n"):
        raise Task13BridgeError("Task 13 journal has a torn record")
    for row in raw.splitlines(keepends=True):
        try:
            value = json.loads(row)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Task13BridgeError("Task 13 journal is not canonical JSONL") from error
        if row != canonical_json_bytes(value) + b"\n":
            raise Task13BridgeError("Task 13 journal is not canonical JSONL")


def _validate_journal_set(bridge: Task13LaunchBridge) -> None:
    paths = [
        Path(value)
        for value in (
            bridge.fence_journal_path,
            bridge.support_journal_path,
            bridge.campaign_journal_path,
        )
    ]
    resolved = [path.resolve(strict=False) for path in paths]
    if paths != resolved or len(set(resolved)) != 3:
        raise Task13BridgeError(
            "Task 13 journal paths must be distinct and non-aliasing"
        )
    inodes = []
    for path in paths:
        _validate_journal(str(path))
        if path.exists():
            info = path.lstat()
            inodes.append((info.st_dev, info.st_ino))
    if len(inodes) != len(set(inodes)):
        raise Task13BridgeError(
            "Task 13 journal files must not alias one inode"
        )


class Task13LaunchBoundary:
    """Narrow Task 10 boundary which delegates effects to Task 13 only."""

    def __init__(self, bridge: Task13LaunchBridge, *, repo_root: Path) -> None:
        self._bridge = bridge
        self._repo_root = repo_root

    def _sources(self) -> tuple[bytes, bytes, bytes, Path]:
        bridge = self._bridge
        package = _read_pinned_canonical(
            bridge.package_path,
            file_sha256=bridge.package_file_sha256,
            body_sha256=bridge.package_body_sha256,
            label="Task 13 package",
        )
        artifacts = _read_pinned_canonical(
            bridge.reviewed_artifacts_path,
            file_sha256=bridge.reviewed_artifacts_file_sha256,
            body_sha256=bridge.reviewed_artifacts_body_sha256,
            label="Task 13 reviewed artifacts",
        )
        authority = _read_pinned_canonical(
            bridge.controller_authority_path,
            file_sha256=bridge.controller_authority_file_sha256,
            body_sha256=bridge.controller_authority_body_sha256,
            label="Task 13 controller authority",
        )
        _validate_journal_set(bridge)
        runner = self._repo_root / bridge.runner_relative_path
        coordinator = self._repo_root / bridge.coordinator_relative_path
        for path, expected, label in (
            (runner, bridge.runner_file_sha256, "Task 13 runner"),
            (coordinator, bridge.coordinator_file_sha256, "Task 13 coordinator"),
        ):
            if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise Task13BridgeError(label + " identity drifted")
        return package, artifacts, authority, runner

    def inspect(self, authority: object, execution_input: object) -> Mapping[str, object]:
        self._assert_authority(authority)
        self._sources()
        body = {
            "current_activation": True,
            "action_consumed": True,
            "task8_live_authority": getattr(
                authority, "task8_live_h1d_identity_sha256"
            ),
            "task8_spend_authority": getattr(
                authority, "task8_spend_authority_identity_sha256"
            ),
            "task8_spend_reserve": getattr(
                authority, "task8_spend_reserve_identity_sha256"
            ),
            "task9_launch_admission": getattr(
                authority, "task9_admission_identity_sha256"
            ),
            "task9_liability_custody": getattr(
                authority, "task9_custody_identity_sha256"
            ),
            "workflow_version_arn": getattr(authority, "workflow_version_arn"),
        }
        return {**body, "inspection_identity_sha256": canonical_sha256(body)}

    def _assert_authority(self, authority: object) -> None:
        if (
            getattr(authority, "task13_route_binding_identity_sha256", None)
            != self._bridge.task13_route_binding_identity_sha256
            or getattr(authority, "activation_id", None) != self._bridge.activation_id
            or getattr(authority, "h100_resume_ready_bucket", None)
            != self._bridge.h100_resume_ready_bucket
            or getattr(authority, "h100_resume_ready_key", None)
            != self._bridge.h100_resume_ready_key
            or getattr(authority, "h100_resume_ready_version_id", None)
            != self._bridge.h100_resume_ready_version_id
            or getattr(authority, "h100_resume_ready_file_sha256", None)
            != self._bridge.h100_resume_ready_file_sha256
            or getattr(authority, "h100_resume_ready_body_sha256", None)
            != self._bridge.h100_resume_ready_body_sha256
        ):
            raise Task13BridgeError("Task 10 authority does not bind Task 13 route")

    def _invoke(self, stage: str, authority: object) -> Mapping[str, object]:
        self._assert_authority(authority)
        package, artifacts, controller, runner = self._sources()
        with tempfile.TemporaryDirectory(prefix="glm52-task10-task13-") as directory:
            root = Path(directory)
            paths = []
            for name, raw in (("package.json", package), ("reviewed.json", artifacts), ("authority.json", controller)):
                path = root / name
                path.write_bytes(raw)
                os.chmod(path, 0o400)
                paths.append(path)
            completed = subprocess.run(
                [
                    "/usr/bin/python3", "-I", str(runner),
                    "--package", str(paths[0]),
                    "--reviewed-artifacts", str(paths[1]),
                    "--stage", stage,
                    "--authority", str(paths[2]),
                    "--fence-journal",
                    self._bridge.fence_journal_path,
                    "--support-journal",
                    self._bridge.support_journal_path,
                    "--campaign-journal",
                    self._bridge.campaign_journal_path,
                ],
                check=False,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={"HOME": os.environ.get("HOME", ""), "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"},
                timeout=15 * 60,
            )
        if completed.returncode != 0:
            raise Task13BridgeError("Task 13 guarded runner refused launch route")
        try:
            value = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Task13BridgeError("Task 13 guarded runner output is not JSON") from error
        if completed.stdout != canonical_json_bytes(value) + b"\n" or type(value) is not dict:
            raise Task13BridgeError("Task 13 guarded runner output drifted")
        return value

    def start_once(self, authority: object, execution_input: object) -> Mapping[str, object]:
        value = self._invoke("launch", authority)
        if (
            value.get("status") != "STAGE_COMMITTED"
            or value.get("stage") != "launch"
            or value.get("accepted_execution_arn") != getattr(authority, "expected_execution_arn", None)
        ):
            raise Task13BridgeError("Task 13 launch did not commit exact execution")
        body = {
            "classification": "STARTED",
            "execution_arn": getattr(authority, "expected_execution_arn"),
            "reserve_identity_sha256": getattr(authority, "task8_spend_reserve_identity_sha256"),
            "task9_custody_identity_sha256": getattr(authority, "task9_custody_identity_sha256"),
        }
        return {**body, "observation_identity_sha256": canonical_sha256(body)}

    def reconcile(self, authority: object, execution_input: object) -> Mapping[str, object]:
        value = self._invoke("launch", authority)
        if value.get("status") != "STAGE_COMMITTED" or value.get("stage") != "launch":
            raise Task13BridgeError("Task 13 launch journal reconciliation drifted")
        observation = value.get("workflow_reconciliation")
        if type(observation) is not dict:
            raise Task13BridgeError(
                "Task 13 launch result lacks durable workflow reconciliation"
            )
        return observation


__all__ = [
    "Task13BridgeError",
    "Task13LaunchBoundary",
    "Task13LaunchBridge",
    "task13_launch_bridge_from_mapping",
]
