#!/usr/bin/env python3
"""Run the durable GLM-5.2 teacher-to-training campaign on one P5 node.

This controller is intentionally restart-oriented: it restores S3 artifacts,
validates the append-only campaign ledger, and derives the next phase from
durable outputs.  PIDs are used only to signal processes in the current boot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm52_enforcement.task10_worker import (  # noqa: E402
    DEADLINE_STATE_PATH,
    WORKER_BOOTSTRAP_DESCRIPTOR_PATH,
    DeadlineState,
    build_bootstrap_ready,
    validate_deadline_state,
)
from mlx_vq.quality.glm52_campaign_descriptor import (  # noqa: E402
    RuntimeCampaignKind,
    resolve_runtime_authority,
    validate_runtime_descriptor,
)
from mlx_vq.quality.glm52_campaign_watchdog import (  # noqa: E402
    build_campaign_heartbeat,
)
from mlx_vq.quality.glm52_teich_campaign import (  # noqa: E402
    CampaignLedger,
    CampaignPhase,
    training_preflight_decision,
)
from mlx_vq.quality.glm52_teich_training_campaign import (  # noqa: E402
    build_training_schedule,
)
from mlx_vq.quality.glm52_sky_campaign import SkyCampaignLedger  # noqa: E402
from mlx_vq.quality.glm52_sky_training_config import (  # noqa: E402
    validate_sky_training_config,
)

PHASES = tuple(CampaignPhase)
READY_NAME = "TEACHER_CACHE_READY.json"
MANIFEST_NAME = "glm52-teacher-signal-cache-v3-manifest.json"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = _canonical_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _run(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    output: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    for forbidden in ("GLM_MLX_WIRED_LIMIT_GB", "GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB"):
        merged.pop(forbidden, None)
    if output is None:
        return subprocess.run(command, env=merged, text=True, check=check)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        return subprocess.run(
            command,
            env=merged,
            text=True,
            check=check,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


class CampaignController:
    def __init__(self, *, descriptor_path: Path, root: Path, repo_root: Path) -> None:
        descriptor_value = _read_json(descriptor_path, label="campaign descriptor")
        self.descriptor, self.kind = validate_runtime_descriptor(descriptor_value)
        self.runtime = resolve_runtime_authority(
            self.descriptor,
            environ=os.environ,
        )
        self.descriptor_path = descriptor_path
        self.root = root
        self.repo_root = repo_root
        self.run_id = str(self.descriptor["run_id"])
        self.bucket = str(self.descriptor["bucket"])
        self.s3_root = f"s3://{self.bucket}/campaigns/{self.run_id}"
        self.execution_deadline = self.runtime.execution_deadline
        self.stop_file = Path(
            os.environ.get("GLM52_CAMPAIGN_STOP_FILE", "/run/keep-glm52/STOP")
        )
        self.ledger_path = self.root / "ledger/campaign-ledger.jsonl"
        self.capture_dir = self.root / "teacher-captures"
        self.teacher_checkpoint_dir = self.root / "teacher-checkpoints"
        self.cache_dir = self.root / "teacher-cache-v3"
        self.training_checkpoint_dir = self.root / "training-checkpoints"
        self.boundary_dir = self.root / "training-boundaries"
        self.adapter_dir = self.root / "canonical-adapter"
        self.evaluation_dir = self.root / "evaluation"
        self.logs_dir = self.root / "logs"
        self.monitor_dir = self.root / "monitor"
        self._sync_lock = threading.Lock()
        self._children_lock = threading.Lock()
        self._children: set[subprocess.Popen[Any]] = set()
        self._stop_monitor = threading.Event()
        self._final_sync_done = False
        self._production_sigterm_received = False
        for path in (
            self.root,
            self.capture_dir,
            self.teacher_checkpoint_dir,
            self.cache_dir,
            self.training_checkpoint_dir,
            self.boundary_dir,
            self.evaluation_dir,
            self.logs_dir,
            self.monitor_dir,
            self.stop_file.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def artifacts(self) -> Mapping[str, str]:
        value = self.descriptor["artifacts"]
        assert isinstance(value, dict)
        return value

    def _aws(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return _run(["aws", *arguments], check=check)

    def _run_managed_child(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        check: bool = True,
        output: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run one phase child that the deadline monitor can always terminate."""

        merged = os.environ.copy()
        if env:
            merged.update(env)
        for forbidden in (
            "GLM_MLX_WIRED_LIMIT_GB",
            "GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB",
        ):
            merged.pop(forbidden, None)
        handle = None
        try:
            if output is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
                handle = output.open("a", encoding="utf-8")
            child = subprocess.Popen(
                list(command),
                env=merged,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT if handle is not None else None,
            )
            with self._children_lock:
                self._children.add(child)
            try:
                returncode = child.wait()
            finally:
                with self._children_lock:
                    self._children.discard(child)
            if (
                returncode in {-signal.SIGTERM, 128 + signal.SIGTERM}
                and self.stop_file.exists()
            ):
                returncode = 75
            result = subprocess.CompletedProcess(list(command), returncode)
            if check and returncode:
                raise subprocess.CalledProcessError(returncode, list(command))
            return result
        finally:
            if handle is not None:
                handle.close()

    def restore(self) -> None:
        mappings = (
            ("ledger", self.root / "ledger"),
            ("teacher-checkpoints", self.teacher_checkpoint_dir),
            ("teacher-captures", self.capture_dir),
            ("teacher-cache-v3", self.cache_dir),
            ("training-checkpoints", self.training_checkpoint_dir),
            ("training-boundaries", self.boundary_dir),
            ("evaluation", self.evaluation_dir),
        )
        for relative, local in mappings:
            self._aws(
                "s3",
                "sync",
                f"{self.s3_root}/{relative}/",
                str(local) + "/",
                "--only-show-errors",
                check=False,
            )

    def _sync_marker_last(self, local: Path, s3: str) -> None:
        script = self.repo_root / "aws/glm52-gpu/scripts/sync_checkpoint_tree.sh"
        if local.exists():
            _run([str(script), str(local), s3])

    def sync_all(self, *, final: bool = False) -> None:
        if not self._sync_lock.acquire(blocking=False):
            return
        try:
            self._sync_marker_last(
                self.teacher_checkpoint_dir,
                f"{self.s3_root}/teacher-checkpoints",
            )
            self._sync_marker_last(
                self.training_checkpoint_dir,
                f"{self.s3_root}/training-checkpoints",
            )
            if self.capture_dir.exists():
                self._aws(
                    "s3",
                    "sync",
                    str(self.capture_dir) + "/",
                    f"{self.s3_root}/teacher-captures/",
                    "--only-show-errors",
                )
            if self.boundary_dir.exists():
                self._aws(
                    "s3",
                    "sync",
                    str(self.boundary_dir) + "/",
                    f"{self.s3_root}/training-boundaries/",
                    "--only-show-errors",
                )
            if self.evaluation_dir.exists():
                self._aws(
                    "s3",
                    "sync",
                    str(self.evaluation_dir) + "/",
                    f"{self.s3_root}/evaluation/",
                    "--only-show-errors",
                )
            if final and (self.cache_dir / READY_NAME).is_file():
                self._aws(
                    "s3",
                    "sync",
                    str(self.cache_dir) + "/",
                    f"{self.s3_root}/teacher-cache-v3/",
                    "--exclude",
                    MANIFEST_NAME,
                    "--exclude",
                    READY_NAME,
                    "--only-show-errors",
                )
                self._aws(
                    "s3",
                    "cp",
                    str(self.cache_dir / MANIFEST_NAME),
                    f"{self.s3_root}/teacher-cache-v3/{MANIFEST_NAME}",
                    "--only-show-errors",
                )
                self._aws(
                    "s3",
                    "cp",
                    str(self.cache_dir / READY_NAME),
                    f"{self.s3_root}/teacher-cache-v3/{READY_NAME}",
                    "--only-show-errors",
                )
            for marker_name in ("TRAINING_DEFERRED.json", "CAMPAIGN_FAILED.json"):
                marker = self.root / marker_name
                if marker.is_file():
                    self._aws(
                        "s3",
                        "cp",
                        str(marker),
                        f"{self.s3_root}/{marker_name}",
                        "--only-show-errors",
                    )
            self._sync_ledger()
        finally:
            self._sync_lock.release()

    def _sync_ledger(self) -> None:
        if not self.ledger_path.is_file():
            return
        lines = self.ledger_path.read_bytes().splitlines()
        for index, raw in enumerate(lines):
            record = json.loads(raw)
            immutable = self.root / "ledger/records" / (
                f"{index:03d}-{record['phase']}-{record['record_sha256']}.json"
            )
            if not immutable.exists():
                _write_json_atomic(immutable, record)
            self._aws(
                "s3",
                "cp",
                str(immutable),
                f"{self.s3_root}/ledger/records/{immutable.name}",
                "--only-show-errors",
            )
        self._aws(
            "s3",
            "cp",
            str(self.ledger_path),
            f"{self.s3_root}/ledger/campaign-ledger.jsonl",
            "--only-show-errors",
        )
        latest = self.root / "ledger/latest.json"
        _write_json_atomic(
            latest,
            {
                "record_type": (
                    "glm52_sky_campaign_ledger_latest_v2"
                    if self.kind is RuntimeCampaignKind.SKYPILOT
                    else "glm52_teich_campaign_ledger_latest_v1"
                ),
                "run_id": self.run_id,
                "record_count": len(lines),
                "record_sha256": json.loads(lines[-1])["record_sha256"],
                "ledger_sha256": _sha256_file(self.ledger_path),
            },
        )
        self._aws(
            "s3",
            "cp",
            str(latest),
            f"{self.s3_root}/ledger/latest.json",
            "--only-show-errors",
        )

    def ledger(self) -> CampaignLedger | SkyCampaignLedger:
        if self.kind is RuntimeCampaignKind.SKYPILOT:
            assert self.runtime.gpu_spend_authority_sha256 is not None
            return SkyCampaignLedger(
                self.ledger_path,
                run_id=self.run_id,
                execution_deadline=self.execution_deadline.isoformat().replace(
                    "+00:00", "Z"
                ),
                gpu_spend_authority_sha256=(
                    self.runtime.gpu_spend_authority_sha256
                ),
            )
        return CampaignLedger(
            self.ledger_path,
            run_id=self.run_id,
            reservation_deadline=str(self.descriptor["capacity_block_end"]),
        )

    def phase_done(self, phase: CampaignPhase) -> bool:
        current = self.ledger().current_phase
        return current is not None and PHASES.index(current) >= PHASES.index(phase)

    def transition(
        self,
        phase: CampaignPhase,
        *,
        inputs: Mapping[str, str],
        outputs: Mapping[str, str],
    ) -> None:
        if self.phase_done(phase):
            return
        ledger = self.ledger()
        if isinstance(ledger, SkyCampaignLedger):
            assert self.runtime.gpu_allocation_sha256 is not None
            ledger.transition(
                phase,
                input_identities=inputs,
                output_identities=outputs,
                gpu_allocation_sha256=self.runtime.gpu_allocation_sha256,
            )
        else:
            ledger.transition(
                phase,
                input_identities=inputs,
                output_identities=outputs,
            )
        self._sync_ledger()

    def _deadline_monitor(self) -> None:
        last_periodic = 0.0
        term_sent = False
        while not self._stop_monitor.wait(15):
            now = datetime.now(timezone.utc)
            if time.monotonic() - last_periodic >= 300:
                self.sync_all()
                self._publish_heartbeat()
                last_periodic = time.monotonic()
            if now >= self.execution_deadline - timedelta(minutes=60):
                self.stop_file.touch(exist_ok=True)
            if now >= self.execution_deadline - timedelta(minutes=50) and not term_sent:
                with self._children_lock:
                    children = tuple(self._children)
                for child in children:
                    if child.poll() is None:
                        child.terminate()
                term_sent = True
            if (
                now >= self.execution_deadline - timedelta(minutes=35)
                and not self._final_sync_done
            ):
                self.sync_all(final=True)
                self._final_sync_done = True
            drain_offset = 30 if self.kind is RuntimeCampaignKind.SKYPILOT else 32
            if now >= self.execution_deadline - timedelta(minutes=drain_offset):
                drained = self.root / "CAMPAIGN_DRAINED.json"
                if (
                    drained.is_file()
                    and self.kind is RuntimeCampaignKind.CAPACITY_BLOCK
                ):
                    self._terminate_this_instance()
                return

    def _publish_heartbeat(self) -> None:
        current_phase = "BOOTSTRAP_PENDING"
        record_sha256: str | None = None
        if self.ledger_path.is_file():
            ledger = self.ledger()
            if ledger.current_phase is not None:
                current_phase = ledger.current_phase.value
            if ledger.records:
                record_sha256 = ledger.records[-1].record_sha256
        heartbeat = build_campaign_heartbeat(
            run_id=self.run_id,
            phase=current_phase,
            observed_at=datetime.now(timezone.utc),
            ledger_record_sha256=record_sha256,
        )
        local = self.monitor_dir / "heartbeat.json"
        _write_json_atomic(local, heartbeat)
        self._aws(
            "s3",
            "cp",
            str(local),
            f"{self.s3_root}/monitor/heartbeat.json",
            "--only-show-errors",
            check=False,
        )

    def _terminate_this_instance(self) -> None:
        token = subprocess.run(
            [
                "curl",
                "-fsS",
                "-X",
                "PUT",
                "-H",
                "X-aws-ec2-metadata-token-ttl-seconds: 60",
                "http://169.254.169.254/latest/api/token",
            ],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not token:
            return
        instance_id = subprocess.run(
            [
                "curl",
                "-fsS",
                "-H",
                f"X-aws-ec2-metadata-token: {token}",
                "http://169.254.169.254/latest/meta-data/instance-id",
            ],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if instance_id:
            self._aws(
                "ec2",
                "terminate-instances",
                "--region",
                str(self.descriptor["region"]),
                "--instance-ids",
                instance_id,
                check=False,
            )

    def ensure_compute_window(self) -> None:
        if (
            self.stop_file.exists()
            or datetime.now(timezone.utc)
            >= self.execution_deadline - timedelta(minutes=60)
        ):
            self.sync_all(final=True)
            raise SystemExit(75)

    def teacher_deadline_arguments(self) -> list[str]:
        if self.kind is RuntimeCampaignKind.SKYPILOT:
            return [
                "--execution-deadline",
                self.execution_deadline.isoformat().replace("+00:00", "Z"),
            ]
        return [
            "--capacity-block-end",
            str(self.descriptor["capacity_block_end"]),
        ]

    def bootstrap(self) -> None:
        marker = self.root / "BOOTSTRAP_READY.json"
        production_bootstrap = os.environ.get("GLM52_MANAGED_MODE") == "production"
        if production_bootstrap:
            wait_deadline = time.monotonic() + 300
            while not marker.is_file() and time.monotonic() < wait_deadline:
                time.sleep(1)
            if not marker.is_file():
                raise RuntimeError(
                    "Task 10 marker-last systemd bootstrap evidence is absent"
                )
            report = _read_json(marker, label="Task 10 BOOTSTRAP_READY")
            wrapper_path = Path(WORKER_BOOTSTRAP_DESCRIPTOR_PATH)
            state_path = Path(DEADLINE_STATE_PATH)
            key_path = Path("/etc/keep-glm52/deadline-state.key")
            state_value = _read_json(
                state_path,
                label="Task 10 persistent deadline state",
            )
            try:
                state = DeadlineState(**state_value)
            except TypeError as exc:
                raise RuntimeError(
                    "Task 10 persistent deadline state schema drifted"
                ) from exc
            state = validate_deadline_state(
                state,
                descriptor_file_sha256=_sha256_file(wrapper_path),
                signing_key=key_path.read_bytes(),
            )
            try:
                expected_report = build_bootstrap_ready(
                    unit_hashes=report["unit_file_sha256"],
                    deadline_state=state,
                    systemd_readback=report["systemd_readback"],
                    descriptor_file_sha256=_sha256_file(
                        self.descriptor_path
                    ),
                    descriptor_version_id=report["descriptor_version_id"],
                    archive_file_sha256=report["archive_file_sha256"],
                    archive_version_id=report["archive_version_id"],
                    instance_id=state.instance_id,
                    allocation_ordinal=state.allocation_ordinal,
                    marker_publish_sequence=report[
                        "marker_publish_sequence"
                    ],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Task 10 BOOTSTRAP_READY identity/readback drifted"
                ) from exc
            if report != expected_report:
                raise RuntimeError(
                    "Task 10 BOOTSTRAP_READY identity/readback drifted"
                )
        elif not marker.is_file():
            if os.environ.get("GLM52_ENV_PREPARED") != "1":
                self._run_managed_child(
                    [str(self.repo_root / "aws/glm52-gpu/scripts/gpu_env_setup.sh")],
                    env={
                        "BUCKET": self.bucket,
                        "ROOT": str(self.root),
                        "CAMPAIGN_DESCRIPTOR": str(self.descriptor_path),
                        "KEEP_REPO_PRESTAGED": "1",
                        "KEEP_REPO_DIR": str(self.repo_root),
                    },
                    output=self.logs_dir / "bootstrap.log",
                )
            if self.kind is RuntimeCampaignKind.SKYPILOT:
                training_config = self.root / "training-config.json"
                validate_sky_training_config(
                    _read_json(
                        training_config,
                        label="Sky training configuration",
                    )
                )
                if _sha256_file(training_config) != self.artifacts[
                    "training_config_sha256"
                ]:
                    raise ValueError(
                        "Sky training configuration file SHA-256 mismatch"
                    )
            report = {
                "record_type": "glm52_teich_campaign_bootstrap_v1",
                "descriptor_body_sha256": self.descriptor["descriptor_body_sha256"],
                "campaign_identity_sha256": self.descriptor.get(
                    "campaign_identity_sha256",
                    self.descriptor["descriptor_body_sha256"],
                ),
                "repo_tar_sha256": self.descriptor["repo_tar_sha256"],
                "artifacts": self.artifacts,
                "environment_prestaged": (
                    os.environ.get("GLM52_ENV_PREPARED") == "1"
                ),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_json_atomic(marker, report)
        self.transition(
            CampaignPhase.BOOTSTRAP,
            inputs={
                "descriptor": str(
                    self.descriptor.get(
                        "campaign_identity_sha256",
                        self.descriptor["descriptor_body_sha256"],
                    )
                ),
                "repo_tar": str(self.descriptor["repo_tar_sha256"]),
            },
            outputs={"bootstrap_report": _sha256_file(marker)},
        )
        if production_bootstrap:
            _run(
                [
                    str(
                        self.repo_root
                        / "aws/glm52-gpu/scripts/"
                        "publish_task10_bootstrap_ledger_receipt.py"
                    )
                ]
            )

    def cuda_gate(self) -> None:
        if self.phase_done(CampaignPhase.CUDA_GATE):
            return
        self.ensure_compute_window()
        report = self.root / "spike-out/parity-report.json"
        self._run_managed_child(
            [str(self.repo_root / "aws/glm52-gpu/scripts/run_spike.sh")],
            env={
                "BUCKET": self.bucket,
                "ROOT": str(self.root),
                "CAMPAIGN_DESCRIPTOR": str(self.descriptor_path),
                "KEEP_REPO_DIR": str(self.repo_root),
            },
            output=self.logs_dir / "cuda-gate.log",
        )
        if not report.is_file():
            raise RuntimeError("CUDA gate did not publish its parity report")
        self.transition(
            CampaignPhase.CUDA_GATE,
            inputs={"bootstrap": self.ledger().records[-1].record_sha256},
            outputs={"cuda_parity_report": _sha256_file(report)},
        )

    def _teacher_partitions(self, count: int = 8) -> list[list[str]]:
        pack = _read_json(self.root / "teich-pack.json", label="Teich prompt pack")
        rows = pack.get("prompt_rows")
        if not isinstance(rows, list):
            raise ValueError("Teich prompt pack lacks prompt_rows")
        ordered = sorted(
            rows,
            key=lambda row: (
                -len(row.get("encoded_token_ids", [])),
                str(row.get("prompt_id")),
            ),
        )
        partitions: list[list[str]] = [[] for _ in range(count)]
        loads = [0] * count
        for row in ordered:
            prompt_id = row.get("prompt_id")
            tokens = row.get("encoded_token_ids")
            if not isinstance(prompt_id, str) or not isinstance(tokens, list):
                raise ValueError("Teich prompt row identity is malformed")
            worker = min(range(count), key=lambda index: (loads[index], index))
            partitions[worker].append(prompt_id)
            loads[worker] += len(tokens)
        return partitions

    def teacher(self) -> None:
        if self.phase_done(CampaignPhase.TEACHER):
            return
        self.ensure_compute_window()
        commands: list[tuple[int, list[str]]] = []
        for gpu, prompt_ids in enumerate(self._teacher_partitions()):
            if not prompt_ids:
                continue
            commands.append(
                (
                    gpu,
                    [
                        sys.executable,
                        str(self.repo_root / "benchmarks/produce_glm52_teich_teacher_cache.py"),
                        "--snapshot-dir",
                        str(self.root / "source-snapshot"),
                        "--non-vq-package-dir",
                        str(self.root / "non-vq-package"),
                        "--teich-pack",
                        str(self.root / "teich-pack.json"),
                        "--out-dir",
                        str(self.capture_dir),
                        "--heartbeat",
                        str(self.capture_dir / f"heartbeat-gpu-{gpu}.json"),
                        "--checkpoint-dir",
                        str(self.teacher_checkpoint_dir),
                        "--checkpoint-s3-prefix",
                        f"{self.s3_root}/teacher-checkpoints",
                        "--checkpoint-run-id",
                        self.run_id,
                        "--code-tar-sha256",
                        str(self.descriptor["repo_tar_sha256"]),
                        "--stop-file",
                        str(self.stop_file),
                        *self.teacher_deadline_arguments(),
                        "--session-ids",
                        *prompt_ids,
                    ],
                )
            )
        children: list[subprocess.Popen[Any]] = []
        for gpu, command in commands:
            log = (self.logs_dir / f"teacher-gpu-{gpu}.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"})
            environment.pop("GLM_MLX_WIRED_LIMIT_GB", None)
            environment.pop("GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB", None)
            child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            child._campaign_log = log  # type: ignore[attr-defined]
            children.append(child)
        with self._children_lock:
            self._children.update(children)
        statuses = []
        try:
            statuses = [child.wait() for child in children]
        finally:
            with self._children_lock:
                self._children.difference_update(children)
            for child in children:
                child._campaign_log.close()  # type: ignore[attr-defined]
        self.sync_all()
        if any(status == 75 for status in statuses):
            raise SystemExit(75)
        if any(status != 0 for status in statuses):
            raise RuntimeError(f"teacher workers failed: {statuses}")
        captures = sorted(self.capture_dir.glob("*.npz"))
        if len(captures) != 257:
            raise RuntimeError(f"teacher completed with {len(captures)} captures, expected 257")
        inventory = [
            {"name": path.name, "sha256": _sha256_file(path)} for path in captures
        ]
        inventory_sha = _sha256_bytes(_canonical_bytes(inventory))
        self.transition(
            CampaignPhase.TEACHER,
            inputs={"cuda_gate": self.ledger().records[-1].record_sha256},
            outputs={"capture_inventory": inventory_sha},
        )

    def cache_audit(self) -> None:
        if self.phase_done(CampaignPhase.CACHE_AUDIT):
            return
        self._run_managed_child(
            [
                sys.executable,
                str(self.repo_root / "benchmarks/finalize_glm52_teich_teacher_cache.py"),
                "--capture-dir",
                str(self.capture_dir),
                "--prompt-pack",
                str(self.root / "teich-pack.json"),
                "--frozen-prompt-pack",
                str(self.root / "frozen-66.json"),
                "--cache-dir",
                str(self.cache_dir),
            ],
            output=self.logs_dir / "cache-audit.log",
        )
        ready = _read_json(self.cache_dir / READY_NAME, label="teacher cache readiness")
        if ready.get("session_count") != 257 or ready.get("supervised_position_count") != 2_500_735:
            raise RuntimeError("teacher cache readiness totals do not match the campaign authority")
        self.sync_all(final=True)
        self.transition(
            CampaignPhase.CACHE_AUDIT,
            inputs={"captures": self.ledger().records[-1].record_sha256},
            outputs={
                "teacher_manifest": str(ready["manifest_sha256"]),
                "teacher_cache_ready": _sha256_file(self.cache_dir / READY_NAME),
            },
        )

    def _training_paths(self, *, preflight: bool, layer: int) -> tuple[Path, Path, Path, Path]:
        if layer == 77:
            checkpoint = self.root / ("preflight-checkpoints" if preflight else "training-checkpoints")
            boundaries = self.root / ("preflight-boundaries" if preflight else "training-boundaries")
            output = self.root / ("preflight-adapter-unused" if preflight else "canonical-adapter")
            sample = self.root / "TRAINING_PREFLIGHT_SAMPLE.json"
        else:
            lane = f"layer-{layer:03d}"
            checkpoint = self.training_checkpoint_dir / "expansions" / lane / (
                "preflight" if preflight else "training"
            )
            boundaries = self.boundary_dir / "expansions" / lane / (
                "preflight" if preflight else "training"
            )
            output = self.root / "expansion-adapters" / lane
            sample = self.evaluation_dir / f"{lane}-preflight.json"
        return checkpoint, boundaries, output, sample

    def _training_command(self, *, preflight: bool, layer: int = 77) -> list[str]:
        ready = _read_json(self.cache_dir / READY_NAME, label="teacher cache readiness")
        checkpoint, boundaries, output, sample = self._training_paths(
            preflight=preflight, layer=layer
        )
        s3_lane = "canonical" if layer == 77 else f"expansions/layer-{layer:03d}"
        command = [
            sys.executable,
            str(self.repo_root / "benchmarks/finetune_glm52_teich_resumable.py"),
            "--teacher-cache-dir",
            str(self.cache_dir),
            "--prompt-pack",
            str(self.root / "teich-pack.json"),
            "--frozen-prompt-pack",
            str(self.root / "frozen-66.json"),
            "--expected-teacher-manifest-sha256",
            str(ready["manifest_sha256"]),
            "--checkpoint-dir",
            str(checkpoint),
            "--checkpoint-s3-prefix",
            f"{self.s3_root}/training-checkpoints/{s3_lane}/"
            f"{'preflight' if preflight else 'training'}",
            "--boundary-dir",
            str(boundaries),
            "--output-dir",
            str(output),
            "--run-id",
            self.run_id + f"-layer-{layer}" + ("-preflight" if preflight else ""),
            "--stop-file",
            str(self.stop_file),
            "--layer",
            str(layer),
        ]
        if (checkpoint / "latest.json").is_file():
            command.append("--resume")
        if preflight:
            command.extend(
                [
                    "--preflight-only",
                    "--preflight-output",
                    str(sample),
                ]
            )
        return command

    def training_preflight(self) -> bool:
        decision_path = self.root / "TRAINING_PREFLIGHT_DECISION.json"
        if not self.phase_done(CampaignPhase.TRAINING_PREFLIGHT):
            self.ensure_compute_window()
            preflight_result = self._run_managed_child(
                self._training_command(preflight=True),
                env={
                    "CUDA_VISIBLE_DEVICES": "0",
                    "GLM52_TRAINING_BASELINE_JSON": str(
                        self.root / "training-baseline/training-baseline.json"
                    ),
                    "GLM52_TRAIN_PROGRESS": "1",
                    "GLM52_TRAIN_GRADIENT_CHECKPOINT": "1",
                },
                check=False,
                output=self.logs_dir / "training-preflight.log",
            )
            sample_path = self.root / "TRAINING_PREFLIGHT_SAMPLE.json"
            if preflight_result.returncode == 75:
                raise SystemExit(75)
            if preflight_result.returncode != 0 or not sample_path.is_file():
                payload = {
                    "record_type": "glm52_teich_training_preflight_decision_v1",
                    "start_training": False,
                    "reason": "authenticated_baseline_or_gpu_smoke_failed",
                    "preflight_exit_code": preflight_result.returncode,
                    "teacher_manifest_sha256": _sha256_file(
                        self.cache_dir / MANIFEST_NAME
                    ),
                }
            else:
                sample = _read_json(sample_path, label="training preflight sample")
                manifest = _read_json(self.cache_dir / MANIFEST_NAME, label="teacher manifest")
                schedule = build_training_schedule(
                    manifest["shards"], window_size=64, seed=20260712
                )
                decision = training_preflight_decision(
                    peak_gpu_gib=float(sample["peak_gpu_gib"]),
                    sample_steps=int(sample["sample_steps"]),
                    sample_seconds=float(sample["sample_seconds"]),
                    remaining_steps=len(schedule.windows),
                    now=datetime.now(timezone.utc),
                    execution_deadline=self.execution_deadline,
                )
                payload = {
                    "record_type": "glm52_teich_training_preflight_decision_v1",
                    **asdict(decision),
                    "sample_sha256": _sha256_file(sample_path),
                    "teacher_manifest_sha256": _sha256_file(
                        self.cache_dir / MANIFEST_NAME
                    ),
                }
            _write_json_atomic(decision_path, payload)
            self.transition(
                CampaignPhase.TRAINING_PREFLIGHT,
                inputs={"teacher_cache": self.ledger().records[-1].record_sha256},
                outputs={"training_preflight": _sha256_file(decision_path)},
            )
        return bool(_read_json(decision_path, label="training decision")["start_training"])

    def _audit_local_adapter(self) -> str:
        manifest_path = self.adapter_dir / "glm52-low-rank-adapter-manifest.json"
        manifest = _read_json(manifest_path, label="canonical adapter manifest")
        body = dict(manifest)
        candidate_sha = body.pop("candidate_identity_sha256", None)
        body_sha = body.pop("manifest_body_sha256", None)
        if body_sha != _sha256_bytes(_canonical_bytes(body)):
            raise ValueError("canonical adapter manifest body SHA-256 mismatch")
        for sidecar in body.get("sidecars", []):
            path = self.adapter_dir / sidecar["relative_path"]
            if _sha256_file(path) != sidecar["file_sha256"]:
                raise ValueError("canonical adapter sidecar SHA-256 mismatch")
        if not isinstance(candidate_sha, str) or len(candidate_sha) != 64:
            raise ValueError("canonical adapter candidate identity is invalid")
        return candidate_sha

    def _run_gpu_lanes(
        self,
        *,
        stage: str,
        commands: Mapping[int, Sequence[str]],
    ) -> dict[int, int]:
        children: dict[int, subprocess.Popen[Any]] = {}
        for gpu, command in commands.items():
            log = (self.logs_dir / f"{stage}-gpu-{gpu}.log").open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONUNBUFFERED": "1",
                    "GLM52_TRAINING_BASELINE_JSON": str(
                        self.root / "training-baseline/training-baseline.json"
                    ),
                    "GLM52_TRAIN_PROGRESS": "1",
                    "GLM52_TRAIN_GRADIENT_CHECKPOINT": "1",
                }
            )
            environment.pop("GLM_MLX_WIRED_LIMIT_GB", None)
            environment.pop("GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB", None)
            child = subprocess.Popen(
                list(command),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            child._campaign_log = log  # type: ignore[attr-defined]
            children[gpu] = child
        with self._children_lock:
            self._children.update(children.values())
        try:
            return {gpu: child.wait() for gpu, child in children.items()}
        finally:
            with self._children_lock:
                self._children.difference_update(children.values())
            for child in children.values():
                child._campaign_log.close()  # type: ignore[attr-defined]

    def expansion_candidates(self, canonical_report: Mapping[str, Any]) -> dict[str, Any]:
        expansion_report = self.evaluation_dir / "expansion-campaign.json"
        if expansion_report.is_file():
            return _read_json(expansion_report, label="expansion campaign report")
        gate = canonical_report.get("gate")
        if not isinstance(gate, dict) or gate.get("promising") is not True:
            payload = {
                "record_type": "glm52_teich_expansion_campaign_v1",
                "status": "not_started",
                "reason": "canonical_gate_not_passed",
                "layers": [],
                "automatic_promotion": False,
            }
            _write_json_atomic(expansion_report, payload)
            return payload
        manifest = _read_json(self.cache_dir / MANIFEST_NAME, label="teacher manifest")
        schedule = build_training_schedule(manifest["shards"], window_size=64, seed=20260712)
        canonical_sample = _read_json(
            self.root / "TRAINING_PREFLIGHT_SAMPLE.json",
            label="canonical training preflight sample",
        )
        initial_time_gate = training_preflight_decision(
            peak_gpu_gib=float(canonical_sample["peak_gpu_gib"]),
            sample_steps=20,
            sample_seconds=float(canonical_sample["sample_seconds"]),
            remaining_steps=len(schedule.windows),
            now=datetime.now(timezone.utc),
            execution_deadline=self.execution_deadline,
        )
        if not initial_time_gate.start_training:
            payload = {
                "record_type": "glm52_teich_expansion_campaign_v1",
                "status": "not_started",
                "reason": initial_time_gate.reason,
                "layers": [],
                "automatic_promotion": False,
            }
            _write_json_atomic(expansion_report, payload)
            return payload
        self.ensure_compute_window()
        layers = (75, 74, 76, 72, 73, 71, 70)
        gpu_for_layer = {layer: gpu for gpu, layer in enumerate(layers, start=1)}
        preflight_commands = {
            gpu_for_layer[layer]: self._training_command(preflight=True, layer=layer)
            for layer in layers
        }
        preflight_statuses = self._run_gpu_lanes(
            stage="expansion-preflight",
            commands=preflight_commands,
        )
        if any(status == 75 for status in preflight_statuses.values()):
            raise SystemExit(75)
        layer_results: dict[int, dict[str, Any]] = {}
        qualified: list[int] = []
        for layer in layers:
            gpu = gpu_for_layer[layer]
            _checkpoint, _boundaries, _output, sample_path = self._training_paths(
                preflight=True, layer=layer
            )
            result: dict[str, Any] = {
                "layer": layer,
                "gpu": gpu,
                "preflight_exit_code": preflight_statuses[gpu],
            }
            if preflight_statuses[gpu] == 0 and sample_path.is_file():
                sample = _read_json(sample_path, label=f"layer {layer} preflight sample")
                decision = training_preflight_decision(
                    peak_gpu_gib=float(sample["peak_gpu_gib"]),
                    sample_steps=int(sample["sample_steps"]),
                    sample_seconds=float(sample["sample_seconds"]),
                    remaining_steps=len(schedule.windows),
                    now=datetime.now(timezone.utc),
                    execution_deadline=self.execution_deadline,
                )
                result["preflight"] = {**asdict(decision), "sample_sha256": _sha256_file(sample_path)}
                if decision.start_training:
                    qualified.append(layer)
            layer_results[layer] = result
        training_commands = {}
        for layer in qualified:
            _checkpoint, _boundaries, output, _sample = self._training_paths(
                preflight=False, layer=layer
            )
            if not (output / "glm52-low-rank-adapter-manifest.json").is_file():
                training_commands[gpu_for_layer[layer]] = self._training_command(
                    preflight=False, layer=layer
                )
        training_statuses = self._run_gpu_lanes(
            stage="expansion-training",
            commands=training_commands,
        ) if training_commands else {}
        if any(status == 75 for status in training_statuses.values()):
            raise SystemExit(75)
        evaluation_commands = {}
        for layer in qualified:
            gpu = gpu_for_layer[layer]
            _checkpoint, _boundaries, output, _sample = self._training_paths(
                preflight=False, layer=layer
            )
            status = training_statuses.get(gpu, 0)
            layer_results[layer]["training_exit_code"] = status
            manifest_path = output / "glm52-low-rank-adapter-manifest.json"
            if status == 0 and manifest_path.is_file():
                adapter_manifest = _read_json(manifest_path, label=f"layer {layer} adapter")
                layer_results[layer]["candidate_identity_sha256"] = adapter_manifest[
                    "candidate_identity_sha256"
                ]
                report = self.evaluation_dir / f"layer-{layer:03d}/candidate-evaluation.json"
                if not report.is_file():
                    evaluation_commands[gpu] = [
                        sys.executable,
                        str(self.repo_root / "benchmarks/evaluate_glm52_teich_candidate.py"),
                        "--teacher-cache-dir",
                        str(self.cache_dir),
                        "--prompt-pack",
                        str(self.root / "teich-pack.json"),
                        "--frozen-prompt-pack",
                        str(self.root / "frozen-66.json"),
                        "--adapter-dir",
                        str(output),
                        "--output",
                        str(report),
                        "--layer",
                        str(layer),
                    ]
        evaluation_statuses = self._run_gpu_lanes(
            stage="expansion-evaluation",
            commands=evaluation_commands,
        ) if evaluation_commands else {}
        if any(status == 75 for status in evaluation_statuses.values()):
            raise SystemExit(75)
        for layer in qualified:
            gpu = gpu_for_layer[layer]
            report = self.evaluation_dir / f"layer-{layer:03d}/candidate-evaluation.json"
            layer_results[layer]["evaluation_exit_code"] = evaluation_statuses.get(gpu, 0)
            if report.is_file():
                layer_results[layer]["evaluation_report_sha256"] = _sha256_file(report)
        payload = {
            "record_type": "glm52_teich_expansion_campaign_v1",
            "status": "completed",
            "reason": "canonical_gate_passed_and_time_available",
            "layers": [layer_results[layer] for layer in layers],
            "automatic_promotion": False,
            "model_upload_performed": False,
        }
        _write_json_atomic(expansion_report, payload)
        self.sync_all()
        return payload

    def training(self, *, allowed: bool) -> bool:
        deferred_path = self.root / "TRAINING_DEFERRED.json"
        if self.phase_done(CampaignPhase.TRAINING):
            return not deferred_path.is_file()
        if not allowed:
            decision = _read_json(
                self.root / "TRAINING_PREFLIGHT_DECISION.json",
                label="training decision",
            )
            _write_json_atomic(
                deferred_path,
                {
                    "record_type": "glm52_teich_training_deferred_v1",
                    "run_id": self.run_id,
                    "reason": decision["reason"],
                    "teacher_cache_ready_sha256": _sha256_file(self.cache_dir / READY_NAME),
                    "decision_sha256": _sha256_file(
                        self.root / "TRAINING_PREFLIGHT_DECISION.json"
                    ),
                },
            )
            self.transition(
                CampaignPhase.TRAINING,
                inputs={"preflight": self.ledger().records[-1].record_sha256},
                outputs={"training_deferred": _sha256_file(deferred_path)},
            )
            return False
        self.ensure_compute_window()
        if not (self.adapter_dir / "glm52-low-rank-adapter-manifest.json").is_file():
            result = self._run_managed_child(
                self._training_command(preflight=False),
                env={
                    "CUDA_VISIBLE_DEVICES": "0",
                    "GLM52_TRAINING_BASELINE_JSON": str(
                        self.root / "training-baseline/training-baseline.json"
                    ),
                    "GLM52_TRAIN_PROGRESS": "1",
                    "GLM52_TRAIN_GRADIENT_CHECKPOINT": "1",
                },
                check=False,
                output=self.logs_dir / "training.log",
            )
            self.sync_all()
            if result.returncode == 75:
                raise SystemExit(75)
            if result.returncode != 0:
                raise RuntimeError(f"canonical adapter training failed: {result.returncode}")
        candidate_sha = self._audit_local_adapter()
        # Deliberately do not upload the final adapter.  Durable optimizer
        # checkpoints are authorized campaign state; model publication is not.
        self.transition(
            CampaignPhase.TRAINING,
            inputs={"preflight": self.ledger().records[-1].record_sha256},
            outputs={"local_candidate_identity": candidate_sha},
        )
        return True

    def evaluation(self, *, candidate_exists: bool) -> None:
        if self.phase_done(CampaignPhase.EVALUATION):
            return
        report = self.evaluation_dir / "candidate-evaluation.json"
        if candidate_exists:
            self.ensure_compute_window()
            if not report.is_file():
                self._run_managed_child(
                    [
                    sys.executable,
                    str(self.repo_root / "benchmarks/evaluate_glm52_teich_candidate.py"),
                    "--teacher-cache-dir",
                    str(self.cache_dir),
                    "--prompt-pack",
                    str(self.root / "teich-pack.json"),
                    "--frozen-prompt-pack",
                    str(self.root / "frozen-66.json"),
                    "--adapter-dir",
                    str(self.adapter_dir),
                    "--output",
                    str(report),
                    ],
                    env={
                        "CUDA_VISIBLE_DEVICES": "0",
                        "GLM52_TRAINING_BASELINE_JSON": str(
                            self.root / "training-baseline/training-baseline.json"
                        ),
                    },
                    output=self.logs_dir / "evaluation.log",
                )
            canonical = _read_json(report, label="canonical evaluation report")
            expansion = self.expansion_candidates(canonical)
            expansion_path = self.evaluation_dir / "expansion-campaign.json"
        else:
            _write_json_atomic(
                report,
                {
                    "record_type": "glm52_teich_candidate_evaluation_deferred_v1",
                    "automatic_acceptance": False,
                    "reason": "training_deferred",
                },
            )
            expansion = {
                "record_type": "glm52_teich_expansion_campaign_v1",
                "status": "not_started",
                "reason": "training_deferred",
                "layers": [],
            }
            expansion_path = self.evaluation_dir / "expansion-campaign.json"
            _write_json_atomic(expansion_path, expansion)
        self.transition(
            CampaignPhase.EVALUATION,
            inputs={"training": self.ledger().records[-1].record_sha256},
            outputs={
                "evaluation_report": _sha256_file(report),
                "expansion_report": _sha256_file(expansion_path),
            },
        )

    def drain(
        self,
        *,
        outcome: str | None = None,
        drain_reason: str | None = None,
    ) -> None:
        if self.phase_done(CampaignPhase.DRAINED):
            if self.kind is RuntimeCampaignKind.SKYPILOT:
                drained = self.root / "CAMPAIGN_DRAINED.json"
                if not drained.is_file():
                    self._aws(
                        "s3",
                        "cp",
                        f"{self.s3_root}/CAMPAIGN_DRAINED.json",
                        str(drained),
                        "--only-show-errors",
                    )
            return
        ledger = self.ledger()
        current_phase = ledger.current_phase
        if current_phase is None:
            raise RuntimeError("campaign cannot drain before BOOTSTRAP")
        if self.kind is RuntimeCampaignKind.SKYPILOT:
            self.sync_all(final=True)
            if outcome is None:
                if (self.root / "TRAINING_DEFERRED.json").is_file():
                    outcome = "training_deferred"
                    drain_reason = "training_deferred"
                else:
                    outcome = "completed"
                    drain_reason = "campaign_complete"
        payload = {
            "record_type": (
                "glm52_sky_campaign_drained_v2"
                if self.kind is RuntimeCampaignKind.SKYPILOT
                else "glm52_teich_campaign_drained_v1"
            ),
            "run_id": self.run_id,
            "prior_record_sha256": ledger.records[-1].record_sha256,
            "teacher_cache_ready_sha256": (
                _sha256_file(self.cache_dir / READY_NAME)
                if (self.cache_dir / READY_NAME).is_file()
                else None
            ),
            "automatic_model_promotion": False,
            "model_uploaded": False,
        }
        if self.kind is RuntimeCampaignKind.SKYPILOT:
            payload["outcome"] = outcome
            payload["last_completed_phase"] = current_phase.value
            payload["drain_reason"] = drain_reason
            payload["execution_deadline"] = self.execution_deadline.isoformat().replace(
                "+00:00", "Z"
            )
            payload["gpu_allocation_sha256"] = (
                self.runtime.gpu_allocation_sha256
            )
        else:
            payload["reservation_deadline"] = self.descriptor[
                "capacity_block_end"
            ]
        drained = self.root / "CAMPAIGN_DRAINED.json"
        _write_json_atomic(drained, payload)
        self.transition(
            CampaignPhase.DRAINED,
            inputs={
                (
                    "evaluation"
                    if current_phase is CampaignPhase.EVALUATION
                    else "prestop"
                ): ledger.records[-1].record_sha256
            },
            outputs={"drained": _sha256_file(drained)},
        )
        self.sync_all(final=True)
        self._aws(
            "s3",
            "cp",
            str(drained),
            f"{self.s3_root}/CAMPAIGN_DRAINED.json",
            "--only-show-errors",
        )

    def _production_sigterm(self, signum: int, _frame: object) -> None:
        """Convert systemd's fixed service SIGTERM into the status-75 path."""

        if signum != signal.SIGTERM:
            raise RuntimeError("production signal handler received wrong signal")
        if self._production_sigterm_received:
            return
        self._production_sigterm_received = True
        self.stop_file.touch(exist_ok=True)
        raise SystemExit(75)

    def run(self) -> int:
        production_signal_path = (
            self.kind is RuntimeCampaignKind.SKYPILOT
            and os.environ.get("GLM52_MANAGED_MODE") == "production"
        )
        previous_sigterm = None
        if production_signal_path:
            previous_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._production_sigterm)
        monitor = threading.Thread(target=self._deadline_monitor, daemon=True)
        try:
            self.restore()
            self._publish_heartbeat()
            monitor.start()
            self.bootstrap()
            self.cuda_gate()
            self.teacher()
            self.cache_audit()
            allowed = self.training_preflight()
            candidate_exists = self.training(allowed=allowed)
            self.evaluation(candidate_exists=candidate_exists)
            self.drain()
            return 0
        except SystemExit as error:
            self.sync_all(final=True)
            status = int(error.code or 0)
            if (
                status == 75
                and self.kind is RuntimeCampaignKind.SKYPILOT
                and self.stop_file.exists()
                and self.ledger_path.is_file()
                and self.ledger().current_phase is not CampaignPhase.DRAINED
            ):
                reason = (
                    "execution_window_prestop"
                    if datetime.now(timezone.utc)
                    >= self.execution_deadline - timedelta(minutes=60)
                    else "operator_graceful_stop"
                )
                self.drain(
                    outcome="resumable_deadline",
                    drain_reason=reason,
                )
                return 0
            return status
        except Exception as error:
            current_phase = self.ledger().current_phase if self.ledger_path.is_file() else None
            failure = {
                "record_type": "glm52_campaign_worker_failure_v1",
                "run_id": self.run_id,
                "observed_at": datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
                "phase": current_phase.value if current_phase is not None else None,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write_json_atomic(self.root / "CAMPAIGN_FAILED.json", failure)
            self.sync_all(final=True)
            raise
        finally:
            self._stop_monitor.set()
            if monitor.ident is not None:
                monitor.join(timeout=30)
            self._publish_heartbeat()
            if production_signal_path:
                signal.signal(signal.SIGTERM, previous_sigterm)


def _notify_systemd_ready() -> None:
    """Notify systemd only after production controller construction succeeds."""

    if os.environ.get("GLM52_MANAGED_MODE") != "production":
        return
    notify_socket = os.environ.get("NOTIFY_SOCKET")
    if not notify_socket:
        raise RuntimeError("production systemd notify socket is absent")
    address = (
        "\0" + notify_socket[1:]
        if notify_socket.startswith("@")
        else notify_socket
    )
    payload = b"READY=1\nSTATUS=authenticated controller constructed"
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        channel.connect(address)
        if channel.send(payload) != len(payload):
            raise RuntimeError("production systemd readiness send was partial")
    finally:
        channel.close()


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--descriptor",
        type=Path,
        default=Path(os.environ.get("CAMPAIGN_DESCRIPTOR", "/etc/keep-glm52/campaign.json")),
    )
    parser.add_argument("--root", type=Path, default=Path("/mnt/nvme/glm52-campaign"))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    controller = CampaignController(
        descriptor_path=args.descriptor,
        root=args.root,
        repo_root=args.repo_root,
    )
    _notify_systemd_ready()
    return controller.run()


if __name__ == "__main__":
    raise SystemExit(main())
