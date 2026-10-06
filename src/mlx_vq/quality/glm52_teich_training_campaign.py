"""Deterministic schedule and durable state for the Teich adapter campaign."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from safetensors.numpy import load as load_safetensors
from safetensors.numpy import save as save_safetensors

_HEX64 = re.compile(r"[0-9a-f]{64}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return _sha256_bytes(_canonical_bytes(payload))


def _require_sha(value: str, *, label: str) -> None:
    if _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise ValueError(f"refusing to replace immutable training object {path.name}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_bytes(payload) + b"\n"
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        payload = json.loads(path.read_bytes())
    except Exception as error:
        raise ValueError(f"{label} is malformed: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


@dataclass(frozen=True)
class TrainingWindow:
    prompt_id: str
    start: int
    end: int


@dataclass(frozen=True)
class TrainingSchedule:
    windows: tuple[TrainingWindow, ...]
    seed: int
    window_size: int
    epochs: int = 1

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                "record_type": "glm52_teich_training_schedule_v1",
                "seed": self.seed,
                "window_size": self.window_size,
                "epochs": self.epochs,
                "windows": [
                    {"prompt_id": item.prompt_id, "start": item.start, "end": item.end}
                    for item in self.windows
                ],
            }
        )


def build_training_schedule(
    manifest_entries: Sequence[Mapping[str, object]],
    *,
    window_size: int = 64,
    seed: int = 20260712,
) -> TrainingSchedule:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    windows: list[TrainingWindow] = []
    seen: set[str] = set()
    for entry in manifest_entries:
        prompt_id = entry.get("prompt_id")
        split = entry.get("split")
        tuning_eligible = entry.get("tuning_eligible")
        count = entry.get("supervised_position_count")
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in seen:
            raise ValueError("training schedule entries require unique prompt IDs")
        seen.add(prompt_id)
        if split not in ("train", "validation", "holdout"):
            raise ValueError("training schedule entry has an invalid split")
        if tuning_eligible != (split == "train"):
            raise ValueError("only train schedule entries may be tuning eligible")
        if type(count) is not int or count <= 0:
            raise ValueError("training schedule position counts must be positive integers")
        if split != "train":
            continue
        for start in range(0, count, window_size):
            windows.append(TrainingWindow(prompt_id, start, min(start + window_size, count)))
    if not windows:
        raise ValueError("training schedule must contain at least one train window")
    random.Random(seed).shuffle(windows)
    return TrainingSchedule(windows=tuple(windows), seed=seed, window_size=window_size)


def build_fixed_split_windows(
    manifest_entries: Sequence[Mapping[str, object]],
    *,
    split: str,
    window_size: int = 64,
    seed: int = 20260712,
    max_windows: int | None = None,
) -> tuple[TrainingWindow, ...]:
    if split not in ("validation", "holdout"):
        raise ValueError("fixed evaluation windows must use validation or holdout")
    windows: list[TrainingWindow] = []
    for entry in manifest_entries:
        if entry.get("split") != split:
            continue
        if entry.get("tuning_eligible") is not False:
            raise ValueError("evaluation entries must not be tuning eligible")
        prompt_id = entry.get("prompt_id")
        count = entry.get("supervised_position_count")
        if not isinstance(prompt_id, str) or type(count) is not int or count <= 0:
            raise ValueError("evaluation window entry is malformed")
        for start in range(0, count, window_size):
            windows.append(TrainingWindow(prompt_id, start, min(start + window_size, count)))
    if not windows:
        raise ValueError(f"{split} split contains no evaluation windows")
    random.Random(seed).shuffle(windows)
    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError("max_windows must be positive")
        windows = windows[:max_windows]
    return tuple(windows)


@dataclass(frozen=True)
class TrainingCheckpointConfig:
    local_checkpoint_dir: Path
    s3_checkpoint_prefix: str | None = None
    local_every_steps: int = 50
    s3_every_steps: int = 250
    s3_every_seconds: int = 300
    validation_every_steps: int = 2048
    stop_file: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_checkpoint_dir", Path(self.local_checkpoint_dir))
        if self.stop_file is not None:
            object.__setattr__(self, "stop_file", Path(self.stop_file))
        for name in (
            "local_every_steps",
            "s3_every_steps",
            "s3_every_seconds",
            "validation_every_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.s3_checkpoint_prefix is not None and not self.s3_checkpoint_prefix.startswith("s3://"):
            raise ValueError("s3_checkpoint_prefix must be an s3:// URI")

    def canonical_candidate(self) -> dict[str, object]:
        return {
            "layers": [77],
            "projections": ["gate_proj", "up_proj", "down_proj"],
            "rank": 4,
            "learning_rate": 0.2,
            "initialization_scale": 0.02,
            "gradient_clipping": 1.0,
            "top_k_kl_enabled": True,
            "top_k": 2048,
            "cka_layers": [77],
            "cka_balance": "dynamic_kl_relative",
            "router_kl_weight": 1.0,
            "router_entropy_beta": 0.01,
            "monte_carlo_expert_exploration": True,
            "target_nll_weight": 0.0,
            "top1_margin_weight": 0.0,
            "maximum_epochs": 1,
            "validation_every_steps": self.validation_every_steps,
            "early_stop_validation_regressions": 2,
        }

    def canonical_candidate_json(self) -> str:
        return _canonical_bytes(self.canonical_candidate()).decode("utf-8")


@dataclass(frozen=True)
class TrainingIdentity:
    run_id: str
    baseline_sha256: str
    teacher_manifest_sha256: str
    schedule_sha256: str
    training_config_sha256: str

    def __post_init__(self) -> None:
        if not self.run_id or "/" in self.run_id or ".." in self.run_id:
            raise ValueError("training run_id must be a path-safe identifier")
        for name in (
            "baseline_sha256",
            "teacher_manifest_sha256",
            "schedule_sha256",
            "training_config_sha256",
        ):
            _require_sha(getattr(self, name), label=name)

    def as_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "baseline_sha256": self.baseline_sha256,
            "teacher_manifest_sha256": self.teacher_manifest_sha256,
            "schedule_sha256": self.schedule_sha256,
            "training_config_sha256": self.training_config_sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class TrainingCheckpoint:
    params: dict[str, np.ndarray]
    completed_schedule_index: int
    losses: np.ndarray
    latest_gradient_norms: dict[str, float]
    best_validation_metric: float | None
    best_checkpoint_sha256: str | None
    consecutive_validation_regressions: int
    rng_state: dict[str, object]
    checkpoint_filename: str
    checkpoint_sha256: str
    previous_checkpoint_sha256: str
    ledger_record_filename: str
    ledger_record_sha256: str
    previous_ledger_record_sha256: str


class TrainingCheckpointStore:
    """Immutable training states with a hash-linked ledger and marker-last resume."""

    def __init__(self, root: str | Path, *, identity: TrainingIdentity):
        self.root = Path(root)
        self.identity = identity
        self.objects = self.root / "objects"
        self.ledger = self.root / "ledger"
        self.latest = self.root / "latest.json"

    @property
    def _checkpoint_genesis(self) -> str:
        return _canonical_sha256(
            {"record_type": "glm52_training_checkpoint_genesis_v1", "identity": self.identity.as_dict()}
        )

    @property
    def _ledger_genesis(self) -> str:
        return _canonical_sha256(
            {"record_type": "glm52_training_ledger_genesis_v1", "checkpoint": self._checkpoint_genesis}
        )

    def _publish_latest(self, payload: object) -> None:
        _write_json_atomic(self.latest, payload)

    def publish(
        self,
        *,
        params: Mapping[str, np.ndarray],
        completed_schedule_index: int,
        losses: np.ndarray,
        latest_gradient_norms: Mapping[str, float],
        best_validation_metric: float | None,
        best_checkpoint_sha256: str | None,
        consecutive_validation_regressions: int,
        rng_state: Mapping[str, object],
        best_is_current: bool = False,
    ) -> TrainingCheckpoint:
        if completed_schedule_index <= 0:
            raise ValueError("completed_schedule_index must be positive")
        normalized_params: dict[str, np.ndarray] = {}
        for name, value in sorted(params.items()):
            if not name:
                raise ValueError("training parameter names must be non-empty")
            array = np.ascontiguousarray(value)
            if array.dtype != np.float32:
                raise ValueError("training checkpoint parameters must be FP32")
            normalized_params[name] = array
        if not normalized_params:
            raise ValueError("training checkpoint parameters must not be empty")
        loss_array = np.ascontiguousarray(losses)
        if loss_array.dtype != np.float32 or loss_array.shape != (completed_schedule_index,):
            raise ValueError("training loss history must be FP32 with one value per completed step")
        if consecutive_validation_regressions < 0:
            raise ValueError("validation regression count must be non-negative")
        if best_is_current and best_checkpoint_sha256 is not None:
            raise ValueError("best_is_current cannot also name an older checkpoint")
        if best_checkpoint_sha256 is not None:
            _require_sha(best_checkpoint_sha256, label="best_checkpoint_sha256")
        previous: TrainingCheckpoint | None = None
        if self.latest.exists() or self.latest.is_symlink():
            previous = self.load_latest()
            if completed_schedule_index <= previous.completed_schedule_index:
                raise ValueError("training checkpoint must advance the schedule")
        previous_checkpoint = previous.checkpoint_sha256 if previous else self._checkpoint_genesis
        previous_record = previous.ledger_record_sha256 if previous else self._ledger_genesis
        previous_record_filename = previous.ledger_record_filename if previous else None
        tensors = {f"param::{name}": value for name, value in normalized_params.items()}
        tensors["losses"] = loss_array
        raw = save_safetensors(
            tensors,
            metadata={
                "record_type": "glm52_teich_training_checkpoint_v1",
                "identity_sha256": self.identity.sha256,
                "completed_schedule_index": str(completed_schedule_index),
            },
        )
        checkpoint_sha = _sha256_bytes(raw)
        if best_is_current:
            best_checkpoint_sha256 = checkpoint_sha
        checkpoint_filename = (
            f"step-{completed_schedule_index:09d}-{checkpoint_sha[:20]}.safetensors"
        )
        _write_immutable(self.objects / checkpoint_filename, raw)
        record_body: dict[str, object] = {
            "schema": "glm52_teich_training_ledger_v1",
            "identity": self.identity.as_dict(),
            "identity_sha256": self.identity.sha256,
            "completed_schedule_index": completed_schedule_index,
            "checkpoint_filename": checkpoint_filename,
            "checkpoint_sha256": checkpoint_sha,
            "previous_checkpoint_sha256": previous_checkpoint,
            "previous_ledger_record_filename": previous_record_filename,
            "previous_ledger_record_sha256": previous_record,
            "parameter_names": sorted(normalized_params),
            "latest_gradient_norms": {
                name: float(value) for name, value in sorted(latest_gradient_norms.items())
            },
            "best_validation_metric": best_validation_metric,
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "consecutive_validation_regressions": consecutive_validation_regressions,
            "rng_state": dict(rng_state),
        }
        ledger_sha = _canonical_sha256(record_body)
        record = {**record_body, "ledger_record_sha256": ledger_sha}
        record_filename = (
            f"record-{completed_schedule_index:09d}-{ledger_sha[:20]}.json"
        )
        _write_immutable(self.ledger / record_filename, _canonical_bytes(record) + b"\n")
        latest = {
            "schema": "glm52_teich_training_latest_v1",
            "identity_sha256": self.identity.sha256,
            "completed_schedule_index": completed_schedule_index,
            "checkpoint_filename": checkpoint_filename,
            "checkpoint_sha256": checkpoint_sha,
            "ledger_record_filename": record_filename,
            "ledger_record_sha256": ledger_sha,
        }
        self._publish_latest(latest)
        return self.load_latest()

    def load_latest(self) -> TrainingCheckpoint:
        marker = _load_json(self.latest, label="training latest marker")
        if marker.get("schema") != "glm52_teich_training_latest_v1":
            raise ValueError("training latest marker schema mismatch")
        if marker.get("identity_sha256") != self.identity.sha256:
            raise ValueError("training checkpoint identity does not match requested run")
        record_filename = marker.get("ledger_record_filename")
        if not isinstance(record_filename, str) or Path(record_filename).name != record_filename:
            raise ValueError("training ledger filename is invalid")
        record = _load_json(self.ledger / record_filename, label="training ledger record")
        body = dict(record)
        ledger_sha = body.pop("ledger_record_sha256", None)
        if ledger_sha != _canonical_sha256(body):
            raise ValueError("training ledger record SHA-256 mismatch")
        if record.get("identity") != self.identity.as_dict() or record.get(
            "identity_sha256"
        ) != self.identity.sha256:
            raise ValueError("training checkpoint ledger has a foreign identity")
        for key in (
            "completed_schedule_index",
            "checkpoint_filename",
            "checkpoint_sha256",
            "ledger_record_sha256",
        ):
            if marker.get(key) != record.get(key):
                raise ValueError("training latest marker and ledger disagree")
        previous_filename = record.get("previous_ledger_record_filename")
        if previous_filename is None:
            if record.get("previous_checkpoint_sha256") != self._checkpoint_genesis or record.get(
                "previous_ledger_record_sha256"
            ) != self._ledger_genesis:
                raise ValueError("training checkpoint genesis mismatch")
        else:
            if not isinstance(previous_filename, str) or Path(previous_filename).name != previous_filename:
                raise ValueError("training previous ledger filename is invalid")
            previous = _load_json(self.ledger / previous_filename, label="training previous ledger record")
            if previous.get("ledger_record_sha256") != record.get("previous_ledger_record_sha256"):
                raise ValueError("training previous ledger hash mismatch")
            if previous.get("checkpoint_sha256") != record.get("previous_checkpoint_sha256"):
                raise ValueError("training previous checkpoint hash mismatch")
            if int(previous.get("completed_schedule_index", -1)) >= int(record["completed_schedule_index"]):
                raise ValueError("training checkpoint chain does not advance")
        checkpoint_filename = record.get("checkpoint_filename")
        if not isinstance(checkpoint_filename, str) or Path(checkpoint_filename).name != checkpoint_filename:
            raise ValueError("training checkpoint filename is invalid")
        checkpoint_path = self.objects / checkpoint_filename
        if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
            raise ValueError("training checkpoint object is missing")
        raw = checkpoint_path.read_bytes()
        checkpoint_sha = _sha256_bytes(raw)
        if checkpoint_sha != record.get("checkpoint_sha256"):
            raise ValueError("training checkpoint SHA-256 mismatch")
        try:
            tensors = load_safetensors(raw)
        except Exception as error:
            raise ValueError(f"training checkpoint is malformed: {error}") from error
        names = record.get("parameter_names")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError("training parameter inventory is malformed")
        if set(tensors) != {"losses", *(f"param::{name}" for name in names)}:
            raise ValueError("training checkpoint tensor inventory mismatch")
        params = {name: tensors[f"param::{name}"] for name in names}
        if any(value.dtype != np.float32 for value in params.values()):
            raise ValueError("training checkpoint parameters must be FP32")
        losses = tensors["losses"]
        completed = record.get("completed_schedule_index")
        if type(completed) is not int or losses.dtype != np.float32 or losses.shape != (completed,):
            raise ValueError("training checkpoint loss history mismatch")
        gradient_norms = record.get("latest_gradient_norms")
        rng_state = record.get("rng_state")
        if not isinstance(gradient_norms, dict) or not isinstance(rng_state, dict):
            raise ValueError("training checkpoint runtime state is malformed")
        return TrainingCheckpoint(
            params=params,
            completed_schedule_index=completed,
            losses=losses,
            latest_gradient_norms={name: float(value) for name, value in gradient_norms.items()},
            best_validation_metric=(
                float(record["best_validation_metric"])
                if record.get("best_validation_metric") is not None
                else None
            ),
            best_checkpoint_sha256=(
                str(record["best_checkpoint_sha256"])
                if record.get("best_checkpoint_sha256") is not None
                else None
            ),
            consecutive_validation_regressions=int(record["consecutive_validation_regressions"]),
            rng_state=dict(rng_state),
            checkpoint_filename=checkpoint_filename,
            checkpoint_sha256=checkpoint_sha,
            previous_checkpoint_sha256=str(record["previous_checkpoint_sha256"]),
            ledger_record_filename=record_filename,
            ledger_record_sha256=str(record["ledger_record_sha256"]),
            previous_ledger_record_sha256=str(record["previous_ledger_record_sha256"]),
        )

    def load_best_params(self) -> dict[str, np.ndarray]:
        """Authenticate the full ancestor chain and return its recorded best state."""

        latest = self.load_latest()
        target_sha = latest.best_checkpoint_sha256 or latest.checkpoint_sha256
        record_filename: str | None = latest.ledger_record_filename
        expected_record_sha = latest.ledger_record_sha256
        expected_checkpoint_sha = latest.checkpoint_sha256
        while record_filename is not None:
            record = _load_json(self.ledger / record_filename, label="training ledger record")
            body = dict(record)
            record_sha = body.pop("ledger_record_sha256", None)
            if record_sha != expected_record_sha or record_sha != _canonical_sha256(body):
                raise ValueError("training best-checkpoint ledger chain is corrupt")
            if record.get("identity") != self.identity.as_dict() or record.get(
                "identity_sha256"
            ) != self.identity.sha256:
                raise ValueError("training best-checkpoint ledger has a foreign identity")
            if record.get("checkpoint_sha256") != expected_checkpoint_sha:
                raise ValueError("training best-checkpoint chain disagrees on checkpoint hash")
            if expected_checkpoint_sha == target_sha:
                filename = record.get("checkpoint_filename")
                if not isinstance(filename, str) or Path(filename).name != filename:
                    raise ValueError("training best checkpoint filename is invalid")
                path = self.objects / filename
                if path.is_symlink() or not path.is_file():
                    raise ValueError("training best checkpoint object is missing")
                raw = path.read_bytes()
                if _sha256_bytes(raw) != target_sha:
                    raise ValueError("training best checkpoint SHA-256 mismatch")
                try:
                    tensors = load_safetensors(raw)
                except Exception as error:
                    raise ValueError(f"training best checkpoint is malformed: {error}") from error
                names = record.get("parameter_names")
                if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                    raise ValueError("training best parameter inventory is malformed")
                expected_names = {"losses", *(f"param::{name}" for name in names)}
                if set(tensors) != expected_names:
                    raise ValueError("training best checkpoint tensor inventory mismatch")
                params = {name: tensors[f"param::{name}"] for name in names}
                if any(value.dtype != np.float32 for value in params.values()):
                    raise ValueError("training best checkpoint parameters must be FP32")
                return params
            previous_filename = record.get("previous_ledger_record_filename")
            if previous_filename is None:
                break
            if not isinstance(previous_filename, str) or Path(previous_filename).name != previous_filename:
                raise ValueError("training best-checkpoint previous ledger filename is invalid")
            record_filename = previous_filename
            expected_record_sha = str(record.get("previous_ledger_record_sha256"))
            expected_checkpoint_sha = str(record.get("previous_checkpoint_sha256"))
        raise ValueError("recorded best validation checkpoint is not in the authenticated chain")


@dataclass(frozen=True)
class ValidationState:
    best_metric: float | None = None
    best_checkpoint_sha256: str | None = None
    consecutive_regressions: int = 0

    @property
    def should_stop(self) -> bool:
        return self.consecutive_regressions >= 2

    def observe(self, *, metric: float, checkpoint_sha256: str) -> "ValidationState":
        _require_sha(checkpoint_sha256, label="validation checkpoint")
        if not np.isfinite(metric):
            raise ValueError("validation metric must be finite")
        if self.best_metric is None or metric < self.best_metric:
            return ValidationState(metric, checkpoint_sha256, 0)
        return ValidationState(
            self.best_metric,
            self.best_checkpoint_sha256,
            self.consecutive_regressions + 1,
        )


@dataclass(frozen=True)
class NumpyTrainingResult:
    params: dict[str, np.ndarray]
    losses: np.ndarray
    completed_schedule_index: int


def run_numpy_sgd_schedule(
    *,
    initial_params: Mapping[str, np.ndarray],
    schedule: TrainingSchedule,
    learning_rate: float,
    gradient_fn: Callable[[Mapping[str, np.ndarray], TrainingWindow, int], Mapping[str, np.ndarray]],
    checkpoint_store: TrainingCheckpointStore | None = None,
    checkpoint_every_steps: int = 50,
    stop_after_steps: int | None = None,
    resume: bool = False,
) -> NumpyTrainingResult:
    """Import-light SGD oracle used to prove exact schedule resume semantics."""

    if learning_rate <= 0 or checkpoint_every_steps <= 0:
        raise ValueError("learning rate and checkpoint cadence must be positive")
    params = {name: np.ascontiguousarray(value, dtype=np.float32).copy() for name, value in initial_params.items()}
    losses: list[np.float32] = []
    start = 0
    if resume:
        if checkpoint_store is None:
            raise ValueError("resume requires a checkpoint store")
        restored = checkpoint_store.load_latest()
        params = {name: value.copy() for name, value in restored.params.items()}
        losses = [np.float32(value) for value in restored.losses]
        start = restored.completed_schedule_index
    for index in range(start, len(schedule.windows)):
        window = schedule.windows[index]
        gradients = {
            name: np.ascontiguousarray(value, dtype=np.float32)
            for name, value in gradient_fn(params, window, index).items()
        }
        if set(gradients) != set(params) or any(gradients[name].shape != params[name].shape for name in params):
            raise ValueError("gradient inventory must exactly match parameters")
        loss = np.float32(
            sum(np.mean(np.square(value, dtype=np.float32), dtype=np.float32) for value in gradients.values())
        )
        params = {
            name: np.ascontiguousarray(
                params[name] - np.float32(learning_rate) * gradients[name], dtype=np.float32
            )
            for name in params
        }
        losses.append(loss)
        completed = index + 1
        if checkpoint_store is not None and (
            completed % checkpoint_every_steps == 0 or completed == len(schedule.windows)
        ):
            norms = {
                name: float(np.linalg.norm(value.astype(np.float64)))
                for name, value in gradients.items()
            }
            checkpoint_store.publish(
                params=params,
                completed_schedule_index=completed,
                losses=np.asarray(losses, dtype=np.float32),
                latest_gradient_norms=norms,
                best_validation_metric=None,
                best_checkpoint_sha256=None,
                consecutive_validation_regressions=0,
                rng_state={"seed": schedule.seed, "offset": completed},
            )
        if stop_after_steps is not None and completed >= stop_after_steps:
            if checkpoint_store is not None and completed % checkpoint_every_steps != 0:
                checkpoint_store.publish(
                    params=params,
                    completed_schedule_index=completed,
                    losses=np.asarray(losses, dtype=np.float32),
                    latest_gradient_norms={
                        name: float(np.linalg.norm(value.astype(np.float64)))
                        for name, value in gradients.items()
                    },
                    best_validation_metric=None,
                    best_checkpoint_sha256=None,
                    consecutive_validation_regressions=0,
                    rng_state={"seed": schedule.seed, "offset": completed},
                )
            break
    return NumpyTrainingResult(
        params=params,
        losses=np.asarray(losses, dtype=np.float32),
        completed_schedule_index=len(losses),
    )


__all__ = [
    "NumpyTrainingResult",
    "TrainingCheckpoint",
    "TrainingCheckpointConfig",
    "TrainingCheckpointStore",
    "TrainingIdentity",
    "TrainingSchedule",
    "TrainingWindow",
    "ValidationState",
    "build_training_schedule",
    "build_fixed_split_windows",
    "run_numpy_sgd_schedule",
]
