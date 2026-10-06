"""Authenticated, immutable checkpoints for the Teich teacher campaign.

The store deliberately models S3 publication semantics on the local filesystem:
checkpoint objects and hash-linked ledger records are immutable, while
``latest.json`` is the only mutable pointer and is published last.  A crash at
any earlier point therefore leaves the previously authenticated pointer as the
resume authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
from safetensors.numpy import load as load_safetensors
from safetensors.numpy import save as save_safetensors

__all__ = [
    "CheckpointValidationError",
    "DeadlinePolicy",
    "ForwardCheckpoint",
    "LocalLMHeadSliceStore",
    "LocalTeacherCheckpointStore",
    "TeacherCheckpointConfig",
    "TeacherRunIdentity",
    "should_checkpoint_after_layer",
]

_HEX64 = re.compile(r"[0-9a-f]{64}")
_LATEST_SCHEMA = "glm52_teich_teacher_latest_v1"
_LEDGER_SCHEMA = "glm52_teich_teacher_checkpoint_ledger_v1"
_OBJECT_SCHEMA = "glm52_teich_teacher_forward_checkpoint_v1"


class CheckpointValidationError(ValueError):
    """A checkpoint cannot be authenticated as part of the requested run."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    return _sha256_bytes(_canonical_json_bytes(payload))


def _require_sha256(value: str, *, label: str) -> None:
    if _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


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
        raise CheckpointValidationError(f"refusing to replace immutable object {path.name}")
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


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json_bytes(payload) + b"\n"
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CheckpointValidationError(f"{label} must be a regular non-symlink file")
    try:
        value = json.loads(path.read_bytes())
    except Exception as error:
        raise CheckpointValidationError(f"{label} is malformed: {error}") from error
    if not isinstance(value, dict):
        raise CheckpointValidationError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True)
class TeacherRunIdentity:
    """Immutable inputs that define one teacher-cache run."""

    run_id: str
    model_sha256: str
    non_vq_package_sha256: str
    prompt_pack_sha256: str
    code_sha256: str
    generation_config_sha256: str

    def __post_init__(self) -> None:
        if not self.run_id or "/" in self.run_id or ".." in self.run_id:
            raise ValueError("run_id must be a non-empty path-safe identifier")
        for name in (
            "model_sha256",
            "non_vq_package_sha256",
            "prompt_pack_sha256",
            "code_sha256",
            "generation_config_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)

    def as_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "model_sha256": self.model_sha256,
            "non_vq_package_sha256": self.non_vq_package_sha256,
            "prompt_pack_sha256": self.prompt_pack_sha256,
            "code_sha256": self.code_sha256,
            "generation_config_sha256": self.generation_config_sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class TeacherCheckpointConfig:
    local_checkpoint_dir: Path
    identity: TeacherRunIdentity
    s3_checkpoint_prefix: str | None = None
    checkpoint_every_layers: int = 4
    stop_file: Path | None = None
    capacity_block_deadline: datetime | None = None
    execution_deadline: datetime | None = None
    resume: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_checkpoint_dir", Path(self.local_checkpoint_dir))
        if self.stop_file is not None:
            object.__setattr__(self, "stop_file", Path(self.stop_file))
        if self.checkpoint_every_layers <= 0:
            raise ValueError("checkpoint_every_layers must be positive")
        if self.s3_checkpoint_prefix is not None and not self.s3_checkpoint_prefix.startswith(
            "s3://"
        ):
            raise ValueError("s3_checkpoint_prefix must be an s3:// URI")
        if (
            self.capacity_block_deadline is not None
            and self.execution_deadline is not None
        ):
            raise ValueError(
                "checkpoint config accepts at most one deadline authority"
            )
        deadline = self.effective_deadline
        if deadline is not None and deadline.tzinfo is None:
            raise ValueError("checkpoint deadline must be timezone-aware")

    @property
    def effective_deadline(self) -> datetime | None:
        return self.execution_deadline or self.capacity_block_deadline


@dataclass(frozen=True)
class DeadlinePolicy:
    """Pre-stop thresholds using exactly one orchestrator deadline authority."""

    capacity_block_end: datetime | None = None
    execution_deadline: datetime | None = None
    stop_file: Path | None = None

    def __post_init__(self) -> None:
        deadlines = [
            value
            for value in (self.capacity_block_end, self.execution_deadline)
            if value is not None
        ]
        if len(deadlines) != 1:
            raise ValueError("deadline policy requires exactly one deadline authority")
        if deadlines[0].tzinfo is None:
            raise ValueError("deadline authority must be timezone-aware")
        if self.stop_file is not None:
            object.__setattr__(self, "stop_file", Path(self.stop_file))

    @property
    def deadline(self) -> datetime:
        value = self.execution_deadline or self.capacity_block_end
        assert value is not None
        return value

    def _now(self, now: datetime | None) -> datetime:
        value = now or datetime.now(timezone.utc)
        if value.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        return value

    def stop_assigning(self, *, now: datetime | None = None) -> bool:
        return self._now(now) >= self.deadline - timedelta(minutes=60)

    def force_terminate(self, *, now: datetime | None = None) -> bool:
        return self._now(now) >= self.deadline - timedelta(minutes=50)

    def final_sync_due(self, *, now: datetime | None = None) -> bool:
        return self._now(now) >= self.deadline - timedelta(minutes=35)

    def voluntary_termination_due(self, *, now: datetime | None = None) -> bool:
        minutes = 30 if self.execution_deadline is not None else 32
        return self._now(now) >= self.deadline - timedelta(minutes=minutes)

    def stop_requested(self, *, now: datetime | None = None) -> bool:
        return bool(self.stop_file and self.stop_file.exists()) or self.stop_assigning(now=now)


def should_checkpoint_after_layer(
    next_layer: int,
    *,
    every_layers: int = 4,
    force: bool = False,
) -> bool:
    if next_layer < 0:
        raise ValueError("next_layer must be non-negative")
    if every_layers <= 0:
        raise ValueError("every_layers must be positive")
    return force or (next_layer > 0 and next_layer % every_layers == 0)


@dataclass(frozen=True)
class ForwardCheckpoint:
    next_layer: int
    hidden_bf16_bits: np.ndarray
    prev_topk_indices: np.ndarray | None
    ordered_prompt_ids: tuple[str, ...]
    ordered_token_hashes: tuple[str, ...]
    batch_shape: tuple[int, int]
    valid_mask_sha256: str
    generation_parameters: dict[str, object]
    checkpoint_filename: str
    checkpoint_sha256: str
    previous_checkpoint_sha256: str
    ledger_record_filename: str
    ledger_record_sha256: str
    previous_ledger_record_sha256: str


class LocalTeacherCheckpointStore:
    """Local implementation of the immutable-object/marker-last contract."""

    def __init__(self, root: str | Path, *, identity: TeacherRunIdentity):
        self.root = Path(root)
        self.identity = identity
        self.objects_dir = self.root / "objects"
        self.ledger_dir = self.root / "ledger"
        self.latest_path = self.root / "latest.json"

    @property
    def _checkpoint_genesis_sha256(self) -> str:
        return _canonical_sha256(
            {
                "record_type": "glm52_teich_teacher_checkpoint_genesis_v1",
                "run_identity_sha256": self.identity.sha256,
            }
        )

    @property
    def _ledger_genesis_sha256(self) -> str:
        return _canonical_sha256(
            {
                "record_type": "glm52_teich_teacher_ledger_genesis_v1",
                "checkpoint_genesis_sha256": self._checkpoint_genesis_sha256,
            }
        )

    def _publish_latest(self, payload: dict[str, object]) -> None:
        _write_json_atomic(self.latest_path, payload)

    def publish_forward(
        self,
        *,
        next_layer: int,
        hidden_bf16_bits: np.ndarray,
        prev_topk_indices: np.ndarray | None,
        ordered_prompt_ids: tuple[str, ...],
        ordered_token_hashes: tuple[str, ...],
        batch_shape: tuple[int, int],
        valid_mask_sha256: str,
        generation_parameters: Mapping[str, object],
    ) -> ForwardCheckpoint:
        if next_layer <= 0:
            raise ValueError("next_layer must be positive")
        hidden = np.ascontiguousarray(hidden_bf16_bits)
        if hidden.dtype != np.uint16 or hidden.ndim != 3:
            raise ValueError("hidden_bf16_bits must be rank-3 uint16 BF16 words")
        if tuple(hidden.shape[:2]) != tuple(batch_shape):
            raise ValueError("hidden batch dimensions do not match batch_shape")
        dsa = None
        tensors: dict[str, np.ndarray] = {"hidden_bf16_bits": hidden}
        if prev_topk_indices is not None:
            dsa = np.ascontiguousarray(prev_topk_indices)
            if dsa.dtype != np.int32 or dsa.ndim != 3:
                raise ValueError("prev_topk_indices must be rank-3 int32")
            if tuple(dsa.shape[:2]) != tuple(batch_shape):
                raise ValueError("DSA state batch dimensions do not match batch_shape")
            tensors["prev_topk_indices"] = dsa
        if len(ordered_prompt_ids) != batch_shape[0] or len(ordered_token_hashes) != batch_shape[0]:
            raise ValueError("prompt identity count must match checkpoint batch")
        for digest in ordered_token_hashes:
            _require_sha256(digest, label="ordered_token_hash")
        _require_sha256(valid_mask_sha256, label="valid_mask_sha256")

        previous: ForwardCheckpoint | None = None
        if self.latest_path.exists() or self.latest_path.is_symlink():
            previous = self.load_latest()
            if next_layer <= previous.next_layer:
                raise CheckpointValidationError("checkpoint next_layer must advance the chain")
        previous_checkpoint_sha256 = (
            previous.checkpoint_sha256 if previous else self._checkpoint_genesis_sha256
        )
        previous_ledger_record_sha256 = (
            previous.ledger_record_sha256 if previous else self._ledger_genesis_sha256
        )
        previous_ledger_record_filename = previous.ledger_record_filename if previous else None

        metadata = {
            "record_type": _OBJECT_SCHEMA,
            "run_identity_sha256": self.identity.sha256,
            "next_layer": str(next_layer),
            "previous_checkpoint_sha256": previous_checkpoint_sha256,
        }
        raw_checkpoint = save_safetensors(tensors, metadata=metadata)
        checkpoint_sha256 = _sha256_bytes(raw_checkpoint)
        checkpoint_filename = f"forward-{next_layer:05d}-{checkpoint_sha256[:20]}.safetensors"
        _write_immutable(self.objects_dir / checkpoint_filename, raw_checkpoint)

        record_body: dict[str, object] = {
            "schema": _LEDGER_SCHEMA,
            "run_identity": self.identity.as_dict(),
            "run_identity_sha256": self.identity.sha256,
            "next_layer": next_layer,
            "ordered_prompt_ids": list(ordered_prompt_ids),
            "ordered_token_hashes": list(ordered_token_hashes),
            "batch_shape": list(batch_shape),
            "hidden_shape": list(hidden.shape),
            "prev_topk_shape": list(dsa.shape) if dsa is not None else None,
            "valid_mask_sha256": valid_mask_sha256,
            "generation_parameters": dict(generation_parameters),
            "checkpoint_filename": checkpoint_filename,
            "checkpoint_sha256": checkpoint_sha256,
            "previous_checkpoint_sha256": previous_checkpoint_sha256,
            "previous_ledger_record_filename": previous_ledger_record_filename,
            "previous_ledger_record_sha256": previous_ledger_record_sha256,
        }
        ledger_record_sha256 = _canonical_sha256(record_body)
        record = {**record_body, "ledger_record_sha256": ledger_record_sha256}
        ledger_record_filename = f"record-{next_layer:05d}-{ledger_record_sha256[:20]}.json"
        _write_immutable(
            self.ledger_dir / ledger_record_filename,
            _canonical_json_bytes(record) + b"\n",
        )

        latest = {
            "schema": _LATEST_SCHEMA,
            "run_identity_sha256": self.identity.sha256,
            "next_layer": next_layer,
            "checkpoint_filename": checkpoint_filename,
            "checkpoint_sha256": checkpoint_sha256,
            "ledger_record_filename": ledger_record_filename,
            "ledger_record_sha256": ledger_record_sha256,
        }
        self._publish_latest(latest)
        return self.load_latest()

    def load_latest(self) -> ForwardCheckpoint:
        if not self.latest_path.exists():
            raise CheckpointValidationError("latest checkpoint marker does not exist")
        latest = _read_json_object(self.latest_path, label="latest checkpoint marker")
        expected_latest_keys = {
            "schema",
            "run_identity_sha256",
            "next_layer",
            "checkpoint_filename",
            "checkpoint_sha256",
            "ledger_record_filename",
            "ledger_record_sha256",
        }
        if set(latest) != expected_latest_keys or latest.get("schema") != _LATEST_SCHEMA:
            raise CheckpointValidationError("latest checkpoint marker schema mismatch")
        if latest.get("run_identity_sha256") != self.identity.sha256:
            raise CheckpointValidationError("latest checkpoint marker has a foreign run identity")
        record_filename = latest.get("ledger_record_filename")
        if not isinstance(record_filename, str) or Path(record_filename).name != record_filename:
            raise CheckpointValidationError("latest checkpoint ledger filename is invalid")
        record = self._validate_record_chain(record_filename)
        for key in (
            "next_layer",
            "checkpoint_filename",
            "checkpoint_sha256",
            "ledger_record_sha256",
        ):
            if latest.get(key) != record.get(key):
                raise CheckpointValidationError("latest marker and ledger record disagree")
        return self._load_checkpoint(record, ledger_record_filename=record_filename)

    def _validate_record_chain(self, record_filename: str) -> dict[str, object]:
        seen: set[str] = set()
        child_next_layer: int | None = None
        current_filename: str | None = record_filename
        current_record: dict[str, object] | None = None
        while current_filename is not None:
            if current_filename in seen:
                raise CheckpointValidationError("checkpoint ledger contains a cycle")
            seen.add(current_filename)
            record = _read_json_object(
                self.ledger_dir / current_filename,
                label=f"checkpoint ledger record {current_filename}",
            )
            record_sha256 = record.get("ledger_record_sha256")
            body = dict(record)
            body.pop("ledger_record_sha256", None)
            if record_sha256 != _canonical_sha256(body):
                raise CheckpointValidationError("checkpoint ledger record SHA-256 mismatch")
            if record.get("schema") != _LEDGER_SCHEMA:
                raise CheckpointValidationError("checkpoint ledger record schema mismatch")
            if record.get("run_identity") != self.identity.as_dict() or record.get(
                "run_identity_sha256"
            ) != self.identity.sha256:
                raise CheckpointValidationError("checkpoint ledger has a foreign run identity")
            next_layer = record.get("next_layer")
            if not isinstance(next_layer, int) or next_layer <= 0:
                raise CheckpointValidationError("checkpoint ledger next_layer is invalid")
            if child_next_layer is not None and next_layer >= child_next_layer:
                raise CheckpointValidationError("checkpoint ledger chain is noncontiguous")
            if current_record is None:
                current_record = record
            previous_filename = record.get("previous_ledger_record_filename")
            previous_record_sha = record.get("previous_ledger_record_sha256")
            previous_checkpoint_sha = record.get("previous_checkpoint_sha256")
            if previous_filename is None:
                if previous_record_sha != self._ledger_genesis_sha256:
                    raise CheckpointValidationError("checkpoint ledger genesis hash mismatch")
                if previous_checkpoint_sha != self._checkpoint_genesis_sha256:
                    raise CheckpointValidationError("checkpoint genesis hash mismatch")
            else:
                if not isinstance(previous_filename, str) or Path(previous_filename).name != previous_filename:
                    raise CheckpointValidationError("checkpoint previous ledger filename is invalid")
                previous = _read_json_object(
                    self.ledger_dir / previous_filename,
                    label=f"checkpoint ledger record {previous_filename}",
                )
                if previous.get("ledger_record_sha256") != previous_record_sha:
                    raise CheckpointValidationError("checkpoint ledger previous-record hash mismatch")
                if previous.get("checkpoint_sha256") != previous_checkpoint_sha:
                    raise CheckpointValidationError("checkpoint ledger previous-checkpoint hash mismatch")
            child_next_layer = next_layer
            current_filename = previous_filename if isinstance(previous_filename, str) else None
        assert current_record is not None
        return current_record

    def _load_checkpoint(
        self,
        record: Mapping[str, object],
        *,
        ledger_record_filename: str,
    ) -> ForwardCheckpoint:
        filename = record.get("checkpoint_filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise CheckpointValidationError("checkpoint filename is invalid")
        checkpoint_path = self.objects_dir / filename
        if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
            raise CheckpointValidationError("checkpoint object must be a regular non-symlink file")
        raw = checkpoint_path.read_bytes()
        checkpoint_sha256 = _sha256_bytes(raw)
        if checkpoint_sha256 != record.get("checkpoint_sha256"):
            raise CheckpointValidationError("checkpoint SHA-256 mismatch")
        try:
            tensors = load_safetensors(raw)
        except Exception as error:
            raise CheckpointValidationError(f"checkpoint object is malformed: {error}") from error
        expected_names = {"hidden_bf16_bits"}
        if record.get("prev_topk_shape") is not None:
            expected_names.add("prev_topk_indices")
        if set(tensors) != expected_names:
            raise CheckpointValidationError("checkpoint tensor inventory mismatch")
        hidden = tensors["hidden_bf16_bits"]
        if hidden.dtype != np.uint16 or list(hidden.shape) != record.get("hidden_shape"):
            raise CheckpointValidationError("checkpoint hidden tensor contract mismatch")
        dsa = tensors.get("prev_topk_indices")
        if dsa is not None and (
            dsa.dtype != np.int32 or list(dsa.shape) != record.get("prev_topk_shape")
        ):
            raise CheckpointValidationError("checkpoint DSA tensor contract mismatch")
        prompt_ids = record.get("ordered_prompt_ids")
        token_hashes = record.get("ordered_token_hashes")
        batch_shape = record.get("batch_shape")
        generation_parameters = record.get("generation_parameters")
        if not isinstance(prompt_ids, list) or not all(isinstance(x, str) for x in prompt_ids):
            raise CheckpointValidationError("checkpoint prompt IDs are malformed")
        if not isinstance(token_hashes, list) or not all(isinstance(x, str) for x in token_hashes):
            raise CheckpointValidationError("checkpoint token hashes are malformed")
        if not isinstance(batch_shape, list) or len(batch_shape) != 2 or not all(
            isinstance(x, int) for x in batch_shape
        ):
            raise CheckpointValidationError("checkpoint batch shape is malformed")
        if not isinstance(generation_parameters, dict):
            raise CheckpointValidationError("checkpoint generation parameters are malformed")
        return ForwardCheckpoint(
            next_layer=int(record["next_layer"]),
            hidden_bf16_bits=hidden,
            prev_topk_indices=dsa,
            ordered_prompt_ids=tuple(prompt_ids),
            ordered_token_hashes=tuple(token_hashes),
            batch_shape=(batch_shape[0], batch_shape[1]),
            valid_mask_sha256=str(record["valid_mask_sha256"]),
            generation_parameters=dict(generation_parameters),
            checkpoint_filename=filename,
            checkpoint_sha256=checkpoint_sha256,
            previous_checkpoint_sha256=str(record["previous_checkpoint_sha256"]),
            ledger_record_filename=ledger_record_filename,
            ledger_record_sha256=str(record["ledger_record_sha256"]),
            previous_ledger_record_sha256=str(record["previous_ledger_record_sha256"]),
        )


class LocalLMHeadSliceStore:
    """Authenticated per-session LM-head slices with immutable completion markers."""

    _TENSOR_DTYPES = {
        "topk_logit_ids": np.dtype(np.int32),
        "topk_logit_values": np.dtype(np.float16),
        "logsumexp": np.dtype(np.float32),
        "tail_mass": np.dtype(np.float16),
    }

    def __init__(
        self,
        root: str | Path,
        *,
        identity: TeacherRunIdentity,
        prompt_id: str,
        token_ids_sha256: str,
        positions: np.ndarray,
        target_token_ids: np.ndarray,
        top_k: int,
    ):
        self.root = Path(root)
        self.identity = identity
        self.prompt_id = prompt_id
        self.token_ids_sha256 = token_ids_sha256
        self.positions = np.ascontiguousarray(positions)
        self.target_token_ids = np.ascontiguousarray(target_token_ids)
        self.top_k = top_k
        if not prompt_id or "/" in prompt_id or ".." in prompt_id:
            raise ValueError("prompt_id must be a non-empty path-safe identifier")
        _require_sha256(token_ids_sha256, label="token_ids_sha256")
        if self.positions.dtype != np.int32 or self.positions.ndim != 1:
            raise ValueError("positions must be rank-1 int32")
        if self.target_token_ids.dtype != np.int32 or self.target_token_ids.shape != self.positions.shape:
            raise ValueError("target_token_ids must be rank-1 int32 matching positions")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.objects_dir = self.root / "objects"
        self.markers_dir = self.root / "markers"

    @property
    def _session_identity(self) -> dict[str, object]:
        return {
            "record_type": "glm52_teich_lm_head_session_v1",
            "run_identity_sha256": self.identity.sha256,
            "prompt_id": self.prompt_id,
            "token_ids_sha256": self.token_ids_sha256,
            "positions_sha256": _sha256_bytes(self.positions.tobytes()),
            "target_token_ids_sha256": _sha256_bytes(self.target_token_ids.tobytes()),
            "supervised_position_count": int(self.positions.size),
            "top_k": self.top_k,
        }

    @property
    def _session_identity_sha256(self) -> str:
        return _canonical_sha256(self._session_identity)

    def _marker_path(self, start: int, end: int) -> Path:
        return self.markers_dir / f"slice-{start:09d}-{end:09d}.json"

    def _publish_slice_marker(self, path: Path, payload: Mapping[str, object]) -> None:
        _write_immutable(path, _canonical_json_bytes(payload) + b"\n")

    def publish_slice(
        self,
        *,
        start: int,
        end: int,
        tensors: Mapping[str, np.ndarray],
    ) -> None:
        if not 0 <= start < end <= self.positions.size:
            raise ValueError("slice range is outside the supervised positions")
        if set(tensors) != set(self._TENSOR_DTYPES):
            raise ValueError("LM-head slice tensor inventory mismatch")
        row_count = end - start
        normalized: dict[str, np.ndarray] = {}
        for name, expected_dtype in self._TENSOR_DTYPES.items():
            value = np.ascontiguousarray(tensors[name])
            if value.dtype != expected_dtype:
                raise ValueError(f"{name} dtype must be {expected_dtype}")
            expected_shape = (row_count, self.top_k) if name.startswith("topk_") else (row_count,)
            if value.shape != expected_shape:
                raise ValueError(f"{name} shape must be {expected_shape}")
            normalized[name] = value
        completed = self._load_completed()
        for existing_start, existing_end, _, _ in completed:
            if max(start, existing_start) < min(end, existing_end):
                raise CheckpointValidationError("LM-head slice overlaps a completed range")

        raw = save_safetensors(
            normalized,
            metadata={
                "record_type": "glm52_teich_lm_head_slice_v1",
                "session_identity_sha256": self._session_identity_sha256,
                "start": str(start),
                "end": str(end),
            },
        )
        object_sha256 = _sha256_bytes(raw)
        object_filename = f"slice-{start:09d}-{end:09d}-{object_sha256[:20]}.safetensors"
        _write_immutable(self.objects_dir / object_filename, raw)
        marker_body: dict[str, object] = {
            "schema": "glm52_teich_lm_head_slice_marker_v1",
            "session_identity": self._session_identity,
            "session_identity_sha256": self._session_identity_sha256,
            "start": start,
            "end": end,
            "object_filename": object_filename,
            "object_sha256": object_sha256,
            "tensor_names": sorted(normalized),
            "tensor_shapes": {name: list(value.shape) for name, value in normalized.items()},
            "tensor_dtypes": {name: str(value.dtype) for name, value in normalized.items()},
        }
        marker_sha256 = _canonical_sha256(marker_body)
        marker = {**marker_body, "marker_sha256": marker_sha256}
        self._publish_slice_marker(self._marker_path(start, end), marker)

    def _load_completed(
        self,
    ) -> list[tuple[int, int, dict[str, np.ndarray], dict[str, object]]]:
        if not self.markers_dir.exists():
            return []
        if self.markers_dir.is_symlink() or not self.markers_dir.is_dir():
            raise CheckpointValidationError("LM-head marker root must be a regular directory")
        completed: list[tuple[int, int, dict[str, np.ndarray], dict[str, object]]] = []
        for marker_path in sorted(self.markers_dir.glob("slice-*.json")):
            marker = _read_json_object(marker_path, label=f"LM-head marker {marker_path.name}")
            marker_sha256 = marker.get("marker_sha256")
            body = dict(marker)
            body.pop("marker_sha256", None)
            if marker_sha256 != _canonical_sha256(body):
                raise CheckpointValidationError("LM-head slice marker SHA-256 mismatch")
            if marker.get("schema") != "glm52_teich_lm_head_slice_marker_v1":
                raise CheckpointValidationError("LM-head slice marker schema mismatch")
            if marker.get("session_identity") != self._session_identity or marker.get(
                "session_identity_sha256"
            ) != self._session_identity_sha256:
                raise CheckpointValidationError("LM-head slice has a foreign session identity")
            start = marker.get("start")
            end = marker.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= self.positions.size:
                raise CheckpointValidationError("LM-head slice range is invalid")
            if marker_path != self._marker_path(start, end):
                raise CheckpointValidationError("LM-head slice marker filename mismatch")
            object_filename = marker.get("object_filename")
            if not isinstance(object_filename, str) or Path(object_filename).name != object_filename:
                raise CheckpointValidationError("LM-head slice object filename is invalid")
            object_path = self.objects_dir / object_filename
            if object_path.is_symlink() or not object_path.is_file():
                raise CheckpointValidationError("LM-head slice object is missing")
            raw = object_path.read_bytes()
            if _sha256_bytes(raw) != marker.get("object_sha256"):
                raise CheckpointValidationError("LM-head slice object SHA-256 mismatch")
            try:
                tensors = load_safetensors(raw)
            except Exception as error:
                raise CheckpointValidationError(f"LM-head slice object is malformed: {error}") from error
            expected_names = set(self._TENSOR_DTYPES)
            if set(tensors) != expected_names or marker.get("tensor_names") != sorted(expected_names):
                raise CheckpointValidationError("LM-head slice tensor inventory mismatch")
            for name, expected_dtype in self._TENSOR_DTYPES.items():
                expected_shape = (end - start, self.top_k) if name.startswith("topk_") else (end - start,)
                if tensors[name].dtype != expected_dtype or tensors[name].shape != expected_shape:
                    raise CheckpointValidationError(f"LM-head slice {name} contract mismatch")
            completed.append((start, end, tensors, marker))
        completed.sort(key=lambda item: item[0])
        prior_end = 0
        for index, (start, end, _, _) in enumerate(completed):
            if index and start < prior_end:
                raise CheckpointValidationError("LM-head completed slices overlap")
            prior_end = end
        return completed

    def missing_ranges(self, *, slice_size: int) -> tuple[tuple[int, int], ...]:
        if slice_size <= 0:
            raise ValueError("slice_size must be positive")
        covered = np.zeros(self.positions.size, dtype=np.bool_)
        for start, end, _, _ in self._load_completed():
            covered[start:end] = True
        missing: list[tuple[int, int]] = []
        cursor = 0
        while cursor < self.positions.size:
            if covered[cursor]:
                cursor += 1
                continue
            end = min(cursor + slice_size, self.positions.size)
            while end > cursor and np.any(covered[cursor:end]):
                end -= 1
            missing.append((cursor, end))
            cursor = end
        return tuple(missing)

    def assemble(self) -> dict[str, np.ndarray]:
        completed = self._load_completed()
        cursor = 0
        pieces: dict[str, list[np.ndarray]] = {name: [] for name in self._TENSOR_DTYPES}
        for start, end, tensors, _ in completed:
            if start != cursor:
                raise CheckpointValidationError("LM-head slices do not form a complete contiguous inventory")
            for name in pieces:
                pieces[name].append(tensors[name])
            cursor = end
        if cursor != self.positions.size:
            raise CheckpointValidationError("LM-head slices do not cover all supervised positions")
        return {
            "positions": self.positions.copy(),
            "target_token_ids": self.target_token_ids.copy(),
            **{name: np.concatenate(parts, axis=0) for name, parts in pieces.items()},
        }
