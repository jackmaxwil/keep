from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask
from safetensors import safe_open

from mlx_vq.io.authenticated_artifacts import AuthenticatedFile
from mlx_vq.io.source_safetensors import (
    MAX_SAFETENSORS_HEADER_BYTES,
    SafetensorsFileHeader,
    SafetensorsTensorHeader,
    read_safetensors_tensor_mlx,
)
from mlx_vq.quality.mlx_surrogate import (
    RouteLocalSwitchLinearSurrogate,
    switch_linear_layer_sidecar,
)
from mlx_vq.quality.glm52_teich_training_campaign import (
    TrainingCheckpointConfig,
    TrainingCheckpointStore,
    TrainingSchedule,
    TrainingWindow,
    ValidationState,
)


SUPPORTED_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
MAX_ADAPTER_LAYERS = 8
ADAPTER_MANIFEST = "glm52-low-rank-adapter-manifest.json"
SIDECAR_DIRECTORY = "sidecars"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: str, *, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal SHA-256") from error


def _read_regular_file(path: Path, *, label: str) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    before = path.stat()
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"{label} changed while it was read")
    return raw


def _parse_safetensors_bytes(raw: bytes, *, label: str) -> SafetensorsFileHeader:
    if len(raw) < 8:
        raise ValueError(f"{label} is not a valid safetensors file")
    header_size = struct.unpack("<Q", raw[:8])[0]
    if header_size <= 0 or header_size > MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(f"{label} has an invalid safetensors header size")
    payload_offset = 8 + header_size
    if payload_offset > len(raw):
        raise ValueError(f"{label} has a truncated safetensors header")
    try:
        raw_header = json.loads(raw[8:payload_offset])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} has an invalid safetensors header") from error
    if not isinstance(raw_header, dict):
        raise ValueError(f"{label} safetensors header must be an object")
    tensors: dict[str, SafetensorsTensorHeader] = {}
    intervals = []
    for tensor_name, tensor_info in raw_header.items():
        if tensor_name == "__metadata__":
            continue
        if not isinstance(tensor_info, dict):
            raise ValueError(f"{label} tensor header for {tensor_name!r} must be an object")
        dtype = tensor_info.get("dtype")
        shape = tensor_info.get("shape")
        offsets = tensor_info.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or payload_offset + offsets[1] > len(raw)
        ):
            raise ValueError(f"{label} has an invalid tensor header for {tensor_name!r}")
        header = SafetensorsTensorHeader(
            dtype=dtype,
            shape=tuple(shape),
            data_offsets=(offsets[0], offsets[1]),
        )
        tensors[str(tensor_name)] = header
        intervals.append(header.data_offsets)
    ordered = sorted(intervals)
    if any(previous[1] > current[0] for previous, current in zip(ordered, ordered[1:])):
        raise ValueError(f"{label} safetensors payload intervals overlap")
    declared_payload_bytes = max((end for _start, end in intervals), default=0)
    if payload_offset + declared_payload_bytes != len(raw):
        raise ValueError(f"{label} safetensors payload extent does not match its header")
    return SafetensorsFileHeader(header_size=header_size, tensors=tensors)


def _read_fp32_tensor_from_bytes(
    raw: bytes,
    header: SafetensorsFileHeader,
    tensor_name: str,
    *,
    label: str,
) -> mx.array:
    tensor = header.tensors[tensor_name]
    if tensor.dtype != "F32":
        raise ValueError("adapter low-rank tensors must be FP32")
    start, end = tensor.data_offsets
    tensor_raw = raw[header.payload_offset + start : header.payload_offset + end]
    if len(tensor_raw) != tensor.element_count * 4:
        raise ValueError(f"{label} tensor {tensor_name!r} byte extent does not match its shape")
    values = np.frombuffer(tensor_raw, dtype="<f4").reshape(tensor.shape).copy()
    return mx.array(values, dtype=mx.float32)


def _safe_child(root: Path, relative: str, *, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"{label} must be a normalized cache-relative path")
    child = root.joinpath(*pure.parts)
    if child.resolve().parent != root.resolve() and root.resolve() not in child.resolve().parents:
        raise ValueError(f"{label} escapes its artifact root")
    return child


@dataclass(frozen=True)
class PreparedGLM52TeacherRow:
    row_index: int
    prompt_id: str
    split: str
    tuning_eligible: bool
    input_token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    teacher_logits: mx.array | None
    teacher_manifest_body_sha256: str
    teacher_shard_sha256: str
    non_release_waiver: bool
    topk_logit_ids: mx.array | None = None
    topk_logit_values: mx.array | None = None
    tail_mass: mx.array | None = None
    teacher_probe_hidden_states: Mapping[int, mx.array] | None = None
    teacher_router_topk_ids: Mapping[int, mx.array] | None = None
    teacher_router_topk_weights: Mapping[int, mx.array] | None = None


@dataclass(frozen=True)
class PreparedGLM52TeichTeacherRow(PreparedGLM52TeacherRow):
    """Schema-v3 sparse row; its split authority is train/validation/holdout."""

    cache_schema_version: int = 3


@dataclass(frozen=True)
class PreparedGLM52TeichDiskRow:
    """Authenticated schema-v3 metadata with windowed safetensors access.

    The full teacher signal is deliberately absent from this object.  A fresh
    cache audit authenticates the shard before construction, and every window
    load rejects replacement or mutation of that audited local file.
    """

    row_index: int
    prompt_id: str
    split: Literal["train", "validation", "holdout"]
    tuning_eligible: bool
    input_token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    teacher_manifest_body_sha256: str
    teacher_shard_sha256: str
    non_release_waiver: bool
    shard_path: Path
    top_k: int
    hidden_size: int
    audited_file_stat: tuple[int, int, int, int]
    router_layers: tuple[int, ...] = tuple(range(70, 78))
    cache_schema_version: int = 3

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation", "holdout"):
            raise ValueError("Teich disk row split must be train, validation, or holdout")
        if self.tuning_eligible != (self.split == "train"):
            raise ValueError("only Teich train disk rows may be tuning eligible")
        if not self.positions or len(self.positions) != len(self.target_token_ids):
            raise ValueError("Teich disk row positions and targets must be non-empty and aligned")
        if any(position < 0 or position + 1 >= len(self.input_token_ids) for position in self.positions):
            raise ValueError("Teich disk row positions exceed the input token authority")
        if self.top_k <= 0 or self.hidden_size <= 0:
            raise ValueError("Teich disk row tensor dimensions must be positive")
        _require_sha256(self.teacher_manifest_body_sha256, label="Teich manifest body")
        _require_sha256(self.teacher_shard_sha256, label="Teich teacher shard")

    def _assert_audited_file_unchanged(self) -> None:
        path = Path(self.shard_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("audited Teich teacher shard is no longer a regular file")
        stat = path.stat()
        current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if current != self.audited_file_stat:
            raise ValueError("audited Teich teacher shard changed before window load")

    def load_window(self, start: int, end: int) -> PreparedGLM52TeichTeacherRow:
        if not 0 <= start < end <= len(self.positions):
            raise ValueError("Teich disk window is outside the supervised positions")
        self._assert_audited_file_unchanged()
        with safe_open(str(self.shard_path), framework="np") as tensors:
            positions = np.asarray(tensors.get_slice("positions")[start:end], dtype=np.int32)
            targets = np.asarray(
                tensors.get_slice("target_token_ids")[start:end], dtype=np.int32
            )
            topk_ids = np.asarray(
                tensors.get_slice("topk_logit_ids")[start:end], dtype=np.int32
            )
            topk_values = np.asarray(
                tensors.get_slice("topk_logit_values")[start:end], dtype=np.float16
            )
            tail_mass = np.asarray(
                tensors.get_slice("tail_mass")[start:end], dtype=np.float16
            )
            probe = np.asarray(
                tensors.get_slice("layer_77_hidden_probe")[start:end], dtype=np.float16
            )
            router_ids = np.asarray(
                tensors.get_slice("router_top8_expert_ids")[:, start:end],
                dtype=np.int32,
            )
            router_weights = np.asarray(
                tensors.get_slice("router_top8_normalized_weights")[:, start:end],
                dtype=np.float16,
            )
        expected_positions = self.positions[start:end]
        expected_targets = self.target_token_ids[start:end]
        if tuple(int(value) for value in positions) != expected_positions:
            raise ValueError("Teich shard positions drifted from its manifest authority")
        if tuple(int(value) for value in targets) != expected_targets:
            raise ValueError("Teich shard targets drifted from its manifest authority")
        count = end - start
        if topk_ids.shape != (count, self.top_k) or topk_values.shape != topk_ids.shape:
            raise ValueError("Teich disk window top-K tensor shape mismatch")
        if tail_mass.shape != (count,) or probe.shape != (count, self.hidden_size):
            raise ValueError("Teich disk window tail/probe tensor shape mismatch")
        if router_ids.shape != (len(self.router_layers), count, 8):
            raise ValueError("Teich disk window router ID tensor shape mismatch")
        if router_weights.shape != router_ids.shape:
            raise ValueError("Teich disk window router weight tensor shape mismatch")
        row = PreparedGLM52TeichTeacherRow(
            row_index=self.row_index,
            prompt_id=self.prompt_id,
            split=self.split,
            tuning_eligible=self.tuning_eligible,
            input_token_ids=self.input_token_ids,
            positions=expected_positions,
            target_token_ids=expected_targets,
            teacher_logits=None,
            teacher_manifest_body_sha256=self.teacher_manifest_body_sha256,
            teacher_shard_sha256=self.teacher_shard_sha256,
            non_release_waiver=self.non_release_waiver,
            topk_logit_ids=mx.array(topk_ids, dtype=mx.int32),
            topk_logit_values=mx.array(topk_values, dtype=mx.float16),
            tail_mass=mx.array(tail_mass, dtype=mx.float16),
            teacher_probe_hidden_states={77: mx.array(probe, dtype=mx.float16)},
            teacher_router_topk_ids={
                layer: mx.array(router_ids[offset], dtype=mx.int32)
                for offset, layer in enumerate(self.router_layers)
            },
            teacher_router_topk_weights={
                layer: mx.array(router_weights[offset], dtype=mx.float16)
                for offset, layer in enumerate(self.router_layers)
            },
        )
        validate_teich_teacher_row(row)
        return row

    def load_distillation_window(
        self, start: int, end: int
    ) -> PreparedGLM52TeichTeacherRow:
        """Load only logits/targets for memory-bounded held-out evaluation."""

        if not 0 <= start < end <= len(self.positions):
            raise ValueError("Teich disk window is outside the supervised positions")
        self._assert_audited_file_unchanged()
        with safe_open(str(self.shard_path), framework="np") as tensors:
            positions = np.asarray(tensors.get_slice("positions")[start:end], dtype=np.int32)
            targets = np.asarray(
                tensors.get_slice("target_token_ids")[start:end], dtype=np.int32
            )
            topk_ids = np.asarray(
                tensors.get_slice("topk_logit_ids")[start:end], dtype=np.int32
            )
            topk_values = np.asarray(
                tensors.get_slice("topk_logit_values")[start:end], dtype=np.float16
            )
            tail_mass = np.asarray(
                tensors.get_slice("tail_mass")[start:end], dtype=np.float16
            )
        expected_positions = self.positions[start:end]
        expected_targets = self.target_token_ids[start:end]
        if tuple(int(value) for value in positions) != expected_positions:
            raise ValueError("Teich shard positions drifted from its manifest authority")
        if tuple(int(value) for value in targets) != expected_targets:
            raise ValueError("Teich shard targets drifted from its manifest authority")
        count = end - start
        if (
            topk_ids.shape != (count, self.top_k)
            or topk_values.shape != topk_ids.shape
            or tail_mass.shape != (count,)
        ):
            raise ValueError("Teich distillation window tensor shape mismatch")
        row = PreparedGLM52TeichTeacherRow(
            row_index=self.row_index,
            prompt_id=self.prompt_id,
            split=self.split,
            tuning_eligible=self.tuning_eligible,
            input_token_ids=self.input_token_ids,
            positions=expected_positions,
            target_token_ids=expected_targets,
            teacher_logits=None,
            teacher_manifest_body_sha256=self.teacher_manifest_body_sha256,
            teacher_shard_sha256=self.teacher_shard_sha256,
            non_release_waiver=self.non_release_waiver,
            topk_logit_ids=mx.array(topk_ids, dtype=mx.int32),
            topk_logit_values=mx.array(topk_values, dtype=mx.float16),
            tail_mass=mx.array(tail_mass, dtype=mx.float16),
        )
        validate_teich_teacher_row(row)
        return row


@dataclass(frozen=True)
class AdapterTrainingConfig:
    layers: tuple[int, ...]
    projections: tuple[str, ...]
    rank: int
    steps: int
    learning_rate: float = 1.0e-3
    low_rank_init_scale: float = 1.0e-3
    grad_clip_norm: float | None = 1.0
    target_nll_weight: float = 0.0
    teacher_top1_margin_weight: float = 0.0
    teacher_top1_margin: float = 0.0
    teacher_top1_competitor_token_ids: tuple[tuple[int, ...], ...] | None = None
    teacher_top1_include_hardest_competitor: bool = True
    tail_kld_weight: float = 0.0
    seed: int = 20260711
    surrogate_output_chunk_size: int = 256
    topk: int | None = None
    cka_enabled: bool = False
    cka_layers: tuple[int, ...] = ()
    router_kl_weight: float = 0.0
    router_entropy_beta: float = 0.0
    mc_expert_explore: bool = False

    def __post_init__(self) -> None:
        if not self.layers or len(self.layers) > MAX_ADAPTER_LAYERS:
            raise ValueError(f"layers must contain between 1 and {MAX_ADAPTER_LAYERS} entries")
        if len(set(self.layers)) != len(self.layers) or any(layer < 0 for layer in self.layers):
            raise ValueError("layers must be unique non-negative integers")
        if not self.projections or len(set(self.projections)) != len(self.projections):
            raise ValueError("projections must be a non-empty unique sequence")
        unknown = sorted(set(self.projections) - set(SUPPORTED_PROJECTIONS))
        if unknown:
            raise ValueError(f"unsupported projections: {unknown}")
        if self.rank <= 0 or self.steps <= 0:
            raise ValueError("rank and steps must be positive")
        if self.learning_rate <= 0 or self.low_rank_init_scale <= 0:
            raise ValueError("learning_rate and low_rank_init_scale must be positive")
        if self.surrogate_output_chunk_size <= 0:
            raise ValueError("surrogate_output_chunk_size must be positive")
        if self.topk is not None and not 2048 <= self.topk <= 8192:
            raise ValueError("topk must be between 2048 and 8192 for production caches")
        if self.cka_enabled:
            if not self.cka_layers:
                raise ValueError("cka_layers must be non-empty when CKA is enabled")
            if tuple(sorted(self.cka_layers)) != self.cka_layers or len(set(self.cka_layers)) != len(
                self.cka_layers
            ):
                raise ValueError("cka_layers must be ordered unique layer indices")
            if any(layer < 0 for layer in self.cka_layers):
                raise ValueError("cka_layers must contain non-negative layer indices")
        if self.router_kl_weight < 0 or self.router_entropy_beta < 0:
            raise ValueError("router KL weight and entropy beta must be non-negative")
        if type(self.mc_expert_explore) is not bool:
            raise ValueError("mc_expert_explore must be boolean")


@dataclass(frozen=True)
class AdapterTrainingResult:
    params: Mapping[str, mx.array]
    losses: mx.array
    gradient_norms: Mapping[str, mx.array]


@dataclass(frozen=True)
class ValidatedGLM52Adapter:
    adapter_dir: Path
    parent_candidate_identity_sha256: str
    teacher_manifest_body_sha256: str
    manifest_body_sha256: str
    candidate_identity_sha256: str
    params: Mapping[str, mx.array]
    sidecar_paths: Mapping[tuple[int, str], Path]
    release_eligible: bool


@dataclass(frozen=True)
class ValidatedGLM52TrainingBaseline:
    model: Any
    candidate_identity_sha256: str

    def __post_init__(self) -> None:
        if self.model is None:
            raise ValueError("validated GLM52 training baseline model must not be None")
        _require_sha256(
            self.candidate_identity_sha256,
            label="validated GLM52 training baseline candidate identity",
        )


@dataclass(frozen=True)
class ProjectionInventoryAttestation:
    projection_count: int
    projection_metadata_sha256: str

    def __post_init__(self) -> None:
        if type(self.projection_count) is not int or self.projection_count <= 0:
            raise ValueError("projection inventory count must be a positive integer")
        _require_sha256(
            self.projection_metadata_sha256,
            label="projection metadata inventory",
        )


def attest_glm52_projection_inventory(
    projection_map: Mapping[tuple[int, str], Any],
) -> ProjectionInventoryAttestation:
    if not projection_map:
        raise ValueError("projection inventory must not be empty")
    metadata = []
    for layer, projection_name in sorted(projection_map):
        if type(layer) is not int or layer < 0 or projection_name not in SUPPORTED_PROJECTIONS:
            raise ValueError("projection inventory contains an invalid routed projection key")
        projection = projection_map[(layer, projection_name)]
        record = {
            "layer": layer,
            "projection": projection_name,
            "num_experts": getattr(projection, "num_experts", None),
            "input_dims": getattr(projection, "input_dims", None),
            "output_dims": getattr(projection, "output_dims", None),
            "group_size": getattr(projection, "group_size", None),
            "code_bits": getattr(projection, "code_bits", None),
        }
        if any(type(record[field]) is not int or record[field] <= 0 for field in (
            "num_experts",
            "input_dims",
            "output_dims",
            "group_size",
            "code_bits",
        )):
            raise ValueError("projection inventory metadata must contain positive integers")
        metadata.append(record)
    return ProjectionInventoryAttestation(
        projection_count=len(metadata),
        projection_metadata_sha256=_sha256_bytes(_canonical_bytes(metadata)),
    )


def validate_tuning_row(row: PreparedGLM52TeacherRow) -> None:
    if isinstance(row, PreparedGLM52TeichTeacherRow):
        if row.split != "train" or row.tuning_eligible is not True:
            raise ValueError("GLM52 Teich adapter tuning is train-only; validation/holdout rows are forbidden")
    elif row.split != "selection" or row.tuning_eligible is not True:
        raise ValueError("GLM52 adapter tuning is selection-only; report/holdout rows are forbidden")
    if row.teacher_logits is not None:
        if row.teacher_logits.ndim != 2:
            raise ValueError("teacher logits must have shape [positions, vocab]")
        if tuple(row.teacher_logits.shape[:1]) != (len(row.positions),):
            raise ValueError("teacher logits position count must match positions")
    topk_signals = (row.topk_logit_ids, row.topk_logit_values, row.tail_mass)
    if any(value is not None for value in topk_signals):
        if any(value is None for value in topk_signals):
            raise ValueError("top-K teacher signal requires IDs, values, and tail mass")
        assert row.topk_logit_ids is not None
        assert row.topk_logit_values is not None
        assert row.tail_mass is not None
        if (
            row.topk_logit_ids.ndim != 2
            or row.topk_logit_values.shape != row.topk_logit_ids.shape
            or row.topk_logit_ids.shape[0] != len(row.positions)
            or row.tail_mass.shape != (len(row.positions),)
        ):
            raise ValueError("top-K teacher signal shapes must be [positions, K] plus [positions]")
        if row.topk_logit_values.dtype != mx.float16 or row.tail_mass.dtype != mx.float16:
            raise ValueError("top-K logit values and tail mass must be FP16")
        if row.topk_logit_ids.dtype != mx.int32:
            raise ValueError("top-K logit IDs must be INT32")
        if not bool(mx.all((row.tail_mass >= 0) & (row.tail_mass < 1)).item()):
            raise ValueError("top-K tail mass must be in [0, 1)")
    if row.teacher_logits is None and row.topk_logit_ids is None:
        raise ValueError("teacher row requires dense logits or a top-K teacher signal")
    if len(row.positions) != len(row.target_token_ids):
        raise ValueError("positions and target token IDs must have equal lengths")
    if row.teacher_probe_hidden_states is not None:
        for layer, hidden in row.teacher_probe_hidden_states.items():
            if type(layer) is not int or layer < 0 or hidden.ndim != 2 or hidden.shape[0] != len(row.positions):
                raise ValueError("teacher CKA probes must map layers to [positions, hidden] tensors")
            if hidden.dtype != mx.float16:
                raise ValueError("teacher CKA probes must be FP16")
    router_ids = row.teacher_router_topk_ids
    router_weights = row.teacher_router_topk_weights
    if (router_ids is None) != (router_weights is None):
        raise ValueError("teacher router targets require both IDs and normalized weights")
    if router_ids is not None and router_weights is not None:
        if set(router_ids) != set(router_weights):
            raise ValueError("teacher router ID and weight layers must match")
        for layer in router_ids:
            ids = router_ids[layer]
            weights = router_weights[layer]
            if (
                type(layer) is not int
                or layer < 0
                or ids.ndim != 2
                or ids.shape != weights.shape
                or ids.shape[0] != len(row.positions)
                or ids.shape[1] > 8
            ):
                raise ValueError("teacher router targets must map layers to [positions, top-8] tensors")
            if weights.dtype != mx.float16:
                raise ValueError("teacher router weights must be FP16")
            if ids.dtype != mx.int32:
                raise ValueError("teacher router IDs must be INT32")
            weight_sums = mx.sum(weights.astype(mx.float32), axis=-1)
            if not bool(mx.all(weights >= 0).item()) or not bool(
                mx.all(mx.abs(weight_sums - 1.0) <= 1.0e-3).item()
            ):
                raise ValueError("teacher router weights must be normalized per position")


def prepare_glm52_teacher_rows(
    *,
    teacher_cache_dir: Path,
    prompt_pack_path: Path,
    split: Literal["selection"] = "selection",
    max_rows: int | None,
    max_positions: int | None,
    row_indices: tuple[int, ...] | None = None,
    allow_non_release_teacher_cache: bool = False,
    expected_manifest_body_sha256: str | None = None,
    test_only_synthetic: bool = False,
) -> list[PreparedGLM52TeacherRow]:
    if split != "selection":
        raise ValueError("GLM52 adapter tuning is selection-only")
    root = Path(teacher_cache_dir)
    manifest_path = root / "glm52-teacher-cache-fp32-manifest.json"
    manifest_raw = _read_regular_file(manifest_path, label="teacher cache manifest")
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, dict):
        raise ValueError("teacher cache manifest must be an object")
    recorded_body = manifest.get("manifest_body_sha256")
    body = dict(manifest)
    body.pop("manifest_body_sha256", None)
    actual_body = _sha256_bytes(_canonical_bytes(body))
    if recorded_body != actual_body:
        raise ValueError("teacher cache manifest body SHA-256 mismatch")

    production_sections = ("producer", "source_evidence", "non_vq_package", "prompt_authority")
    if test_only_synthetic:
        release_eligible = False
    else:
        if expected_manifest_body_sha256 is None:
            raise ValueError("production teacher cache requires an externally supplied manifest identity")
        _require_sha256(expected_manifest_body_sha256, label="expected teacher manifest body")
        if actual_body != expected_manifest_body_sha256:
            raise ValueError("teacher cache manifest body does not match external authority")
        if not all(isinstance(manifest.get(section), dict) for section in production_sections):
            raise ValueError("teacher cache production schema is incomplete")
        is_distill_signal_contract = (
            manifest.get("schema_version") == 2
            and manifest.get("record_type") == "glm52_teacher_signal_cache"
        )
        if is_distill_signal_contract:
            # Generation/auditing of the richer cache is a separate GPU step.
            # Consumption is still fail-closed: the caller pins the canonical
            # manifest body, and every shard is hashed and schema-checked below.
            release_eligible = manifest.get("release_eligible") is True
        else:
            from mlx_vq.quality.glm52_teacher_cache import (
                GLM52TeacherCacheContract,
                audit_glm52_teacher_cache,
            )

            contract = GLM52TeacherCacheContract.from_frozen_prompt_pack(
                prompt_pack_path,
                producer=manifest["producer"],
                source_evidence=manifest["source_evidence"],
                non_vq_package=manifest["non_vq_package"],
            )
            audit = audit_glm52_teacher_cache(root, contract=contract)
            if audit.manifest_body_sha256 != actual_body:
                raise ValueError("strict teacher cache audit returned a different manifest identity")
            release_eligible = audit.release_eligible

    if not release_eligible and not allow_non_release_teacher_cache:
        raise ValueError("teacher cache is non-release; an explicit waiver is required")
    prompt_raw = _read_regular_file(Path(prompt_pack_path), label="prompt pack")
    prompt_sha = _sha256_bytes(prompt_raw)
    declared_prompt_sha = manifest.get("prompt_pack_sha256")
    if declared_prompt_sha is None and isinstance(manifest.get("prompt_authority"), dict):
        declared_prompt_sha = manifest["prompt_authority"].get("file_sha256")
    if declared_prompt_sha != prompt_sha:
        raise ValueError("prompt pack SHA-256 does not match teacher manifest authority")
    prompt_payload = json.loads(prompt_raw)
    raw_prompt_rows = prompt_payload.get("prompt_rows") if isinstance(prompt_payload, dict) else None
    if not isinstance(raw_prompt_rows, list):
        raise ValueError("prompt pack must contain prompt_rows")
    prompts: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, prompt in enumerate(raw_prompt_rows):
        if not isinstance(prompt, dict) or not isinstance(prompt.get("prompt_id"), str):
            raise ValueError(f"prompt_rows[{index}] must contain a prompt_id")
        prompt_id = prompt["prompt_id"]
        if prompt_id in prompts:
            raise ValueError(f"duplicate prompt ID {prompt_id!r}")
        prompts[prompt_id] = (index, prompt)

    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("teacher manifest must contain an ordered shards list")
    selected_entries = []
    for shard in raw_shards:
        if not isinstance(shard, dict):
            raise ValueError("teacher shard entries must be objects")
        if shard.get("split") == "selection":
            selected_entries.append(shard)
        elif shard.get("tuning_eligible") is True:
            raise ValueError("report/holdout teacher shard cannot be tuning eligible")
    if row_indices is not None:
        if len(set(row_indices)) != len(row_indices):
            raise ValueError("row_indices must be unique")
        try:
            selected_entries = [selected_entries[index] for index in row_indices]
        except IndexError as error:
            raise ValueError("row index is out of range for selection rows") from error
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        selected_entries = selected_entries[:max_rows]

    rows: list[PreparedGLM52TeacherRow] = []
    for selection_index, shard in enumerate(selected_entries):
        prompt_id = shard.get("prompt_id")
        if prompt_id not in prompts:
            raise ValueError(f"teacher shard prompt {prompt_id!r} is absent from prompt pack")
        prompt_index, prompt = prompts[prompt_id]
        if prompt.get("split") != "selection" or prompt.get("tuning_eligible") is not True:
            raise ValueError("GLM52 adapter tuning is selection-only")
        token_ids = prompt.get("encoded_token_ids")
        if not isinstance(token_ids, list) or len(token_ids) < 2 or any(type(v) is not int for v in token_ids):
            raise ValueError(f"prompt {prompt_id!r} has invalid encoded_token_ids")
        declared_token_sha = shard.get("token_ids_sha256")
        if declared_token_sha is not None and declared_token_sha != _sha256_bytes(_canonical_bytes(token_ids)):
            raise ValueError(f"prompt {prompt_id!r} token-ID SHA-256 mismatch")
        relative = shard.get("relative_path")
        if not isinstance(relative, str):
            raise ValueError("teacher shard relative_path must be a string")
        shard_path = _safe_child(root, relative, label="teacher shard relative_path")
        shard_raw = _read_regular_file(shard_path, label="teacher logit shard")
        shard_sha = _sha256_bytes(shard_raw)
        if shard_sha != shard.get("file_sha256"):
            raise ValueError(f"teacher shard {prompt_id!r} file SHA-256 mismatch")
        position_count = len(token_ids) - 1
        tensor_name = shard.get("tensor_name")
        logits: mx.array | None = None
        if tensor_name is not None or "topk" not in shard:
            tensor_name = "logits" if tensor_name is None else tensor_name
            logits = read_safetensors_tensor_mlx(shard_path, tensor_name).astype(mx.float32)
            if logits.ndim != 2 or tuple(logits.shape) != tuple(shard.get("shape", ())):
                raise ValueError(f"teacher shard {prompt_id!r} logits shape mismatch")
            if shard.get("dtype") != "F32":
                raise ValueError(f"teacher shard {prompt_id!r} dense logits must be FP32")
            if logits.shape[0] != position_count:
                raise ValueError(f"teacher shard {prompt_id!r} position count mismatch")

        topk_ids: mx.array | None = None
        topk_values: mx.array | None = None
        tail_mass: mx.array | None = None
        topk_meta = shard.get("topk")
        if topk_meta is not None:
            if not isinstance(topk_meta, dict):
                raise ValueError("teacher shard topk metadata must be an object")
            k = topk_meta.get("k")
            if type(k) is not int or k <= 0 or (not test_only_synthetic and not 2048 <= k <= 8192):
                raise ValueError("teacher shard topk must be 2048..8192 in production")
            vocab_size = topk_meta.get("vocab_size")
            if type(vocab_size) is not int or vocab_size < k:
                raise ValueError("teacher shard topk vocab_size must be at least K")
            names = (
                topk_meta.get("ids_tensor"),
                topk_meta.get("values_tensor"),
                topk_meta.get("tail_mass_tensor"),
            )
            if any(not isinstance(name, str) for name in names):
                raise ValueError("teacher shard topk tensor names must be strings")
            topk_ids = read_safetensors_tensor_mlx(shard_path, names[0])
            topk_values = read_safetensors_tensor_mlx(shard_path, names[1])
            tail_mass = read_safetensors_tensor_mlx(shard_path, names[2])
            if (
                topk_ids.shape != (position_count, k)
                or topk_values.shape != (position_count, k)
                or tail_mass.shape != (position_count,)
                or topk_ids.dtype != mx.int32
                or topk_values.dtype != mx.float16
                or tail_mass.dtype != mx.float16
            ):
                raise ValueError("teacher shard top-K tensors violate the [positions, K] FP16 contract")

        teacher_probes: dict[int, mx.array] = {}
        raw_probes = shard.get("cka_probes", [])
        if not isinstance(raw_probes, list):
            raise ValueError("teacher shard cka_probes must be a list")
        for probe in raw_probes:
            if not isinstance(probe, dict) or type(probe.get("layer")) is not int:
                raise ValueError("teacher CKA probe metadata must contain an integer layer")
            layer = probe["layer"]
            if layer in teacher_probes or not isinstance(probe.get("tensor_name"), str):
                raise ValueError("teacher CKA probe layers must be unique with tensor names")
            hidden = read_safetensors_tensor_mlx(shard_path, probe["tensor_name"])
            if (
                probe.get("dtype") != "F16"
                or hidden.dtype != mx.float16
                or hidden.ndim != 2
                or hidden.shape[0] != position_count
                or tuple(hidden.shape) != tuple(probe.get("shape", ()))
            ):
                raise ValueError("teacher CKA probe violates its FP16 [positions, hidden] contract")
            teacher_probes[layer] = hidden

        teacher_router_ids: dict[int, mx.array] = {}
        teacher_router_weights: dict[int, mx.array] = {}
        raw_router = shard.get("router_targets", [])
        if not isinstance(raw_router, list):
            raise ValueError("teacher shard router_targets must be a list")
        for target in raw_router:
            if not isinstance(target, dict) or type(target.get("layer")) is not int:
                raise ValueError("teacher router target metadata must contain an integer layer")
            layer = target["layer"]
            if layer in teacher_router_ids:
                raise ValueError("teacher router target layers must be unique")
            ids_name = target.get("ids_tensor")
            weights_name = target.get("weights_tensor")
            if not isinstance(ids_name, str) or not isinstance(weights_name, str):
                raise ValueError("teacher router target tensor names must be strings")
            ids = read_safetensors_tensor_mlx(shard_path, ids_name)
            weights = read_safetensors_tensor_mlx(shard_path, weights_name)
            if (
                target.get("weights_dtype") != "F16"
                or ids.ndim != 2
                or ids.shape != weights.shape
                or ids.shape[0] != position_count
                or ids.shape[1] > 8
                or ids.dtype != mx.int32
                or weights.dtype != mx.float16
            ):
                raise ValueError("teacher router target violates its [positions, top-8] FP16 contract")
            teacher_router_ids[layer] = ids
            teacher_router_weights[layer] = weights
        input_token_ids = token_ids
        if max_positions is not None:
            if max_positions <= 0:
                raise ValueError("max_positions must be positive")
            position_count = min(position_count, max_positions)
            if logits is not None:
                logits = logits[:position_count]
            if topk_ids is not None and topk_values is not None and tail_mass is not None:
                topk_ids = topk_ids[:position_count]
                topk_values = topk_values[:position_count]
                tail_mass = tail_mass[:position_count]
            teacher_probes = {layer: value[:position_count] for layer, value in teacher_probes.items()}
            teacher_router_ids = {
                layer: value[:position_count] for layer, value in teacher_router_ids.items()
            }
            teacher_router_weights = {
                layer: value[:position_count] for layer, value in teacher_router_weights.items()
            }
            # Only the first `position_count` positions are supervised, and each
            # depends causally only on tokens up to it — so forwarding just the
            # first position_count+1 tokens is exact for those positions and
            # collapses the O(sequence^2) attention memory that dominates a long
            # prompt's per-step cost.
            input_token_ids = token_ids[: position_count + 1]
        row = PreparedGLM52TeacherRow(
            row_index=prompt_index,
            prompt_id=prompt_id,
            split="selection",
            tuning_eligible=True,
            input_token_ids=tuple(input_token_ids),
            positions=tuple(range(position_count)),
            target_token_ids=tuple(token_ids[1 : position_count + 1]),
            teacher_logits=logits,
            teacher_manifest_body_sha256=actual_body,
            teacher_shard_sha256=shard_sha,
            non_release_waiver=not release_eligible,
            topk_logit_ids=topk_ids,
            topk_logit_values=topk_values,
            tail_mass=tail_mass,
            teacher_probe_hidden_states=teacher_probes or None,
            teacher_router_topk_ids=teacher_router_ids or None,
            teacher_router_topk_weights=teacher_router_weights or None,
        )
        validate_tuning_row(row)
        rows.append(row)
    if not rows:
        raise ValueError("selection teacher row set must not be empty")
    return rows


def validate_teich_teacher_row(row: PreparedGLM52TeichTeacherRow) -> None:
    if not isinstance(row, PreparedGLM52TeichTeacherRow) or row.cache_schema_version != 3:
        raise ValueError("Teich teacher row must carry the schema-v3 sparse authority")
    if row.split not in ("train", "validation", "holdout"):
        raise ValueError("Teich teacher row split must be train, validation, or holdout")
    if row.tuning_eligible != (row.split == "train"):
        raise ValueError("only Teich train rows may be tuning eligible")
    # Reuse the tensor/shape/range validator under a train-only proxy.  The
    # real row retains its immutable split and is never silently promoted.
    validate_tuning_row(replace(row, split="train", tuning_eligible=True))


def prepare_glm52_teich_teacher_rows(
    *,
    teacher_cache_dir: Path,
    prompt_pack_path: Path,
    frozen_prompt_pack_path: Path,
    split: Literal["train", "validation", "holdout"],
    expected_manifest_sha256: str,
    max_rows: int | None = None,
    max_positions: int | None = None,
    row_indices: tuple[int, ...] | None = None,
    allow_non_release_teacher_cache: bool = False,
    lazy: bool = False,
) -> list[PreparedGLM52TeichTeacherRow | PreparedGLM52TeichDiskRow]:
    """Load authenticated sparse Teich rows without weakening selection-only v1/v2."""

    if split not in ("train", "validation", "holdout"):
        raise ValueError("Teich split must be train, validation, or holdout")
    _require_sha256(expected_manifest_sha256, label="expected Teich teacher manifest")
    root = Path(teacher_cache_dir)
    from mlx_vq.quality.glm52_teich_training_cache import (
        MANIFEST_FILENAME as TEICH_MANIFEST_FILENAME,
        audit_glm52_teich_teacher_cache,
    )

    manifest_raw = _read_regular_file(
        root / TEICH_MANIFEST_FILENAME,
        label="Teich teacher cache manifest",
    )
    if _sha256_bytes(manifest_raw) != expected_manifest_sha256:
        raise ValueError("Teich teacher cache manifest does not match external authority")
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, dict):
        raise ValueError("Teich teacher cache manifest must be an object")
    if manifest.get("schema_version") != 3 or manifest.get("record_type") != "glm52_teacher_signal_cache":
        raise ValueError("Teich teacher cache must use schema v3")
    release_eligible = manifest.get("release_eligible") is True
    if not release_eligible and not allow_non_release_teacher_cache:
        raise ValueError("Teich teacher cache is non-release; an explicit waiver is required")
    expected_sessions = manifest.get("session_count")
    expected_positions = manifest.get("supervised_position_count")
    top_k = manifest.get("top_k")
    hidden_size = manifest.get("hidden_size")
    if not all(type(value) is int and value > 0 for value in (
        expected_sessions,
        expected_positions,
        top_k,
        hidden_size,
    )):
        raise ValueError("Teich teacher cache dimensions are malformed")
    audit = audit_glm52_teich_teacher_cache(
        cache_dir=root,
        prompt_pack_path=prompt_pack_path,
        frozen_prompt_pack_path=frozen_prompt_pack_path,
        expected_session_count=expected_sessions,
        expected_supervised_positions=expected_positions,
        top_k=top_k,
        hidden_size=hidden_size,
    )
    if audit.manifest_sha256 != expected_manifest_sha256:
        raise ValueError("fresh Teich cache audit returned a different manifest identity")

    prompt_raw = _read_regular_file(Path(prompt_pack_path), label="Teich prompt pack")
    prompt_payload = json.loads(prompt_raw)
    raw_prompt_rows = prompt_payload.get("prompt_rows") if isinstance(prompt_payload, dict) else None
    if not isinstance(raw_prompt_rows, list):
        raise ValueError("Teich prompt pack must contain prompt_rows")
    prompts: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, prompt in enumerate(raw_prompt_rows):
        if not isinstance(prompt, dict) or not isinstance(prompt.get("prompt_id"), str):
            raise ValueError(f"Teich prompt_rows[{index}] must contain a prompt_id")
        prompt_id = prompt["prompt_id"]
        if prompt_id in prompts:
            raise ValueError(f"duplicate Teich prompt ID {prompt_id!r}")
        prompts[prompt_id] = (index, prompt)

    entries = manifest.get("shards")
    if not isinstance(entries, list):
        raise ValueError("Teich teacher manifest must contain shards")
    selected = [entry for entry in entries if isinstance(entry, dict) and entry.get("split") == split]
    if row_indices is not None:
        if len(set(row_indices)) != len(row_indices):
            raise ValueError("row_indices must be unique")
        try:
            selected = [selected[index] for index in row_indices]
        except IndexError as error:
            raise ValueError("row index is out of range for the requested Teich split") from error
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        selected = selected[:max_rows]
    if max_positions is not None and max_positions <= 0:
        raise ValueError("max_positions must be positive")

    rows: list[PreparedGLM52TeichTeacherRow | PreparedGLM52TeichDiskRow] = []
    for entry in selected:
        prompt_id = entry.get("prompt_id")
        if prompt_id not in prompts:
            raise ValueError(f"Teich teacher shard prompt {prompt_id!r} is absent from prompt pack")
        prompt_index, prompt = prompts[prompt_id]
        token_ids = prompt.get("encoded_token_ids")
        raw_positions = prompt.get("positions")
        raw_targets = prompt.get("target_token_ids")
        if (
            not isinstance(token_ids, list)
            or not isinstance(raw_positions, list)
            or not isinstance(raw_targets, list)
            or len(raw_positions) != len(raw_targets)
        ):
            raise ValueError(f"Teich prompt {prompt_id!r} token/position authority is malformed")
        relative = entry.get("relative_path")
        if not isinstance(relative, str):
            raise ValueError("Teich teacher shard relative_path must be a string")
        shard_path = _safe_child(root, relative, label="Teich teacher shard relative_path")
        count = len(raw_positions)
        if max_positions is not None:
            count = min(count, max_positions)
        positions = tuple(int(value) for value in raw_positions[:count])
        targets = tuple(int(value) for value in raw_targets[:count])
        if count == 0:
            raise ValueError("Teich teacher row must retain at least one position")
        input_token_ids = tuple(int(value) for value in token_ids[: positions[-1] + 2])
        if lazy:
            stat = shard_path.stat()
            rows.append(
                PreparedGLM52TeichDiskRow(
                    row_index=prompt_index,
                    prompt_id=prompt_id,
                    split=split,
                    tuning_eligible=split == "train",
                    input_token_ids=input_token_ids,
                    positions=positions,
                    target_token_ids=targets,
                    teacher_manifest_body_sha256=audit.manifest_body_sha256,
                    teacher_shard_sha256=str(entry.get("file_sha256")),
                    non_release_waiver=not release_eligible,
                    shard_path=shard_path,
                    top_k=top_k,
                    hidden_size=hidden_size,
                    audited_file_stat=(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
                )
            )
            continue
        tensors = {
            name: read_safetensors_tensor_mlx(shard_path, name)
            for name in (
                "positions",
                "target_token_ids",
                "topk_logit_ids",
                "topk_logit_values",
                "tail_mass",
                "layer_77_hidden_probe",
                "router_top8_expert_ids",
                "router_top8_normalized_weights",
            )
        }
        router_ids = tensors["router_top8_expert_ids"][:, :count]
        router_weights = tensors["router_top8_normalized_weights"][:, :count]
        row = PreparedGLM52TeichTeacherRow(
            row_index=prompt_index,
            prompt_id=prompt_id,
            split=split,
            tuning_eligible=split == "train",
            input_token_ids=input_token_ids,
            positions=positions,
            target_token_ids=targets,
            teacher_logits=None,
            teacher_manifest_body_sha256=audit.manifest_body_sha256,
            teacher_shard_sha256=str(entry.get("file_sha256")),
            non_release_waiver=not release_eligible,
            topk_logit_ids=tensors["topk_logit_ids"][:count],
            topk_logit_values=tensors["topk_logit_values"][:count],
            tail_mass=tensors["tail_mass"][:count],
            teacher_probe_hidden_states={77: tensors["layer_77_hidden_probe"][:count]},
            teacher_router_topk_ids={
                layer: router_ids[offset] for offset, layer in enumerate(range(70, 78))
            },
            teacher_router_topk_weights={
                layer: router_weights[offset] for offset, layer in enumerate(range(70, 78))
            },
        )
        validate_teich_teacher_row(row)
        rows.append(row)
    if not rows:
        raise ValueError(f"Teich {split} teacher row set must not be empty")
    return rows


def target_glm52_projection(model: Any, *, layer: int, projection: str) -> Any:
    if projection not in SUPPORTED_PROJECTIONS:
        raise ValueError(f"unsupported projection {projection!r}")
    if layer < 0 or layer >= len(model.model.layers):
        raise ValueError(f"layer {layer} is out of range")
    moe = model.model.layers[layer].mlp
    switch_mlp = getattr(moe, "switch_mlp", None)
    if switch_mlp is None:
        raise ValueError(f"layer {layer} is not a bound sparse GLM52 MoE layer")
    return getattr(switch_mlp, projection)


def route_local_projection(projection: Any, x: mx.array, indices: mx.array, *, output_chunk_size: int) -> mx.array:
    return RouteLocalSwitchLinearSurrogate.from_layer(
        projection,
        sidecar=switch_linear_layer_sidecar(projection),
        output_chunk_size=output_chunk_size,
    )(x, indices)


def _call_switch_with_surrogates(
    switch_mlp: Any,
    x: mx.array,
    indices: mx.array,
    *,
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
) -> mx.array:
    def call(name: str, value: mx.array) -> mx.array:
        projection = getattr(switch_mlp, name)
        if name in surrogate_projections:
            return route_local_projection(projection, value, indices, output_chunk_size=output_chunk_size)
        return projection(value, indices)

    gate = call("gate_proj", x)
    up = call("up_proj", x)
    return call("down_proj", nn.silu(gate) * up)


def call_glm52_vq_moe_with_route_local_surrogates(
    moe: Any,
    x: mx.array,
    *,
    layer: int,
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
) -> mx.array:
    switch_mlp = getattr(moe, "switch_mlp", None)
    if switch_mlp is None:
        raise RuntimeError(f"layer {layer} GLM52 VQ switch_mlp is not bound")
    indices, scores = moe.gate(x)
    indices = mx.stop_gradient(indices)
    y = _call_switch_with_surrogates(
        switch_mlp,
        x,
        indices,
        surrogate_projections=surrogate_projections,
        output_chunk_size=output_chunk_size,
    )
    y = mx.sum(y * scores[..., None], axis=-2).astype(y.dtype)
    shared = moe.get("shared_experts") if callable(getattr(moe, "get", None)) else None
    if shared is not None:
        y = y + shared(x)
    return y


def _selected_hidden_for_layers(
    model: Any,
    row: PreparedGLM52TeacherRow,
    *,
    layers: frozenset[int],
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
    surrogate_downstream: bool = False,
) -> mx.array:
    validate_tuning_row(row)
    if not layers:
        raise ValueError("at least one target layer is required")
    if min(layers) < 0 or max(layers) >= len(model.model.layers):
        raise ValueError("target layer is out of range")
    tokens = mx.array([list(row.input_token_ids)], dtype=mx.int32)
    h = model.model.embed_tokens(tokens)
    cache = [None] * len(model.model.layers)
    mask = create_attention_mask(h, None, return_array=True)
    prev_topk_indices = None
    for layer_index, layer_module in enumerate(model.model.layers):
        attention_result = layer_module.self_attn(
            layer_module.input_layernorm(h),
            mask,
            cache[layer_index],
            prev_topk_indices,
        )
        if not isinstance(attention_result, tuple) or len(attention_result) != 2:
            raise TypeError("GLM52 attention must return (attention_output, topk_indices)")
        attention, prev_topk_indices = attention_result
        if prev_topk_indices is not None:
            prev_topk_indices = mx.stop_gradient(prev_topk_indices)
        h = h + attention
        mlp_input = layer_module.post_attention_layernorm(h)
        switch_mlp = getattr(layer_module.mlp, "switch_mlp", None)
        use_surrogate = layer_index in layers or (
            surrogate_downstream and layer_index >= min(layers) and switch_mlp is not None
        )
        if use_surrogate:
            y = call_glm52_vq_moe_with_route_local_surrogates(
                layer_module.mlp,
                mlp_input,
                layer=layer_index,
                surrogate_projections=(
                    frozenset(SUPPORTED_PROJECTIONS)
                    if surrogate_downstream
                    else surrogate_projections
                ),
                output_chunk_size=output_chunk_size,
            )
        else:
            y = layer_module.mlp(mlp_input)
        h = h + y
    out = model.model.norm(h)
    positions = mx.array(row.positions, dtype=mx.int32)
    return mx.take(out[0], positions, axis=0)


def _selected_logits_for_layers(
    model: Any,
    row: PreparedGLM52TeacherRow,
    *,
    layers: frozenset[int],
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
    surrogate_downstream: bool = False,
) -> mx.array:
    selected = _selected_hidden_for_layers(
        model,
        row,
        layers=layers,
        surrogate_projections=surrogate_projections,
        output_chunk_size=output_chunk_size,
        surrogate_downstream=surrogate_downstream,
    )
    return model.lm_head(selected).astype(mx.float32)


def _frozen_prefix_boundary(
    model: Any,
    row: PreparedGLM52TeacherRow,
    *,
    layers: frozenset[int],
) -> tuple[mx.array, Any, Any]:
    validate_tuning_row(row)
    if not layers:
        raise ValueError("at least one target layer is required")
    if min(layers) < 0 or max(layers) >= len(model.model.layers):
        raise ValueError("target layer is out of range")
    first_layer = min(layers)
    tokens = mx.array([list(row.input_token_ids)], dtype=mx.int32)
    h = mx.stop_gradient(model.model.embed_tokens(tokens))
    cache = [None] * len(model.model.layers)
    mask = _stop_gradient_array_leaves(create_attention_mask(h, None, return_array=True))
    prev_topk_indices = None
    for layer_index in range(first_layer):
        layer_module = model.model.layers[layer_index]
        attention_result = layer_module.self_attn(
            layer_module.input_layernorm(h),
            mask,
            cache[layer_index],
            prev_topk_indices,
        )
        if not isinstance(attention_result, tuple) or len(attention_result) != 2:
            raise TypeError("GLM52 attention must return (attention_output, topk_indices)")
        attention, prev_topk_indices = attention_result
        prev_topk_indices = _stop_gradient_array_leaves(prev_topk_indices)
        h = mx.stop_gradient(h + attention)
        mlp_input = layer_module.post_attention_layernorm(h)
        h = mx.stop_gradient(h + layer_module.mlp(mlp_input))
        # Evaluate per layer so the frozen-prefix forward is submitted as many
        # small Metal command buffers instead of one 74-layer buffer that trips
        # the GPU watchdog (kIOGPUCommandBufferCallbackErrorTimeout); this also
        # frees each layer's activations immediately.
        mx.eval(h)
        if isinstance(prev_topk_indices, mx.array):
            mx.eval(prev_topk_indices)
    boundary = (
        mx.stop_gradient(h),
        _stop_gradient_array_leaves(prev_topk_indices),
        _stop_gradient_array_leaves(mask),
    )
    mx.eval(*(value for value in boundary if isinstance(value, mx.array)))
    return boundary


def _slice_batched_route_state(
    value: Any,
    *,
    batch_index: int,
    sequence_length: int,
) -> Any:
    if isinstance(value, mx.array):
        slices = [slice(None)] * value.ndim
        slices[0] = slice(batch_index, batch_index + 1)
        if value.ndim >= 3:
            slices[-2] = slice(0, sequence_length)
        elif value.ndim == 2:
            slices[1] = slice(0, sequence_length)
        return mx.stop_gradient(value[tuple(slices)])
    if isinstance(value, tuple):
        return tuple(
            _slice_batched_route_state(
                item,
                batch_index=batch_index,
                sequence_length=sequence_length,
            )
            for item in value
        )
    if isinstance(value, list):
        return [
            _slice_batched_route_state(
                item,
                batch_index=batch_index,
                sequence_length=sequence_length,
            )
            for item in value
        ]
    return value


def _prefix_length_buckets(
    model: Any,
    rows: Sequence[PreparedGLM52TeacherRow],
    *,
    first_layer: int,
    max_padding_fraction: float = 0.25,
) -> list[list[int]]:
    # IndexShare changes behavior at index_topk, so rows on opposite sides of
    # that threshold must never share a padded batch even when their lengths
    # are close. Within each regime, greedily cap right-padding at 25% of the
    # valid token count. Equal-length rows therefore remain one batch.
    thresholds = sorted(
        {
            int(indexer.index_topk)
            for layer_module in model.model.layers[:first_layer]
            for indexer in (getattr(layer_module.self_attn, "indexer", None),)
            if indexer is not None and hasattr(indexer, "index_topk")
        }
    )

    def regime(row_index: int) -> tuple[bool, ...]:
        length = len(rows[row_index].input_token_ids)
        return tuple(length > threshold for threshold in thresholds)

    ordered = sorted(
        range(len(rows)),
        key=lambda row_index: (regime(row_index), len(rows[row_index].input_token_ids)),
    )
    buckets: list[list[int]] = []
    current: list[int] = []
    current_regime: tuple[bool, ...] | None = None
    valid_tokens = 0
    max_length = 0
    for row_index in ordered:
        row_length = len(rows[row_index].input_token_ids)
        row_regime = regime(row_index)
        next_valid_tokens = valid_tokens + row_length
        next_max_length = max(max_length, row_length)
        next_padding = next_max_length * (len(current) + 1) - next_valid_tokens
        exceeds_padding = next_padding > max_padding_fraction * next_valid_tokens
        if current and (row_regime != current_regime or exceeds_padding):
            buckets.append(current)
            current = []
            valid_tokens = 0
            max_length = 0
            next_valid_tokens = row_length
            next_max_length = row_length
        current.append(row_index)
        current_regime = row_regime
        valid_tokens = next_valid_tokens
        max_length = next_max_length
    if current:
        buckets.append(current)
    return buckets


def _batched_frozen_prefix_boundaries(
    model: Any,
    rows: Sequence[PreparedGLM52TeacherRow],
    *,
    layers: frozenset[int],
    boundary_sink: Callable[
        [PreparedGLM52TeacherRow, tuple[mx.array, Any, Any]], None
    ]
    | None = None,
) -> list[tuple[mx.array, Any, Any]]:
    if not rows:
        return []
    for row in rows:
        validate_tuning_row(row)
    if not layers:
        raise ValueError("at least one target layer is required")
    if min(layers) < 0 or max(layers) >= len(model.model.layers):
        raise ValueError("target layer is out of range")
    first_layer = min(layers)
    boundaries: list[tuple[mx.array, Any, Any] | None] = [None] * len(rows)

    for bucket in _prefix_length_buckets(model, rows, first_layer=first_layer):
        lengths = [len(rows[row_index].input_token_ids) for row_index in bucket]
        max_length = max(lengths)
        padded_tokens = [
            list(rows[row_index].input_token_ids)
            + [0] * (max_length - len(rows[row_index].input_token_ids))
            for row_index in bucket
        ]
        tokens = mx.array(padded_tokens, dtype=mx.int32)
        h = mx.stop_gradient(model.model.embed_tokens(tokens))
        cache = [None] * len(model.model.layers)
        if max_length == 1:
            mask = None
        else:
            positions = mx.arange(max_length)
            causal = positions[:, None] >= positions[None, :]
            valid_keys = positions[None, :] < mx.array(lengths, dtype=mx.int32)[:, None]
            mask = mx.stop_gradient(causal[None, None, :, :] & valid_keys[:, None, None, :])
        prev_topk_indices = None
        for layer_index in range(first_layer):
            layer_module = model.model.layers[layer_index]
            attention_result = layer_module.self_attn(
                layer_module.input_layernorm(h),
                mask,
                cache[layer_index],
                prev_topk_indices,
            )
            if not isinstance(attention_result, tuple) or len(attention_result) != 2:
                raise TypeError("GLM52 attention must return (attention_output, topk_indices)")
            attention, prev_topk_indices = attention_result
            prev_topk_indices = _stop_gradient_array_leaves(prev_topk_indices)
            h = mx.stop_gradient(h + attention)
            mlp_input = layer_module.post_attention_layernorm(h)
            h = mx.stop_gradient(h + layer_module.mlp(mlp_input))
            # Preserve the watchdog discipline, but submit/evaluate once per
            # prefix layer for the entire bucket rather than once per row.
            if isinstance(prev_topk_indices, mx.array):
                mx.eval(h, prev_topk_indices)
            else:
                mx.eval(h)

        for batch_index, (row_index, length) in enumerate(zip(bucket, lengths)):
            row_mask = None if length == 1 else mask[batch_index, 0, :length, :length]
            boundary = (
                mx.stop_gradient(h[batch_index : batch_index + 1, :length]),
                _slice_batched_route_state(
                    prev_topk_indices,
                    batch_index=batch_index,
                    sequence_length=length,
                ),
                _stop_gradient_array_leaves(row_mask),
            )
            mx.eval(*(value for value in boundary if isinstance(value, mx.array)))
            if boundary_sink is None:
                boundaries[row_index] = boundary
            else:
                boundary_sink(rows[row_index], boundary)

        if boundary_sink is not None:
            # The sink has synchronously consumed this bucket's materialized
            # slices, so release its padded batch before starting the next.
            del boundary, row_mask, h, prev_topk_indices, mask, tokens
            if first_layer:
                del attention, mlp_input
            mx.clear_cache()

    if boundary_sink is not None:
        return []
    if any(boundary is None for boundary in boundaries):
        raise RuntimeError("batched frozen-prefix boundary construction was incomplete")
    return [boundary for boundary in boundaries if boundary is not None]


def _save_boundary(path: Path, boundary: tuple[mx.array, Any, Any]) -> None:
    h, prev_topk_indices, mask = boundary
    tensors: dict[str, mx.array] = {"h": h}
    if isinstance(prev_topk_indices, mx.array):
        tensors["prev"] = prev_topk_indices
    if isinstance(mask, mx.array):
        tensors["mask"] = mask
    mx.save_safetensors(str(path), tensors)


def _load_boundary(path: Path) -> tuple[mx.array, Any, Any]:
    tensors = mx.load(str(path))
    return (tensors["h"], tensors.get("prev"), tensors.get("mask"))


def _selected_logits_from_prefix(
    model: Any,
    row: PreparedGLM52TeacherRow,
    boundary: tuple[mx.array, Any, Any],
    *,
    layers: frozenset[int],
    output_chunk_size: int,
    capture_cka_layers: frozenset[int] = frozenset(),
    capture_router_layers: frozenset[int] = frozenset(),
) -> mx.array | tuple[mx.array, Mapping[int, mx.array], Mapping[int, mx.array]]:
    validate_tuning_row(row)
    if not layers:
        raise ValueError("at least one target layer is required")
    if min(layers) < 0 or max(layers) >= len(model.model.layers):
        raise ValueError("target layer is out of range")
    first_layer = min(layers)
    h, prev_topk_indices, mask = boundary
    cache = [None] * len(model.model.layers)
    checkpoint_enabled = os.environ.get("GLM52_TRAIN_GRADIENT_CHECKPOINT") == "1"
    captured_hidden: dict[int, mx.array] = {}
    captured_router_logits: dict[int, mx.array] = {}
    capture_positions = (
        mx.array(row.positions, dtype=mx.int32)
        if capture_cka_layers or capture_router_layers
        else None
    )

    def _layer_forward(h_in: mx.array, prev_in: Any, layer_index: int) -> tuple[mx.array, Any]:
        layer_module = model.model.layers[layer_index]
        attention_result = layer_module.self_attn(
            layer_module.input_layernorm(h_in),
            mask,
            cache[layer_index],
            prev_in,
        )
        if not isinstance(attention_result, tuple) or len(attention_result) != 2:
            raise TypeError("GLM52 attention must return (attention_output, topk_indices)")
        attention, new_prev = attention_result
        if new_prev is not None:
            new_prev = mx.stop_gradient(new_prev)
        h_mid = h_in + attention
        mlp_input = layer_module.post_attention_layernorm(h_mid)
        switch_mlp = getattr(layer_module.mlp, "switch_mlp", None)
        if layer_index in capture_router_layers:
            gate = getattr(layer_module.mlp, "gate", None)
            gate_weight = getattr(gate, "weight", None)
            if gate_weight is None:
                gate_weight = getattr(getattr(gate, "gate", None), "weight", None)
            if gate_weight is None or capture_positions is None:
                raise ValueError(f"layer {layer_index} does not expose student gate logits")
            router_all = mlp_input[0].astype(mx.float32) @ gate_weight.astype(mx.float32).T
            captured_router_logits[layer_index] = mx.take(router_all, capture_positions, axis=0)
        if layer_index in layers or switch_mlp is not None:
            y = call_glm52_vq_moe_with_route_local_surrogates(
                layer_module.mlp,
                mlp_input,
                layer=layer_index,
                surrogate_projections=frozenset(SUPPORTED_PROJECTIONS),
                output_chunk_size=output_chunk_size,
            )
        else:
            y = layer_module.mlp(mlp_input)
        h_out = h_mid + y
        if layer_index in capture_cka_layers:
            if capture_positions is None:
                raise RuntimeError("CKA capture positions were not initialized")
            captured_hidden[layer_index] = mx.take(h_out[0], capture_positions, axis=0)
        return h_out, new_prev

    for layer_index in range(first_layer, len(model.model.layers)):
        # Checkpoint only layers WITHOUT trainable adapters: their gradient flows
        # through the input hidden state (a checkpoint argument), so recomputing
        # them in the backward pass is exact. Trainable layers keep their
        # activations — mx.checkpoint does not propagate gradients to
        # closure-captured params, so checkpointing them would zero the adapter
        # gradient. This trims the retained full-sequence MoE activations of the
        # non-trainable suffix (target..head), the bulk of per-step memory.
        if (
            checkpoint_enabled
            and layer_index not in layers
            and layer_index not in capture_cka_layers
            and layer_index not in capture_router_layers
        ):
            h, prev_topk_indices = mx.checkpoint(_layer_forward)(h, prev_topk_indices, layer_index)
        else:
            h, prev_topk_indices = _layer_forward(h, prev_topk_indices, layer_index)
    out = model.model.norm(h)
    positions = mx.array(row.positions, dtype=mx.int32)
    selected = mx.take(out[0], positions, axis=0)
    logits = model.lm_head(selected).astype(mx.float32)
    if capture_cka_layers or capture_router_layers:
        return logits, captured_hidden, captured_router_logits
    return logits


def selected_glm52_logits_selected_layer(
    model: Any,
    row: PreparedGLM52TeacherRow,
    *,
    layer: int,
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
) -> mx.array:
    return _selected_logits_for_layers(
        model,
        row,
        layers=frozenset((layer,)),
        surrogate_projections=surrogate_projections,
        output_chunk_size=output_chunk_size,
    )


def selected_glm52_hidden_selected_layer(
    model: Any,
    row: PreparedGLM52TeacherRow,
    *,
    layer: int,
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
) -> mx.array:
    """Return selected normalized hidden states so evaluation can slice LM-head work."""

    return _selected_hidden_for_layers(
        model,
        row,
        layers=frozenset((layer,)),
        surrogate_projections=surrogate_projections,
        output_chunk_size=output_chunk_size,
    )


def _teacher_top1_margin_loss(
    student_logits: mx.array,
    teacher_logits: mx.array,
    *,
    margin: float,
    competitor_token_ids: tuple[tuple[int, ...], ...] | None,
    include_hardest_competitor: bool,
) -> mx.array:
    teacher_top1 = mx.argmax(teacher_logits.astype(mx.float32), axis=-1).astype(mx.int32)
    top1_logits = mx.take_along_axis(student_logits.astype(mx.float32), teacher_top1[:, None], axis=-1)[:, 0]
    terms = []
    if competitor_token_ids is None or include_hardest_competitor:
        vocab_ids = mx.arange(student_logits.shape[-1], dtype=mx.int32)[None, :]
        competitors = mx.where(
            vocab_ids == teacher_top1[:, None],
            mx.array(-1.0e30, dtype=mx.float32),
            student_logits.astype(mx.float32),
        )
        terms.append(mx.mean(mx.maximum(mx.max(competitors, axis=-1) - top1_logits + margin, 0.0)))
    if competitor_token_ids is not None:
        if len(competitor_token_ids) != student_logits.shape[0]:
            raise ValueError("competitor group count must match positions")
        explicit = []
        for position, group in enumerate(competitor_token_ids):
            for token in group:
                if token < 0 or token >= student_logits.shape[-1]:
                    raise ValueError("competitor token is outside the vocabulary")
                explicit.append(mx.maximum(student_logits[position, token] - top1_logits[position] + margin, 0.0))
        if not explicit:
            raise ValueError("explicit competitor groups must not all be empty")
        # Deliberately a sum, matching Air. Its scale grows with competitor count.
        terms.append(mx.sum(mx.stack(explicit)))
    if not terms:
        raise ValueError("margin loss requires hardest or explicit competitors")
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return result


def _dense_full_vocab_kl(
    student_logits: mx.array,
    teacher_logits: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Preserve the original dense KL operation order for disabled-path parity."""

    teacher_log_probs = teacher_logits - mx.logsumexp(teacher_logits, axis=-1, keepdims=True)
    teacher_probs = mx.exp(teacher_log_probs)
    student_log_probs = student_logits - mx.logsumexp(student_logits, axis=-1, keepdims=True)
    per_position_kld = mx.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
    return mx.mean(per_position_kld), per_position_kld, student_log_probs


def _topk_aggregated_tail_kl(
    student_logits: mx.array,
    topk_logit_ids: mx.array,
    topk_logit_values: mx.array,
    tail_mass: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    if (
        topk_logit_ids.ndim != 2
        or topk_logit_values.shape != topk_logit_ids.shape
        or topk_logit_ids.shape[0] != student_logits.shape[0]
        or tail_mass.shape != (student_logits.shape[0],)
    ):
        raise ValueError("top-K IDs/values must be [positions, K] and tail mass [positions]")
    ids = topk_logit_ids.astype(mx.int32)
    values = mx.stop_gradient(topk_logit_values.astype(mx.float32))
    tail = mx.stop_gradient(tail_mass.astype(mx.float32))
    vocab = student_logits.shape[-1]
    if ids.shape[-1] > vocab:
        raise ValueError("teacher top-K cannot exceed the student vocabulary")
    if ids.shape[-1] == vocab and bool(mx.all(tail == 0).item()):
        canonical_ids = mx.broadcast_to(mx.arange(vocab, dtype=mx.int32)[None, :], ids.shape)
        if bool(mx.array_equal(ids, canonical_ids).item()):
            return _dense_full_vocab_kl(student_logits, values)

    student_log_probs = student_logits - mx.logsumexp(student_logits, axis=-1, keepdims=True)
    selected_student_log_probs = mx.take_along_axis(student_log_probs, ids, axis=-1)
    conditional_log_probs = values - mx.logsumexp(values, axis=-1, keepdims=True)
    teacher_top_probs = mx.exp(conditional_log_probs) * (1.0 - tail[:, None])
    epsilon = mx.array(1.0e-7, dtype=mx.float32)
    teacher_top_log_probs = mx.log(mx.maximum(teacher_top_probs, epsilon))
    student_tail_probs = mx.maximum(
        1.0 - mx.sum(mx.exp(selected_student_log_probs), axis=-1),
        epsilon,
    )
    tail_terms = mx.where(
        tail > 0,
        tail * (mx.log(mx.maximum(tail, epsilon)) - mx.log(student_tail_probs)),
        mx.zeros_like(tail),
    )
    per_position = mx.sum(
        teacher_top_probs * (teacher_top_log_probs - selected_student_log_probs), axis=-1
    ) + tail_terms
    return mx.mean(per_position), per_position, student_log_probs


def glm52_topk_kl_loss(
    student_logits: mx.array,
    topk_logit_ids: mx.array,
    topk_logit_values: mx.array,
    tail_mass: mx.array,
) -> mx.array:
    return _topk_aggregated_tail_kl(
        student_logits.astype(mx.float32),
        topk_logit_ids,
        topk_logit_values,
        tail_mass,
    )[0]


def glm52_feature_space_linear_cka_loss(
    teacher_hidden: mx.array,
    student_hidden: mx.array,
    *,
    eps: float = 1.0e-12,
) -> mx.array:
    if teacher_hidden.ndim != 2 or student_hidden.ndim != 2:
        raise ValueError("CKA hidden states must have shape [positions, hidden]")
    if teacher_hidden.shape[0] != student_hidden.shape[0]:
        raise ValueError("teacher and student CKA probes must share positions")
    teacher = mx.stop_gradient(teacher_hidden.astype(mx.float32))
    student = student_hidden.astype(mx.float32)
    teacher = teacher - mx.mean(teacher, axis=0, keepdims=True)
    student = student - mx.mean(student, axis=0, keepdims=True)
    cross = teacher.T @ student
    teacher_cov = teacher.T @ teacher
    student_cov = student.T @ student
    numerator = mx.sum(cross * cross)
    denominator = mx.sqrt(mx.sum(teacher_cov * teacher_cov) * mx.sum(student_cov * student_cov))
    similarity = numerator / mx.maximum(denominator, mx.array(eps, dtype=mx.float32))
    return 1.0 - mx.clip(similarity, 0.0, 1.0)


def glm52_router_distillation_loss(
    student_router_logits: mx.array,
    teacher_topk_ids: mx.array,
    teacher_topk_weights: mx.array,
    *,
    entropy_beta: float,
    mc_expert_explore: bool,
    seed: int,
) -> mx.array:
    if (
        student_router_logits.ndim != 2
        or teacher_topk_ids.ndim != 2
        or teacher_topk_ids.shape != teacher_topk_weights.shape
        or teacher_topk_ids.shape[0] != student_router_logits.shape[0]
    ):
        raise ValueError("router logits and teacher targets must share their position dimension")
    if entropy_beta < 0:
        raise ValueError("router entropy beta must be non-negative")
    ids = mx.stop_gradient(teacher_topk_ids.astype(mx.int32))
    weights = mx.stop_gradient(teacher_topk_weights.astype(mx.float32))
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    logits = student_router_logits.astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    selected = mx.take_along_axis(log_probs, ids, axis=-1)
    epsilon = mx.array(1.0e-7, dtype=mx.float32)
    router_kl = mx.mean(mx.sum(weights * (mx.log(mx.maximum(weights, epsilon)) - selected), axis=-1))
    probs = mx.exp(log_probs)
    entropy = mx.mean(-mx.sum(probs * log_probs, axis=-1))
    loss = router_kl - entropy_beta * entropy
    if mc_expert_explore > 0:
        num_experts = logits.shape[-1]
        rng = np.random.default_rng(seed)
        schedule = rng.permutation(num_experts)
        positive_weights = rng.uniform(0.5, 1.5, size=num_experts).astype(np.float32)
        exploration_target = np.empty_like(positive_weights)
        exploration_target[schedule] = positive_weights
        exploration_target /= exploration_target.sum(dtype=np.float32)
        target = mx.stop_gradient(mx.array(exploration_target, dtype=mx.float32))
        loss = loss + mx.mean(-mx.sum(log_probs * target[None, :], axis=-1))
    return loss


def glm52_adapter_loss(
    student_logits: mx.array,
    row: PreparedGLM52TeacherRow,
    *,
    target_nll_weight: float,
    teacher_top1_margin_weight: float,
    teacher_top1_margin: float,
    teacher_top1_competitor_token_ids: tuple[tuple[int, ...], ...] | None,
    teacher_top1_include_hardest_competitor: bool,
    tail_kld_weight: float,
    topk: int | None = None,
    cka_enabled: bool = False,
    cka_layers: tuple[int, ...] = (),
    student_hidden_states: Mapping[int, mx.array] | None = None,
    router_kl_weight: float = 0.0,
    router_entropy_beta: float = 0.0,
    mc_expert_explore: bool = False,
    student_router_logits: Mapping[int, mx.array] | None = None,
    seed: int = 20260711,
) -> mx.array:
    teacher_logits = None if row.teacher_logits is None else mx.stop_gradient(row.teacher_logits.astype(mx.float32))
    if topk is None:
        if teacher_logits is None or student_logits.shape != teacher_logits.shape:
            raise ValueError("dense KL requires matching student and teacher [positions, vocab] logits")
        kl_loss, per_position_kld, student_log_probs = _dense_full_vocab_kl(
            student_logits, teacher_logits
        )
    else:
        if row.topk_logit_ids is None or row.topk_logit_values is None or row.tail_mass is None:
            raise ValueError("configured topk requires teacher top-K IDs, values, and tail mass")
        if row.topk_logit_ids.shape[-1] != topk:
            raise ValueError("configured topk must equal teacher-cache metadata")
        kl_loss, per_position_kld, student_log_probs = _topk_aggregated_tail_kl(
            student_logits,
            row.topk_logit_ids,
            row.topk_logit_values,
            row.tail_mass,
        )
    loss = kl_loss
    if cka_enabled:
        teacher_hidden = row.teacher_probe_hidden_states
        if teacher_hidden is None or student_hidden_states is None:
            raise ValueError("CKA requires teacher and student probe-layer hidden states")
        if set(cka_layers) != set(teacher_hidden) or set(cka_layers) != set(student_hidden_states):
            raise ValueError("CKA probe layers must exactly match configured cka_layers")
        cka_terms = [
            glm52_feature_space_linear_cka_loss(
                teacher_hidden[layer], student_hidden_states[layer]
            )
            for layer in cka_layers
        ]
        cka_loss = mx.mean(mx.stack(cka_terms))
        balance = mx.stop_gradient(kl_loss / (cka_loss + mx.array(1.0e-7, dtype=mx.float32)))
        loss = loss + balance * cka_loss
    if target_nll_weight > 0:
        targets = mx.array(row.target_token_ids, dtype=mx.int32)
        selected = mx.take_along_axis(student_log_probs, targets[:, None], axis=-1)[:, 0]
        loss = loss + target_nll_weight * -mx.mean(selected)
    if teacher_top1_margin_weight > 0:
        if teacher_logits is None:
            raise ValueError("teacher-top1 margin requires dense teacher logits")
        loss = loss + teacher_top1_margin_weight * _teacher_top1_margin_loss(
            student_logits,
            teacher_logits,
            margin=teacher_top1_margin,
            competitor_token_ids=teacher_top1_competitor_token_ids,
            include_hardest_competitor=teacher_top1_include_hardest_competitor,
        )
    if tail_kld_weight > 0:
        loss = loss + tail_kld_weight * mx.max(per_position_kld)
    router_enabled = router_kl_weight > 0 or router_entropy_beta > 0 or mc_expert_explore > 0
    if router_enabled:
        teacher_ids = row.teacher_router_topk_ids
        teacher_weights = row.teacher_router_topk_weights
        if teacher_ids is None or teacher_weights is None or student_router_logits is None:
            raise ValueError("router terms require teacher router targets and student router logits")
        router_layers = tuple(sorted(student_router_logits))
        if not set(router_layers).issubset(teacher_ids) or not set(router_layers).issubset(
            teacher_weights
        ):
            raise ValueError("teacher router targets must cover every student router layer")
        router_terms = [
            glm52_router_distillation_loss(
                student_router_logits[layer],
                teacher_ids[layer],
                teacher_weights[layer],
                entropy_beta=router_entropy_beta,
                mc_expert_explore=mc_expert_explore,
                seed=seed + layer,
            )
            for layer in router_layers
        ]
        loss = loss + router_kl_weight * mx.mean(mx.stack(router_terms))
    return loss


def initialize_low_rank_params(
    projection_map: Mapping[tuple[int, str], Any], *, rank: int, init_scale: float, seed: int
) -> dict[str, mx.array]:
    params: dict[str, mx.array] = {}
    for layer_projection in sorted(projection_map):
        layer, projection_name = layer_projection
        projection = projection_map[layer_projection]
        key = f"{layer}.{projection_name}"
        local_seed = seed + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
        rng = np.random.default_rng(local_seed)
        right = rng.normal(
            scale=init_scale,
            size=(projection.num_experts, rank, projection.input_dims),
        ).astype(np.float32)
        params[f"{key}.low_rank_left"] = mx.zeros(
            (projection.num_experts, projection.output_dims, rank), dtype=mx.float32
        )
        params[f"{key}.low_rank_right"] = mx.array(right, dtype=mx.float32)
    return params


def _bind_params(projection_map: Mapping[tuple[int, str], Any], params: Mapping[str, mx.array]) -> None:
    for (layer, name), projection in projection_map.items():
        prefix = f"{layer}.{name}"
        left = params.get(f"{prefix}.low_rank_left")
        right = params.get(f"{prefix}.low_rank_right")
        if left is None or right is None:
            raise ValueError(f"{prefix} requires both low_rank_left and low_rank_right")
        projection.set_continuous_sidecar(
            scale_delta=projection.get("continuous_scale_delta"),
            output_bias=projection.get("continuous_output_bias"),
            low_rank_left=left.astype(mx.float32),
            low_rank_right=right.astype(mx.float32),
        )


def _stop_gradient_array_leaves(value: Any) -> Any:
    if isinstance(value, mx.array):
        return mx.stop_gradient(value)
    if isinstance(value, tuple):
        return tuple(_stop_gradient_array_leaves(item) for item in value)
    if isinstance(value, list):
        return [_stop_gradient_array_leaves(item) for item in value]
    return value


class _StopGradientIndexer:
    def __init__(self, indexer: Any) -> None:
        self.indexer = indexer

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _stop_gradient_array_leaves(self.indexer(*args, **kwargs))


def _freeze_attention_indexers(model: Any, *, from_layer: int) -> list[tuple[Any, Any]]:
    frozen: list[tuple[Any, Any]] = []
    for layer_module in model.model.layers[from_layer:]:
        attention = layer_module.self_attn
        indexer = getattr(attention, "indexer", None)
        if indexer is None:
            continue
        frozen.append((attention, indexer))
        attention.indexer = _StopGradientIndexer(indexer)
    return frozen


def _apply_mlx_cache_limit() -> None:
    """Bound the MLX allocator's reuse cache so it does not creep across steps.

    With the ~104 GB composite resident on a 128 GB host, an unbounded reuse
    cache is what tips a per-step transient into a jetsam OOM. GLM52_TRAIN_CACHE_LIMIT_GB
    (default 8) caps the cache; 0 disables the cap.
    """
    try:
        limit_gb = float(os.environ.get("GLM52_TRAIN_CACHE_LIMIT_GB", "8") or 0)
    except ValueError:
        limit_gb = 8.0
    if limit_gb <= 0:
        return
    limit_bytes = int(limit_gb * 1024**3)
    setter = getattr(mx, "set_cache_limit", None) or getattr(getattr(mx, "metal", None), "set_cache_limit", None)
    if setter is not None:
        setter(limit_bytes)


def _peak_memory_gb() -> float:
    getter = getattr(mx, "get_peak_memory", None) or getattr(getattr(mx, "metal", None), "get_peak_memory", None)
    if getter is None:
        return 0.0
    try:
        return float(getter()) / 1024**3
    except Exception:
        return 0.0


def _active_memory_gb() -> float:
    getter = getattr(mx, "get_active_memory", None) or getattr(getattr(mx, "metal", None), "get_active_memory", None)
    if getter is None:
        return 0.0
    try:
        return float(getter()) / 1024**3
    except Exception:
        return 0.0


def _mem_stage(label: str) -> None:
    if os.environ.get("GLM52_TRAIN_PROGRESS") == "1":
        print(f"[mem] {label}: active={_active_memory_gb():.1f}GB peak={_peak_memory_gb():.1f}GB", flush=True)


def _slice_teich_training_row(
    row: PreparedGLM52TeacherRow,
    *,
    start: int,
    end: int,
) -> PreparedGLM52TeacherRow:
    if not isinstance(row, PreparedGLM52TeichTeacherRow):
        raise ValueError("deterministic Teich windows require schema-v3 rows")
    if not 0 <= start < end <= len(row.positions):
        raise ValueError("training window is outside the row's supervised positions")

    def sliced(values: Mapping[int, mx.array] | None) -> Mapping[int, mx.array] | None:
        return None if values is None else {layer: value[start:end] for layer, value in values.items()}

    return replace(
        row,
        positions=row.positions[start:end],
        target_token_ids=row.target_token_ids[start:end],
        teacher_logits=(
            None if row.teacher_logits is None else row.teacher_logits[start:end]
        ),
        topk_logit_ids=(
            None if row.topk_logit_ids is None else row.topk_logit_ids[start:end]
        ),
        topk_logit_values=(
            None if row.topk_logit_values is None else row.topk_logit_values[start:end]
        ),
        tail_mass=None if row.tail_mass is None else row.tail_mass[start:end],
        teacher_probe_hidden_states=sliced(row.teacher_probe_hidden_states),
        teacher_router_topk_ids=sliced(row.teacher_router_topk_ids),
        teacher_router_topk_weights=sliced(row.teacher_router_topk_weights),
    )


class GLM52FixedValidationRunner:
    """Fixed sparse validation windows with durable frozen-prefix boundaries."""

    def __init__(
        self,
        *,
        model: Any,
        rows: Sequence[PreparedGLM52TeichTeacherRow | PreparedGLM52TeichDiskRow],
        windows: Sequence[TrainingWindow],
        config: AdapterTrainingConfig,
        boundary_dir: Path,
    ) -> None:
        by_prompt = {row.prompt_id: row for row in rows}
        if len(by_prompt) != len(rows):
            raise ValueError("validation rows must have unique prompt IDs")
        self.rows = []
        for window in windows:
            try:
                source = by_prompt[window.prompt_id]
            except KeyError as error:
                raise ValueError("validation window names an absent row") from error
            if source.split != "validation" or source.tuning_eligible:
                raise ValueError("fixed validation runner accepts validation rows only")
            if isinstance(source, PreparedGLM52TeichDiskRow):
                loaded = source.load_window(window.start, window.end)
            else:
                loaded = _slice_teich_training_row(
                    replace(source, split="train", tuning_eligible=True),
                    start=window.start,
                    end=window.end,
                )
            self.rows.append(replace(loaded, split="train", tuning_eligible=True))
        if not self.rows:
            raise ValueError("fixed validation runner requires at least one window")
        self.model = model
        self.config = config
        self.layers = frozenset(config.layers)
        self.cka_layers = (
            frozenset(config.cka_layers) if config.cka_enabled else frozenset()
        )
        self.router_layers = (
            self.layers
            if (
                config.router_kl_weight > 0
                or config.router_entropy_beta > 0
                or config.mc_expert_explore
            )
            else frozenset()
        )
        self.boundary_dir = Path(boundary_dir)
        self.boundary_dir.mkdir(parents=True, exist_ok=True)
        unique: dict[int, PreparedGLM52TeacherRow] = {}
        for row in self.rows:
            unique.setdefault(row.row_index, row)
        self.boundary_paths: dict[int, Path] = {}
        inventory = []
        for row_index, row in sorted(unique.items()):
            path = self.boundary_dir / f"row-{row_index}.safetensors"
            if not path.exists():
                _save_boundary(
                    path,
                    _frozen_prefix_boundary(model, row, layers=self.layers),
                )
            raw = _read_regular_file(path, label="validation frozen-prefix boundary")
            self.boundary_paths[row_index] = path
            inventory.append(
                {
                    "row_index": row_index,
                    "prompt_id": row.prompt_id,
                    "input_token_ids_sha256": _sha256_bytes(
                        _canonical_bytes(list(row.input_token_ids))
                    ),
                    "file_sha256": _sha256_bytes(raw),
                }
            )
        body = {
            "record_type": "glm52_validation_boundary_manifest_v1",
            "layers": sorted(self.layers),
            "inventory": inventory,
        }
        manifest = {
            **body,
            "manifest_body_sha256": _sha256_bytes(_canonical_bytes(body)),
        }
        manifest_path = self.boundary_dir / "validation-boundary-manifest.json"
        if manifest_path.exists():
            if json.loads(_read_regular_file(manifest_path, label="validation boundary manifest")) != manifest:
                raise ValueError("validation boundary manifest identity drift")
        else:
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_bytes(_canonical_bytes(manifest) + b"\n")
            os.replace(temporary, manifest_path)

    def __call__(self, params: Mapping[str, mx.array], _step: int) -> float:
        _bind_params(
            {
                (layer, projection): target_glm52_projection(
                    self.model, layer=layer, projection=projection
                )
                for layer in self.config.layers
                for projection in self.config.projections
            },
            params,
        )
        weighted_loss = 0.0
        positions = 0
        for row in self.rows:
            boundary = _load_boundary(self.boundary_paths[row.row_index])
            result = _selected_logits_from_prefix(
                self.model,
                row,
                boundary,
                layers=self.layers,
                output_chunk_size=self.config.surrogate_output_chunk_size,
                capture_cka_layers=self.cka_layers,
                capture_router_layers=self.router_layers,
            )
            if self.cka_layers or self.router_layers:
                logits, hidden, routers = result
            else:
                logits = result
                hidden = None
                routers = None
            loss = glm52_adapter_loss(
                logits,
                row,
                target_nll_weight=self.config.target_nll_weight,
                teacher_top1_margin_weight=self.config.teacher_top1_margin_weight,
                teacher_top1_margin=self.config.teacher_top1_margin,
                teacher_top1_competitor_token_ids=self.config.teacher_top1_competitor_token_ids,
                teacher_top1_include_hardest_competitor=self.config.teacher_top1_include_hardest_competitor,
                tail_kld_weight=self.config.tail_kld_weight,
                topk=self.config.topk,
                cka_enabled=self.config.cka_enabled,
                cka_layers=self.config.cka_layers,
                student_hidden_states=hidden,
                router_kl_weight=self.config.router_kl_weight,
                router_entropy_beta=self.config.router_entropy_beta,
                mc_expert_explore=self.config.mc_expert_explore,
                student_router_logits=routers,
                seed=self.config.seed,
            )
            mx.eval(loss)
            weighted_loss += float(loss.item()) * len(row.positions)
            positions += len(row.positions)
            mx.clear_cache()
        return weighted_loss / positions


def run_low_rank_training(
    *,
    model: Any,
    rows: Sequence[PreparedGLM52TeacherRow | PreparedGLM52TeichDiskRow],
    projection_map: Mapping[tuple[int, str], Any],
    config: AdapterTrainingConfig,
    training_schedule: TrainingSchedule | None = None,
    checkpoint_store: TrainingCheckpointStore | None = None,
    checkpoint_config: TrainingCheckpointConfig | None = None,
    resume: bool = False,
    persistent_boundary_dir: Path | None = None,
    validation_fn: Callable[[Mapping[str, mx.array], int], float] | None = None,
    checkpoint_sync: Callable[[Path, int, bool], None] | None = None,
) -> AdapterTrainingResult:
    if not rows:
        raise ValueError("training rows must not be empty")
    rows_by_prompt = {row.prompt_id: row for row in rows}
    if len(rows_by_prompt) != len(rows):
        raise ValueError("training input rows must have unique prompt IDs")
    scheduled_windows: tuple[TrainingWindow, ...] = ()
    if training_schedule is not None:
        if config.steps > len(training_schedule.windows):
            raise ValueError("training steps cannot exceed the one-epoch Teich schedule")
        scheduled_windows = training_schedule.windows[: config.steps]
        for window in scheduled_windows:
            try:
                source = rows_by_prompt[window.prompt_id]
            except KeyError as error:
                raise ValueError("training schedule names a row absent from the train split") from error
            if not isinstance(source, (PreparedGLM52TeichTeacherRow, PreparedGLM52TeichDiskRow)):
                raise ValueError("deterministic Teich schedules require schema-v3 rows")
            if not 0 <= window.start < window.end <= len(source.positions):
                raise ValueError("training schedule window exceeds its Teich row")
    else:
        if any(isinstance(row, PreparedGLM52TeichDiskRow) for row in rows):
            raise ValueError("disk-backed Teich rows require a deterministic training schedule")
        training_rows = list(rows)
    router_enabled = (
        config.router_kl_weight > 0
        or config.router_entropy_beta > 0
        or config.mc_expert_explore > 0
    )
    for row in rows:
        if isinstance(row, PreparedGLM52TeichDiskRow):
            if row.split != "train" or not row.tuning_eligible:
                raise ValueError("GLM52 Teich adapter tuning is train-only")
        else:
            validate_tuning_row(row)
        if config.topk is not None:
            row_topk = (
                row.top_k
                if isinstance(row, PreparedGLM52TeichDiskRow)
                else None if row.topk_logit_ids is None else row.topk_logit_ids.shape[-1]
            )
            if row_topk != config.topk:
                raise ValueError("configured topk must equal every teacher row cache K")
        if config.cka_enabled:
            probe_layers = (
                {77}
                if isinstance(row, PreparedGLM52TeichDiskRow)
                else set() if row.teacher_probe_hidden_states is None else set(row.teacher_probe_hidden_states)
            )
            if probe_layers != set(config.cka_layers):
                raise ValueError("CKA-enabled training requires every configured teacher probe layer")
        if router_enabled:
            router_layers_available = (
                set(row.router_layers)
                if isinstance(row, PreparedGLM52TeichDiskRow)
                else set() if row.teacher_router_topk_ids is None else set(row.teacher_router_topk_ids)
            )
            weights_available = (
                set(row.router_layers)
                if isinstance(row, PreparedGLM52TeichDiskRow)
                else set() if row.teacher_router_topk_weights is None else set(row.teacher_router_topk_weights)
            )
            if not set(config.layers).issubset(router_layers_available) or not set(
                config.layers
            ).issubset(weights_available):
                raise ValueError("router terms require targets for every configured sparse layer")
    expected_keys = {(layer, projection) for layer in config.layers for projection in config.projections}
    if set(projection_map) != expected_keys:
        raise ValueError("projection_map must exactly match configured layers and projections")
    _apply_mlx_cache_limit()
    params = initialize_low_rank_params(
        projection_map,
        rank=config.rank,
        init_scale=config.low_rank_init_scale,
        seed=config.seed,
    )
    restored_losses: list[mx.array] = []
    start_step = 0
    validation_state = ValidationState()
    if resume:
        if checkpoint_store is None:
            raise ValueError("resumable training requires a checkpoint_store")
        restored = checkpoint_store.load_latest()
        if restored.completed_schedule_index > config.steps:
            raise ValueError("training checkpoint is beyond the configured schedule")
        if set(restored.params) != set(params):
            raise ValueError("training checkpoint parameter inventory drift")
        params = {
            name: mx.array(value, dtype=mx.float32)
            for name, value in restored.params.items()
        }
        mx.eval(*params.values())
        restored_losses = [mx.array(value, dtype=mx.float32) for value in restored.losses]
        start_step = restored.completed_schedule_index
        validation_state = ValidationState(
            best_metric=restored.best_validation_metric,
            best_checkpoint_sha256=restored.best_checkpoint_sha256,
            consecutive_regressions=restored.consecutive_validation_regressions,
        )
    layers = frozenset(config.layers)
    cka_layers = frozenset(config.cka_layers) if config.cka_enabled else frozenset()
    if cka_layers:
        if min(cka_layers) < min(layers) or max(cka_layers) >= len(model.model.layers):
            raise ValueError("cka_layers must be present in the trainable suffix")
    router_layers = layers if router_enabled else frozenset()
    _mem_stage("after model resident / params init")
    # Frozen-prefix boundaries are deterministic constants. Build all unique
    # rows in length-bucketed batches so each prefix layer pays its fixed MoE
    # dispatch floor once per bucket, then retain the same per-row artifacts:
    #  * disk mode (GLM52_TRAIN_BOUNDARY_DISK=1, recommended for the real model):
    #    persist each sliced boundary and load one per step. The
    #    per-step forward then materialises only the suffix layers, so peak stays
    #    far below holding all boundaries (and the whole prefix) resident.
    #  * RAM mode (default): retain every sliced boundary by row_index.
    disk_mode = (
        os.environ.get("GLM52_TRAIN_BOUNDARY_DISK") == "1"
        or persistent_boundary_dir is not None
    )
    boundary_dir: Path | None = None
    unique_rows: list[PreparedGLM52TeacherRow] = []
    seen_row_indices: set[int] = set()
    boundary_sources: Sequence[PreparedGLM52TeacherRow | PreparedGLM52TeichDiskRow]
    if training_schedule is not None:
        boundary_sources = [rows_by_prompt[window.prompt_id] for window in scheduled_windows]
    else:
        boundary_sources = training_rows
    for source in boundary_sources:
        if source.row_index in seen_row_indices:
            continue
        row = (
            source.load_window(0, 1)
            if isinstance(source, PreparedGLM52TeichDiskRow)
            else source
        )
        unique_rows.append(row)
        seen_row_indices.add(row.row_index)
    if disk_mode:
        boundary_dir = (
            Path(persistent_boundary_dir)
            if persistent_boundary_dir is not None
            else Path(tempfile.mkdtemp(prefix="glm52-boundaries-"))
        )
        boundary_dir.mkdir(parents=True, exist_ok=True)
        boundary_paths: dict[int, Path] = {}

        def _persist_boundary(
            row: PreparedGLM52TeacherRow,
            boundary: tuple[mx.array, Any, Any],
        ) -> None:
            path = boundary_dir / f"row-{row.row_index}.safetensors"
            _save_boundary(path, boundary)
            boundary_paths[row.row_index] = path

        boundary_manifest_path = boundary_dir / "frozen-prefix-manifest.json"
        expected_boundary_rows = [
            {
                "row_index": row.row_index,
                "prompt_id": row.prompt_id,
                "input_token_count": len(row.input_token_ids),
                "input_token_ids_sha256": _sha256_bytes(
                    _canonical_bytes(list(row.input_token_ids))
                ),
            }
            for row in unique_rows
        ]
        expected_boundary_authority = {
            "record_type": "glm52_frozen_prefix_boundary_manifest_v1",
            "layers": sorted(layers),
            "rows": expected_boundary_rows,
        }
        reuse_boundaries = False
        if boundary_manifest_path.exists():
            existing = json.loads(
                _read_regular_file(
                    boundary_manifest_path,
                    label="frozen prefix boundary manifest",
                )
            )
            body = dict(existing)
            body_sha = body.pop("manifest_body_sha256", None)
            files = body.pop("files", None)
            if (
                body == expected_boundary_authority
                and body_sha == _sha256_bytes(_canonical_bytes({**body, "files": files}))
                and isinstance(files, list)
            ):
                reuse_boundaries = True
                for item in files:
                    if not isinstance(item, dict) or type(item.get("row_index")) is not int:
                        reuse_boundaries = False
                        break
                    path = boundary_dir / f"row-{item['row_index']}.safetensors"
                    if (
                        not path.is_file()
                        or path.is_symlink()
                        or _sha256_bytes(path.read_bytes()) != item.get("file_sha256")
                    ):
                        reuse_boundaries = False
                        break
                    boundary_paths[item["row_index"]] = path
        if not reuse_boundaries:
            boundary_paths.clear()
            _batched_frozen_prefix_boundaries(
                model,
                unique_rows,
                layers=layers,
                boundary_sink=_persist_boundary,
            )
            files = [
                {
                    "row_index": row_index,
                    "relative_path": path.name,
                    "file_sha256": _sha256_bytes(path.read_bytes()),
                }
                for row_index, path in sorted(boundary_paths.items())
            ]
            boundary_body = {**expected_boundary_authority, "files": files}
            boundary_manifest = {
                **boundary_body,
                "manifest_body_sha256": _sha256_bytes(
                    _canonical_bytes(boundary_body)
                ),
            }
            temporary = boundary_manifest_path.with_suffix(".tmp")
            temporary.write_bytes(_canonical_bytes(boundary_manifest) + b"\n")
            os.replace(temporary, boundary_manifest_path)
        _mem_stage(f"after {len(boundary_paths)} boundaries persisted to disk")

        def _boundary_for(row: PreparedGLM52TeacherRow) -> tuple[mx.array, Any, Any]:
            return _load_boundary(boundary_paths[row.row_index])
    else:
        built_boundaries = _batched_frozen_prefix_boundaries(
            model,
            unique_rows,
            layers=layers,
        )
        boundary_cache = {
            row.row_index: boundary
            for row, boundary in zip(unique_rows, built_boundaries)
        }

        def _boundary_for(row: PreparedGLM52TeacherRow) -> tuple[mx.array, Any, Any]:
            return boundary_cache[row.row_index]

    def loss_fn(
        current: Mapping[str, mx.array],
        row: PreparedGLM52TeacherRow,
        boundary: tuple[mx.array, Any, Any],
    ) -> mx.array:
        _bind_params(projection_map, current)
        forward_result = _selected_logits_from_prefix(
            model,
            row,
            boundary,
            layers=layers,
            output_chunk_size=config.surrogate_output_chunk_size,
            capture_cka_layers=cka_layers,
            capture_router_layers=router_layers,
        )
        if cka_layers or router_layers:
            logits, student_hidden_states, student_router_logits = forward_result
        else:
            logits = forward_result
            student_hidden_states = None
            student_router_logits = None
        return glm52_adapter_loss(
            logits,
            row,
            target_nll_weight=config.target_nll_weight,
            teacher_top1_margin_weight=config.teacher_top1_margin_weight,
            teacher_top1_margin=config.teacher_top1_margin,
            teacher_top1_competitor_token_ids=config.teacher_top1_competitor_token_ids,
            teacher_top1_include_hardest_competitor=config.teacher_top1_include_hardest_competitor,
            tail_kld_weight=config.tail_kld_weight,
            topk=config.topk,
            cka_enabled=config.cka_enabled,
            cka_layers=config.cka_layers,
            student_hidden_states=student_hidden_states,
            router_kl_weight=config.router_kl_weight,
            router_entropy_beta=config.router_entropy_beta,
            mc_expert_explore=config.mc_expert_explore,
            student_router_logits=student_router_logits,
            seed=config.seed,
        )

    value_and_grad = mx.value_and_grad(loss_fn)
    losses = restored_losses
    last_norms: dict[str, mx.array] = {}
    progress_enabled = os.environ.get("GLM52_TRAIN_PROGRESS") == "1"
    frozen_indexers = _freeze_attention_indexers(model, from_layer=min(config.layers))
    last_s3_sync = time.monotonic()
    try:
        for step in range(start_step, config.steps):
            if training_schedule is not None:
                window = scheduled_windows[step]
                source = rows_by_prompt[window.prompt_id]
                row = (
                    source.load_window(window.start, window.end)
                    if isinstance(source, PreparedGLM52TeichDiskRow)
                    else _slice_teich_training_row(
                        source,
                        start=window.start,
                        end=window.end,
                    )
                )
            else:
                row = training_rows[step % len(training_rows)]
            boundary = _boundary_for(row)
            loss, grads = value_and_grad(params, row, boundary)
            if step == 0:
                _mem_stage("after first step forward+backward")
            updated = {}
            norms = {}
            for key, param in params.items():
                grad = grads[key].astype(mx.float32)
                norm = mx.sqrt(mx.sum(grad * grad))
                if config.grad_clip_norm is not None and config.grad_clip_norm > 0:
                    scale = mx.minimum(1.0, config.grad_clip_norm / (norm + 1.0e-12))
                    grad = grad * scale
                updated[key] = (param.astype(mx.float32) - config.learning_rate * grad).astype(mx.float32)
                norms[key] = norm
            mx.eval(loss, *updated.values(), *norms.values())
            params = updated
            losses.append(loss.astype(mx.float32))
            last_norms = norms
            if progress_enabled:
                total_norm = float(mx.sqrt(sum(mx.sum(n * n) for n in norms.values())))
                print(
                    f"[train] step {step + 1}/{config.steps} "
                    f"loss={float(loss):.6f} gradnorm={total_norm:.4f} "
                    f"peakGB={_peak_memory_gb():.1f}",
                    flush=True,
                )
            # Return the per-step transient buffers to the OS. With the ~104 GB
            # composite resident on a 128 GB host the headroom is thin, and the
            # MLX caching allocator otherwise holds each step's peak, creeping
            # into an OOM (jetsam Killed:9) after a few steps.
            mx.clear_cache()
            completed = step + 1
            validation_due = bool(
                validation_fn is not None
                and checkpoint_config is not None
                and completed % checkpoint_config.validation_every_steps == 0
            )
            validation_improved = False
            if validation_due:
                metric = float(validation_fn(params, completed))
                if not np.isfinite(metric):
                    raise ValueError("validation callback returned a non-finite metric")
                if validation_state.best_metric is None or metric < validation_state.best_metric:
                    validation_state = ValidationState(
                        best_metric=metric,
                        best_checkpoint_sha256=None,
                        consecutive_regressions=0,
                    )
                    validation_improved = True
                else:
                    validation_state = ValidationState(
                        best_metric=validation_state.best_metric,
                        best_checkpoint_sha256=validation_state.best_checkpoint_sha256,
                        consecutive_regressions=validation_state.consecutive_regressions + 1,
                    )
            stop_requested = bool(
                checkpoint_config is not None
                and checkpoint_config.stop_file is not None
                and checkpoint_config.stop_file.exists()
            )
            local_due = bool(
                checkpoint_store is not None
                and (
                    checkpoint_config is None
                    or completed % checkpoint_config.local_every_steps == 0
                    or completed == config.steps
                    or validation_due
                    or stop_requested
                )
            )
            published = None
            if local_due and checkpoint_store is not None:
                mx.eval(*params.values(), *losses, *norms.values())
                published = checkpoint_store.publish(
                    params={
                        name: np.ascontiguousarray(np.array(value), dtype=np.float32)
                        for name, value in params.items()
                    },
                    completed_schedule_index=completed,
                    losses=np.asarray(
                        [float(value.item()) for value in losses], dtype=np.float32
                    ),
                    latest_gradient_norms={
                        name: float(value.item()) for name, value in norms.items()
                    },
                    best_validation_metric=validation_state.best_metric,
                    best_checkpoint_sha256=(
                        None
                        if validation_improved
                        else validation_state.best_checkpoint_sha256
                    ),
                    best_is_current=validation_improved,
                    consecutive_validation_regressions=validation_state.consecutive_regressions,
                    rng_state={
                        "training_seed": config.seed,
                        "schedule_offset": completed,
                    },
                )
                if validation_improved:
                    validation_state = ValidationState(
                        best_metric=validation_state.best_metric,
                        best_checkpoint_sha256=published.checkpoint_sha256,
                        consecutive_regressions=0,
                    )
            s3_due = bool(
                checkpoint_sync is not None
                and checkpoint_store is not None
                and checkpoint_config is not None
                and (
                    completed % checkpoint_config.s3_every_steps == 0
                    or time.monotonic() - last_s3_sync >= checkpoint_config.s3_every_seconds
                    or stop_requested
                    or completed == config.steps
                )
            )
            if s3_due and checkpoint_sync is not None and checkpoint_store is not None:
                checkpoint_sync(checkpoint_store.root, completed, stop_requested)
                last_s3_sync = time.monotonic()
            if stop_requested or validation_state.should_stop:
                break
    finally:
        for attention, indexer in frozen_indexers:
            attention.indexer = indexer
        if boundary_dir is not None and persistent_boundary_dir is None:
            shutil.rmtree(boundary_dir, ignore_errors=True)
    _bind_params(projection_map, params)
    return AdapterTrainingResult(params=params, losses=mx.stack(losses), gradient_norms=last_norms)


def _config_manifest(config: AdapterTrainingConfig) -> dict[str, Any]:
    value = asdict(config)
    value["layers"] = list(config.layers)
    value["projections"] = list(config.projections)
    value["cka_layers"] = list(config.cka_layers)
    competitors = config.teacher_top1_competitor_token_ids
    value["teacher_top1_competitor_token_ids"] = None if competitors is None else [list(v) for v in competitors]
    return value


def _manifest_body(manifest: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(manifest)
    body.pop("manifest_body_sha256", None)
    body.pop("candidate_identity_sha256", None)
    return body


def save_authenticated_glm52_adapter(
    adapter_dir: str | Path,
    *,
    params: Mapping[str, mx.array],
    projection_map: Mapping[tuple[int, str], Any],
    parent_candidate_identity_sha256: str,
    teacher_manifest_body_sha256: str,
    training_config: AdapterTrainingConfig,
    selection_rows: Sequence[PreparedGLM52TeacherRow | PreparedGLM52TeichDiskRow],
    release_eligible: bool,
    projection_inventory_attestation: ProjectionInventoryAttestation,
) -> ValidatedGLM52Adapter:
    _require_sha256(parent_candidate_identity_sha256, label="parent candidate identity")
    _require_sha256(teacher_manifest_body_sha256, label="teacher manifest body")
    if not selection_rows:
        raise ValueError("selection_rows must not be empty")
    for row in selection_rows:
        if isinstance(row, PreparedGLM52TeichDiskRow):
            if row.split != "train" or not row.tuning_eligible:
                raise ValueError("GLM52 Teich adapter tuning is train-only")
        else:
            validate_tuning_row(row)
        if row.teacher_manifest_body_sha256 != teacher_manifest_body_sha256:
            raise ValueError("selection row teacher identity mismatch")
    saved_projection_inventory = attest_glm52_projection_inventory(projection_map)
    if saved_projection_inventory != projection_inventory_attestation:
        raise ValueError("routed projection inventory changed between training and save")
    root = Path(adapter_dir)
    if root.exists():
        raise ValueError(f"adapter output directory already exists: {root}")
    sidecar_root = root / SIDECAR_DIRECTORY
    sidecar_root.mkdir(parents=True)
    entries = []
    for layer, projection_name in sorted(projection_map):
        prefix = f"{layer}.{projection_name}"
        left_key = f"{prefix}.low_rank_left"
        right_key = f"{prefix}.low_rank_right"
        left = params.get(left_key)
        right = params.get(right_key)
        if left is None or right is None:
            raise ValueError(f"{prefix} requires both low_rank_left and low_rank_right")
        projection = projection_map[(layer, projection_name)]
        expected_left = (projection.num_experts, projection.output_dims, training_config.rank)
        expected_right = (projection.num_experts, training_config.rank, projection.input_dims)
        if left.shape != expected_left or right.shape != expected_right:
            raise ValueError(f"{prefix} low-rank tensor shapes do not match projection")
        relative = f"{SIDECAR_DIRECTORY}/layer-{layer:05d}-{projection_name}.safetensors"
        path = root / relative
        mx.save_safetensors(
            str(path),
            {
                "low_rank_left": left.astype(mx.float32),
                "low_rank_right": right.astype(mx.float32),
            },
            metadata={"format": "glm52_low_rank_adapter_v1"},
        )
        raw = _read_regular_file(path, label="written adapter sidecar")
        entries.append(
            {
                "layer": layer,
                "projection": projection_name,
                "relative_path": relative,
                "file_sha256": _sha256_bytes(raw),
                "num_experts": projection.num_experts,
                "input_dims": projection.input_dims,
                "output_dims": projection.output_dims,
                "rank": training_config.rank,
                "tensors": {
                    "low_rank_left": {"dtype": "F32", "shape": list(expected_left)},
                    "low_rank_right": {"dtype": "F32", "shape": list(expected_right)},
                },
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "glm52_low_rank_adapter_overlay",
        "parent_candidate_identity_sha256": parent_candidate_identity_sha256,
        "teacher_manifest_body_sha256": teacher_manifest_body_sha256,
        "release_eligible": bool(release_eligible),
        "routed_projection_inventory": {
            "count": saved_projection_inventory.projection_count,
            "metadata_sha256": saved_projection_inventory.projection_metadata_sha256,
        },
        "training_config": _config_manifest(training_config),
        "selection_rows": [
            {
                "row_index": row.row_index,
                "prompt_id": row.prompt_id,
                "positions": list(row.positions),
                "teacher_shard_sha256": row.teacher_shard_sha256,
                "non_release_waiver": row.non_release_waiver,
            }
            for row in selection_rows
        ],
        "sidecars": entries,
    }
    body_sha = _sha256_bytes(_canonical_bytes(manifest))
    candidate_sha = _sha256_bytes(
        _canonical_bytes(
            {
                "parent_candidate_identity_sha256": parent_candidate_identity_sha256,
                "teacher_manifest_body_sha256": teacher_manifest_body_sha256,
                "adapter_manifest_body_sha256": body_sha,
            }
        )
    )
    manifest["manifest_body_sha256"] = body_sha
    manifest["candidate_identity_sha256"] = candidate_sha
    manifest_path = root / ADAPTER_MANIFEST
    partial = manifest_path.with_suffix(".partial")
    partial.write_bytes(_canonical_bytes(manifest) + b"\n")
    os.replace(partial, manifest_path)
    return load_authenticated_glm52_adapter(
        root,
        expected_parent_candidate_identity_sha256=parent_candidate_identity_sha256,
        expected_teacher_manifest_body_sha256=teacher_manifest_body_sha256,
        expected_manifest_body_sha256=body_sha,
        expected_candidate_identity_sha256=candidate_sha,
        expected_num_experts=entries[0]["num_experts"],
    )


def load_authenticated_glm52_adapter(
    adapter_dir: str | Path,
    *,
    expected_parent_candidate_identity_sha256: str,
    expected_teacher_manifest_body_sha256: str,
    expected_manifest_body_sha256: str,
    expected_candidate_identity_sha256: str,
    expected_num_experts: int,
    _sidecar_opener: Callable[..., AuthenticatedFile] = AuthenticatedFile.open,
) -> ValidatedGLM52Adapter:
    root = Path(adapter_dir)
    raw = _read_regular_file(root / ADAPTER_MANIFEST, label="adapter manifest")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("adapter manifest must be an object")
    body_sha = _sha256_bytes(_canonical_bytes(_manifest_body(manifest)))
    if manifest.get("manifest_body_sha256") != body_sha or body_sha != expected_manifest_body_sha256:
        raise ValueError("adapter manifest body SHA-256 mismatch")
    parent = manifest.get("parent_candidate_identity_sha256")
    teacher = manifest.get("teacher_manifest_body_sha256")
    if parent != expected_parent_candidate_identity_sha256:
        raise ValueError("adapter parent candidate identity mismatch")
    if teacher != expected_teacher_manifest_body_sha256:
        raise ValueError("adapter teacher manifest identity mismatch")
    candidate = _sha256_bytes(
        _canonical_bytes(
            {
                "parent_candidate_identity_sha256": parent,
                "teacher_manifest_body_sha256": teacher,
                "adapter_manifest_body_sha256": body_sha,
            }
        )
    )
    if manifest.get("candidate_identity_sha256") != candidate or candidate != expected_candidate_identity_sha256:
        raise ValueError("adapter candidate identity SHA-256 mismatch")
    raw_entries = manifest.get("sidecars")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("adapter sidecar inventory must not be empty")
    expected_root_entries = {ADAPTER_MANIFEST, SIDECAR_DIRECTORY}
    actual_root_entries = {path.name for path in root.iterdir()}
    if actual_root_entries != expected_root_entries:
        raise ValueError("adapter directory contains missing or unexpected entries")
    sidecar_root = root / SIDECAR_DIRECTORY
    if sidecar_root.is_symlink() or not sidecar_root.is_dir():
        raise ValueError("adapter sidecars path must be a real directory")
    declared_sidecar_names = {
        PurePosixPath(entry.get("relative_path", "")).name
        for entry in raw_entries
        if isinstance(entry, dict)
    }
    actual_sidecar_names = {path.name for path in sidecar_root.iterdir()}
    if actual_sidecar_names != declared_sidecar_names:
        raise ValueError("adapter sidecar directory contains missing or unexpected files")
    params: dict[str, mx.array] = {}
    paths: dict[tuple[int, str], Path] = {}
    seen = set()
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise ValueError("adapter sidecar entries must be objects")
        key = (entry.get("layer"), entry.get("projection"))
        if key in seen:
            raise ValueError("adapter layer/projection entries must be unique")
        seen.add(key)
        layer, projection_name = key
        if type(layer) is not int or projection_name not in SUPPORTED_PROJECTIONS:
            raise ValueError("invalid adapter layer/projection entry")
        experts = entry.get("num_experts")
        if experts != expected_num_experts:
            raise ValueError(f"adapter expert count {experts} does not match expected {expected_num_experts}")
        path = _safe_child(root, entry.get("relative_path", ""), label="adapter sidecar path")
        with _sidecar_opener(path, label="adapter sidecar") as authenticated:
            file_raw = authenticated.bytes
            if authenticated.sha256 != entry.get("file_sha256"):
                raise ValueError(f"adapter sidecar {path.name} SHA-256 mismatch")
            tensor_specs = entry.get("tensors")
            if not isinstance(tensor_specs, dict) or set(tensor_specs) != {
                "low_rank_left",
                "low_rank_right",
            }:
                raise ValueError("adapter sidecar requires both low_rank_left and low_rank_right")
            header = _parse_safetensors_bytes(file_raw, label=f"adapter sidecar {path.name}")
            if set(header.tensors) != {"low_rank_left", "low_rank_right"}:
                raise ValueError("adapter safetensors requires both low_rank_left and low_rank_right")
            for tensor_name in ("low_rank_left", "low_rank_right"):
                declared = tensor_specs[tensor_name]
                tensor_header = header.tensors[tensor_name]
                if declared.get("dtype") != "F32" or tensor_header.dtype != "F32":
                    raise ValueError("adapter low-rank tensors must be FP32")
                if tuple(declared.get("shape", ())) != tensor_header.shape:
                    raise ValueError("adapter tensor shape does not match authenticated manifest")
                params[f"{layer}.{projection_name}.{tensor_name}"] = (
                    _read_fp32_tensor_from_bytes(
                        file_raw,
                        header,
                        tensor_name,
                        label=f"adapter sidecar {path.name}",
                    )
                )
        paths[(layer, projection_name)] = path
    return ValidatedGLM52Adapter(
        adapter_dir=root,
        parent_candidate_identity_sha256=parent,
        teacher_manifest_body_sha256=teacher,
        manifest_body_sha256=body_sha,
        candidate_identity_sha256=candidate,
        params=params,
        sidecar_paths=paths,
        release_eligible=manifest.get("release_eligible") is True,
    )


def bind_glm52_low_rank_adapters(
    projection_map: Mapping[tuple[int, str], Any], validated: ValidatedGLM52Adapter
) -> None:
    if set(projection_map) != set(validated.sidecar_paths):
        raise ValueError("projection map must exactly match authenticated adapter sidecars")
    _bind_params(projection_map, validated.params)


def glm52_router_kd_loss(
    student_router_logits: mx.array,
    teacher_topk_ids: mx.array,
    teacher_topk_weights: mx.array,
    *,
    entropy_beta: float = 0.0,
    mc_expert_explore: bool = False,
    seed: int = 20260711,
) -> mx.array:
    """Compatibility name for the now-supported sparse router target contract."""

    return glm52_router_distillation_loss(
        student_router_logits,
        teacher_topk_ids,
        teacher_topk_weights,
        entropy_beta=entropy_beta,
        mc_expert_explore=mc_expert_explore,
        seed=seed,
    )


__all__ = [
    "ADAPTER_MANIFEST",
    "AdapterTrainingConfig",
    "AdapterTrainingResult",
    "GLM52FixedValidationRunner",
    "PreparedGLM52TeacherRow",
    "PreparedGLM52TeichDiskRow",
    "PreparedGLM52TeichTeacherRow",
    "ProjectionInventoryAttestation",
    "ValidatedGLM52TrainingBaseline",
    "ValidatedGLM52Adapter",
    "attest_glm52_projection_inventory",
    "bind_glm52_low_rank_adapters",
    "call_glm52_vq_moe_with_route_local_surrogates",
    "glm52_adapter_loss",
    "glm52_feature_space_linear_cka_loss",
    "glm52_router_kd_loss",
    "glm52_router_distillation_loss",
    "glm52_topk_kl_loss",
    "initialize_low_rank_params",
    "load_authenticated_glm52_adapter",
    "prepare_glm52_teacher_rows",
    "prepare_glm52_teich_teacher_rows",
    "route_local_projection",
    "run_low_rank_training",
    "save_authenticated_glm52_adapter",
    "selected_glm52_hidden_selected_layer",
    "selected_glm52_logits_selected_layer",
    "target_glm52_projection",
    "validate_tuning_row",
    "validate_teich_teacher_row",
]
