"""Layer-major FP32-source GLM-5.2 teacher forward primitives."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TypeAlias

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_causal_mask
from mlx.utils import tree_flatten

from keep.convert.nvfp4 import (
    read_modelopt_nvfp4_weight,
    resolve_modelopt_nvfp4_weight_bundle,
)
from keep.io.load import inspect_safetensors
from keep.io.source_safetensors import (
    read_safetensors_file_header,
    read_safetensors_tensor_bytes,
    read_safetensors_tensor_mlx,
)
from ramp.models.glm52_vq_adapter import GLM52VQModel, Glm52VQMoE


ProjectionSource: TypeAlias = np.ndarray | Callable[[], np.ndarray]
ExpertWeightResolver: TypeAlias = Callable[[int], Mapping[str, ProjectionSource]]
LayerExpertResolver: TypeAlias = Callable[[int], ExpertWeightResolver]

_EXPERT_DECODE_WORKERS = 2
_EXPERT_DECODE_QUEUE_DEPTH = 2
_EXPERT_DECODE_PIPELINE_ENABLED = True

_CHECKPOINT_RE = re.compile(r"^layer-(?P<layer>[0-9]{5})\.safetensors$")
_CHECKPOINT_LEDGER = "checkpoint-ledger.jsonl"
_CHECKPOINT_METADATA_KEYS = {
    "schema_version",
    "record_type",
    "layer_number",
    "layer_identity",
    "valid_mask_sha256",
    "prompt_sha256",
    "source_sha256",
    "profile_sha256",
    "config_sha256",
    "policy_sha256",
    "precision_sha256",
    "previous_checkpoint_sha256",
    "indexshare_state",
    "hidden_sha256",
}
_LEDGER_KEYS = {
    "schema_version",
    "record_type",
    "layer_number",
    "checkpoint_filename",
    "checkpoint_sha256",
    "previous_checkpoint_sha256",
    "valid_mask_sha256",
    "identities_sha256",
    "previous_ledger_record_sha256",
    "record_sha256",
}


class CheckpointValidationError(ValueError):
    """A canonical final checkpoint is malformed or does not match this run."""


class SourceTeacherInterrupted(RuntimeError):
    """Intentional test hook raised after a durable layer checkpoint."""


@dataclass(frozen=True)
class CheckpointIdentities:
    prompt_sha256: str
    source_sha256: str
    profile_sha256: str
    config_sha256: str
    policy_sha256: str
    precision_sha256: str

    def __post_init__(self) -> None:
        for field_name, value in self.as_dict().items():
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")

    def as_dict(self) -> dict[str, str]:
        return {
            "prompt_sha256": self.prompt_sha256,
            "source_sha256": self.source_sha256,
            "profile_sha256": self.profile_sha256,
            "config_sha256": self.config_sha256,
            "policy_sha256": self.policy_sha256,
            "precision_sha256": self.precision_sha256,
        }


@dataclass(frozen=True)
class LayerRouteTrace:
    layer_index: int
    expert_ids: np.ndarray
    scores: np.ndarray
    valid_assignment_count: int
    padded_assignment_count: int


@dataclass(frozen=True)
class StreamingPrecision:
    decoded_weight_dtypes: tuple[str, ...]
    matmul_weight_dtypes: tuple[str, ...]
    hidden_dtype: str
    route_score_dtype: str
    route_output_dtype: str


@dataclass(frozen=True)
class StreamedMoEResult:
    output: mx.array
    expert_indices: mx.array
    route_scores: mx.array
    valid_assignment_count: int
    padded_assignment_count: int
    precision: StreamingPrecision
    decode_workers: int
    decode_queue_depth: int
    observed_peak_decoded_experts: int


@dataclass
class _DecodedExpertResidency:
    active: int = 0
    peak: int = 0
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def decoded(self) -> None:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def released(self) -> None:
        with self._lock:
            self.active -= 1
            if self.active < 0:
                raise AssertionError("decoded expert residency became negative")


def _decode_expert_sources(
    expert_index: int,
    expert_weight_resolver: ExpertWeightResolver,
    residency: _DecodedExpertResidency,
) -> tuple[int, dict[str, ProjectionSource]]:
    sources = expert_weight_resolver(expert_index)
    if set(sources) != {"gate_proj", "up_proj", "down_proj"}:
        raise ValueError(
            "expert resolver must return exactly gate_proj, up_proj, and down_proj"
        )
    # Only gate/up are safe to prefetch.  down_proj is downstream of the
    # activated-product precision guard and must remain lazy so a rejected
    # expert does not perform unnecessary source I/O or retain another weight.
    decoded: dict[str, ProjectionSource] = {
        projection: _load_decoded_projection(sources[projection], projection=projection)
        for projection in ("gate_proj", "up_proj")
    }
    decoded["down_proj"] = sources["down_proj"]
    residency.decoded()
    return expert_index, decoded


def make_modelopt_nvfp4_expert_weight_resolver(
    source_dir: str | Path,
    weight_map: Mapping[str, str],
    *,
    layer_index: int,
) -> ExpertWeightResolver:
    """Build a lazy production resolver that decodes one projection at a time."""

    source_root = Path(source_dir)

    def resolve(expert_index: int) -> Mapping[str, ProjectionSource]:
        if type(expert_index) is not int or expert_index < 0:
            raise ValueError("expert_index must be a non-negative integer")
        sources: dict[str, ProjectionSource] = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            weight_name = (
                f"model.layers.{layer_index}.mlp.experts.{expert_index}."
                f"{projection}.weight"
            )
            bundle = resolve_modelopt_nvfp4_weight_bundle(weight_map, weight_name)
            sources[projection] = partial(
                read_modelopt_nvfp4_weight,
                source_root,
                bundle,
            )
        return sources

    return resolve


def _load_decoded_projection(
    source: ProjectionSource,
    *,
    projection: str,
) -> np.ndarray:
    decoded = source() if callable(source) else source
    if not isinstance(decoded, np.ndarray):
        raise TypeError(f"{projection} resolver result must be a numpy.ndarray")
    if decoded.dtype != np.float32:
        raise ValueError(f"{projection} decoded weight must use float32")
    if decoded.ndim != 2:
        raise ValueError(f"{projection} decoded weight must have rank 2")
    if not decoded.flags.c_contiguous:
        raise ValueError(f"{projection} decoded weight must be C-contiguous")
    if not np.isfinite(decoded).all():
        raise ValueError(f"{projection} decoded weight must be finite")
    return decoded


def _bf16_projection(
    hidden: mx.array,
    source: ProjectionSource,
    *,
    projection: str,
    decoded_dtypes: list[str],
    matmul_dtypes: list[str],
) -> mx.array:
    if hidden.dtype != mx.bfloat16:
        raise ValueError(f"{projection} hidden input must use bfloat16")
    decoded = _load_decoded_projection(source, projection=projection)
    decoded_dtypes.append(str(decoded.dtype))
    weight = mx.array(decoded).astype(mx.bfloat16)
    matmul_dtypes.append(str(weight.dtype))
    # Break the lazy FP32-upload -> BF16-cast graph before the matmul.  Without
    # this barrier an evaluated projection output can keep both weight buffers
    # as active graph dependencies long after the Python references are gone.
    mx.eval(weight)
    del decoded
    mx.clear_cache()
    output = hidden @ weight.T
    if output.dtype != mx.bfloat16:
        raise ValueError(f"{projection} raw matmul output must use bfloat16")
    mx.eval(output)
    del weight
    mx.clear_cache()
    return output


def reduce_route_outputs_route_rank(
    route_outputs: mx.array,
    route_scores: mx.array,
) -> mx.array:
    """Apply FP32 scores and preserve the adapter's route-axis reduction."""

    if route_outputs.dtype != mx.bfloat16:
        raise ValueError("route outputs must use bfloat16")
    if route_scores.dtype != mx.float32:
        raise ValueError("route scores must use float32")
    if route_outputs.shape[:-1] != route_scores.shape:
        raise ValueError("route output and score shapes do not align")
    return (route_outputs * route_scores[..., None]).sum(axis=-2).astype(
        mx.bfloat16
    )


def stream_selected_expert_moe(
    hidden: mx.array,
    *,
    gate: Callable[[mx.array], tuple[mx.array, mx.array]],
    expert_weight_resolver: ExpertWeightResolver,
    shared_expert: Callable[[mx.array], mx.array] | None = None,
    num_routed_experts: int | None = None,
) -> StreamedMoEResult:
    """Evaluate only selected experts while retaining top-k route-slot order."""

    if hidden.ndim != 2:
        raise ValueError("streamed MoE hidden input must have shape [tokens, hidden]")
    if hidden.dtype != mx.bfloat16:
        raise ValueError("streamed MoE hidden input must use bfloat16")

    expert_indices, route_scores = gate(hidden)
    if expert_indices.ndim != 2 or route_scores.shape != expert_indices.shape:
        raise ValueError("router must return aligned [tokens, top_k] arrays")
    if expert_indices.shape[0] != hidden.shape[0]:
        raise ValueError("router token count must match valid hidden rows")
    if route_scores.dtype != mx.float32:
        raise ValueError("router scores must use float32")
    mx.eval(expert_indices, route_scores)

    indices_np = np.array(expert_indices)
    if not np.issubdtype(indices_np.dtype, np.integer):
        raise ValueError("router expert indices must use an integer dtype")
    if np.any(indices_np < 0):
        raise ValueError("router expert indices must be non-negative")
    if num_routed_experts is not None and (
        type(num_routed_experts) is not int
        or num_routed_experts <= 0
        or np.any(indices_np >= num_routed_experts)
    ):
        raise ValueError("router expert indices exceed the routed expert inventory")
    token_count, top_k = indices_np.shape
    hidden_size = int(hidden.shape[-1])
    flat_outputs = mx.zeros(
        (token_count * top_k, hidden_size),
        dtype=mx.bfloat16,
    )
    decoded_dtypes: list[str] = []
    matmul_dtypes: list[str] = []

    expert_order = tuple(sorted(int(value) for value in np.unique(indices_np)))
    residency = _DecodedExpertResidency()
    executor: ThreadPoolExecutor | None = None
    decode_futures: dict[int, Future[tuple[int, dict[str, ProjectionSource]]]] = {}
    next_submit = 0
    try:
        if _EXPERT_DECODE_PIPELINE_ENABLED:
            executor = ThreadPoolExecutor(
                max_workers=_EXPERT_DECODE_WORKERS,
                thread_name_prefix="glm52-expert-decode",
            )
            initial_window = min(
                len(expert_order),
                _EXPERT_DECODE_QUEUE_DEPTH,
            )
            while next_submit < initial_window:
                expert_index = expert_order[next_submit]
                decode_futures[expert_index] = executor.submit(
                    _decode_expert_sources,
                    expert_index,
                    expert_weight_resolver,
                    residency,
                )
                next_submit += 1

        for expert_index in expert_order:
            if executor is None:
                _, sources = _decode_expert_sources(
                    expert_index,
                    expert_weight_resolver,
                    residency,
                )
            else:
                resolved_expert_index, sources = decode_futures.pop(expert_index).result()
                if resolved_expert_index != expert_index:
                    raise AssertionError("decoded expert future changed sorted expert identity")
                if next_submit < len(expert_order):
                    submit_index = expert_order[next_submit]
                    decode_futures[submit_index] = executor.submit(
                        _decode_expert_sources,
                        submit_index,
                        expert_weight_resolver,
                        residency,
                    )
                    next_submit += 1
            flat_route_slots = np.flatnonzero(indices_np.reshape(-1) == expert_index)
            hidden_rows = flat_route_slots // top_k
            expert_hidden = mx.take(
                hidden,
                mx.array(hidden_rows.astype(np.int32)),
                axis=0,
            )
            gate_output = _bf16_projection(
                expert_hidden,
                sources["gate_proj"],
                projection="gate_proj",
                decoded_dtypes=decoded_dtypes,
                matmul_dtypes=matmul_dtypes,
            )
            up_output = _bf16_projection(
                expert_hidden,
                sources["up_proj"],
                projection="up_proj",
                decoded_dtypes=decoded_dtypes,
                matmul_dtypes=matmul_dtypes,
            )
            activated = nn.silu(gate_output) * up_output
            if activated.dtype != mx.bfloat16:
                raise ValueError("activated gate/up product must use bfloat16")
            mx.eval(activated)
            del gate_output
            del up_output
            mx.clear_cache()
            expert_output = _bf16_projection(
                activated,
                sources["down_proj"],
                projection="down_proj",
                decoded_dtypes=decoded_dtypes,
                matmul_dtypes=matmul_dtypes,
            )
            mx.eval(expert_output)
            del activated
            del sources

            # Row-wise scatter (axis=0), not a fully-flattened axis=None
            # scatter: flat_outputs' total element count (token_count * top_k
            # * hidden_size) exceeds MLX's internal axis=None flatten limit
            # (signed 32-bit) for long sessions -- e.g. 65,487 tokens * 8
            # experts-per-token * 6144 hidden = 3.22B elements, over the
            # ~2.15B ceiling. Scattering whole rows never needs to flatten
            # past token_count * top_k (523,896 here), which stays in range.
            row_indices = mx.broadcast_to(
                mx.array(flat_route_slots)[:, None],
                (flat_route_slots.size, hidden_size),
            )
            flat_outputs = mx.put_along_axis(
                flat_outputs,
                row_indices,
                expert_output,
                axis=0,
            )
            # Materialize every functional slot-buffer update so the next expert
            # depends on one buffer, not the complete chain of earlier experts.
            mx.eval(flat_outputs)
            del expert_output
            del expert_hidden
            del row_indices
            del flat_route_slots
            del hidden_rows
            residency.released()
            mx.clear_cache()
    finally:
        if executor is not None:
            for future in decode_futures.values():
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            # Drop completed futures (and their decoded NumPy buffers) before
            # MLX teardown continues on the calling thread.
            decode_futures.clear()

    route_outputs = flat_outputs.reshape(token_count, top_k, hidden_size)
    routed = reduce_route_outputs_route_rank(route_outputs, route_scores)
    if shared_expert is not None:
        shared_output = shared_expert(hidden)
        if shared_output.shape != hidden.shape:
            raise ValueError("shared expert output shape must match hidden input")
        routed = routed + shared_output
    routed = routed.astype(mx.bfloat16)
    mx.eval(routed)
    route_output_dtype = str(route_outputs.dtype)
    if shared_expert is not None:
        del shared_output
    del route_outputs
    del flat_outputs
    mx.clear_cache()

    assignment_count = token_count * top_k
    if int(expert_indices.size) != assignment_count:
        raise AssertionError("router assignment count does not equal tokens * top_k")
    if residency.peak > _EXPERT_DECODE_QUEUE_DEPTH + 1:
        raise AssertionError("decoded expert residency exceeded the bounded pipeline")
    return StreamedMoEResult(
        output=routed,
        expert_indices=expert_indices,
        route_scores=route_scores,
        valid_assignment_count=assignment_count,
        padded_assignment_count=0,
        precision=StreamingPrecision(
            decoded_weight_dtypes=tuple(decoded_dtypes),
            matmul_weight_dtypes=tuple(matmul_dtypes),
            hidden_dtype=str(hidden.dtype),
            route_score_dtype=str(route_scores.dtype),
            route_output_dtype=route_output_dtype,
        ),
        decode_workers=(
            _EXPERT_DECODE_WORKERS if _EXPERT_DECODE_PIPELINE_ENABLED else 0
        ),
        decode_queue_depth=(
            _EXPERT_DECODE_QUEUE_DEPTH if _EXPERT_DECODE_PIPELINE_ENABLED else 0
        ),
        observed_peak_decoded_experts=residency.peak,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _bf16_raw_bytes(array: mx.array) -> bytes:
    if array.dtype != mx.bfloat16:
        raise ValueError("checkpoint hidden state must use bfloat16")
    contiguous = mx.contiguous(array)
    words = np.array(contiguous.view(mx.uint16))
    return np.ascontiguousarray(words.astype("<u2", copy=False)).tobytes()


def _valid_mask_sha256(valid_mask: np.ndarray) -> str:
    if valid_mask.dtype != np.bool_ or valid_mask.ndim != 2:
        raise ValueError("valid-position mask must be a rank-2 boolean array")
    payload = (
        json.dumps(list(valid_mask.shape), separators=(",", ":")).encode("ascii")
        + b"\0"
        + np.ascontiguousarray(valid_mask.astype(np.uint8)).tobytes()
    )
    return _sha256_bytes(payload)


def _genesis_identity(
    *,
    identities: CheckpointIdentities,
    encoded_token_ids: tuple[tuple[int, ...], ...],
    valid_mask_sha256: str,
    embedded_hidden: mx.array,
) -> str:
    return _canonical_sha256(
        {
            "record_type": "glm52_source_teacher_checkpoint_genesis_v1",
            "identities": identities.as_dict(),
            "encoded_token_ids": [list(row) for row in encoded_token_ids],
            "valid_mask_sha256": valid_mask_sha256,
            "embedded_hidden_sha256": _sha256_bytes(_bf16_raw_bytes(embedded_hidden)),
            "indexshare_state": "absent",
        }
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _checkpoint_run_lock(checkpoint_dir: Path):
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint_dir.parent / f".{checkpoint_dir.name}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CheckpointValidationError(
                f"another source-teacher producer holds {lock_path.name}"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _json_object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_checkpoint_ledger(checkpoint_dir: Path) -> list[dict[str, object]]:
    ledger_path = checkpoint_dir / _CHECKPOINT_LEDGER
    if not ledger_path.exists():
        return []
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise CheckpointValidationError(
            "checkpoint ledger must be a regular non-symlink file"
        )
    records: list[dict[str, object]] = []
    try:
        with ledger_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n") or not line.strip():
                    raise ValueError(f"invalid ledger line {line_number}")
                record = json.loads(
                    line,
                    object_pairs_hook=_json_object_no_duplicates,
                    parse_constant=_reject_json_constant,
                )
                if not isinstance(record, dict):
                    raise ValueError(f"ledger line {line_number} must be an object")
                records.append(record)
    except Exception as error:
        raise CheckpointValidationError(
            f"checkpoint ledger is malformed: {error}"
        ) from error
    return records


def _ledger_genesis(checkpoint_genesis_sha256: str) -> str:
    return _canonical_sha256(
        {
            "record_type": "glm52_source_teacher_checkpoint_ledger_genesis_v1",
            "checkpoint_genesis_sha256": checkpoint_genesis_sha256,
        }
    )


def _append_checkpoint_ledger_record(
    checkpoint_dir: Path,
    *,
    layer_number: int,
    checkpoint_sha256: str,
    previous_checkpoint_sha256: str,
    valid_mask_sha256: str,
    identities: CheckpointIdentities,
    previous_ledger_record_sha256: str,
) -> str:
    body: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_source_teacher_checkpoint_ledger_record_v1",
        "layer_number": layer_number,
        "checkpoint_filename": f"layer-{layer_number:05d}.safetensors",
        "checkpoint_sha256": checkpoint_sha256,
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "valid_mask_sha256": valid_mask_sha256,
        "identities_sha256": _canonical_sha256(identities.as_dict()),
        "previous_ledger_record_sha256": previous_ledger_record_sha256,
    }
    record_sha256 = _canonical_sha256(body)
    record = {**body, "record_sha256": record_sha256}
    encoded = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    ledger_path = checkpoint_dir / _CHECKPOINT_LEDGER
    descriptor = os.open(ledger_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("short checkpoint ledger append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(checkpoint_dir)
    return record_sha256


def _checkpoint_metadata(
    *,
    layer_number: int,
    layer_identity: str,
    valid_mask_sha256: str,
    identities: CheckpointIdentities,
    previous_checkpoint_sha256: str,
    hidden_sha256: str,
) -> dict[str, str]:
    return {
        "schema_version": "1",
        "record_type": "glm52_source_teacher_layer_checkpoint_v1",
        "layer_number": str(layer_number),
        "layer_identity": layer_identity,
        "valid_mask_sha256": valid_mask_sha256,
        **identities.as_dict(),
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "indexshare_state": "absent",
        "hidden_sha256": hidden_sha256,
    }


def _write_layer_checkpoint(
    checkpoint_dir: Path,
    *,
    layer_number: int,
    layer_identity: str,
    hidden: mx.array,
    valid_mask_sha256: str,
    identities: CheckpointIdentities,
    previous_checkpoint_sha256: str,
    previous_ledger_record_sha256: str,
) -> tuple[str, str]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / f"layer-{layer_number:05d}.safetensors"
    if final_path.exists() or final_path.is_symlink():
        raise CheckpointValidationError(
            f"refusing to overwrite existing checkpoint {final_path.name}"
        )
    hidden = mx.contiguous(hidden.astype(mx.bfloat16))
    mx.eval(hidden)
    hidden_sha256 = _sha256_bytes(_bf16_raw_bytes(hidden))
    metadata = _checkpoint_metadata(
        layer_number=layer_number,
        layer_identity=layer_identity,
        valid_mask_sha256=valid_mask_sha256,
        identities=identities,
        previous_checkpoint_sha256=previous_checkpoint_sha256,
        hidden_sha256=hidden_sha256,
    )
    temporary_path = checkpoint_dir / (
        f".{final_path.stem}.{uuid.uuid4().hex}.tmp.safetensors"
    )
    mx.save_safetensors(str(temporary_path), {"hidden": hidden}, metadata=metadata)
    _fsync_file(temporary_path)
    _, raw_hidden = read_safetensors_tensor_bytes(temporary_path, "hidden")
    if _sha256_bytes(raw_hidden) != hidden_sha256:
        raise RuntimeError("serialized BF16 checkpoint bytes differ from hidden state")
    if final_path.exists() or final_path.is_symlink():
        raise CheckpointValidationError(
            f"refusing to overwrite concurrently published checkpoint {final_path.name}"
        )
    os.replace(temporary_path, final_path)
    _fsync_directory(checkpoint_dir)
    checkpoint_sha256 = _sha256_file(final_path)
    ledger_record_sha256 = _append_checkpoint_ledger_record(
        checkpoint_dir,
        layer_number=layer_number,
        checkpoint_sha256=checkpoint_sha256,
        previous_checkpoint_sha256=previous_checkpoint_sha256,
        valid_mask_sha256=valid_mask_sha256,
        identities=identities,
        previous_ledger_record_sha256=previous_ledger_record_sha256,
    )
    return checkpoint_sha256, ledger_record_sha256


def _expected_checkpoint_metadata(
    *,
    layer_number: int,
    valid_mask_sha256: str,
    identities: CheckpointIdentities,
    previous_checkpoint_sha256: str,
) -> dict[str, str]:
    return _checkpoint_metadata(
        layer_number=layer_number,
        layer_identity=f"model.layers.{layer_number}",
        valid_mask_sha256=valid_mask_sha256,
        identities=identities,
        previous_checkpoint_sha256=previous_checkpoint_sha256,
        hidden_sha256="",
    )


def _validate_layer_checkpoint(
    path: Path,
    *,
    layer_number: int,
    hidden_shape: tuple[int, int, int],
    valid_mask_sha256: str,
    identities: CheckpointIdentities,
    previous_checkpoint_sha256: str,
) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise CheckpointValidationError(
            f"checkpoint {path.name} must be a regular non-symlink file"
        )
    try:
        inspection = inspect_safetensors(path)
        file_header = read_safetensors_file_header(path)
        tensor_header, raw_hidden = read_safetensors_tensor_bytes(path, "hidden")
    except Exception as error:
        raise CheckpointValidationError(
            f"checkpoint {path.name} is malformed: {error}"
        ) from error

    if set(inspection.metadata) != _CHECKPOINT_METADATA_KEYS:
        raise CheckpointValidationError(
            f"checkpoint {path.name} metadata keys do not match the strict schema"
        )
    expected = _expected_checkpoint_metadata(
        layer_number=layer_number,
        valid_mask_sha256=valid_mask_sha256,
        identities=identities,
        previous_checkpoint_sha256=previous_checkpoint_sha256,
    )
    expected_hidden_sha256 = inspection.metadata.get("hidden_sha256", "")
    expected["hidden_sha256"] = expected_hidden_sha256
    if inspection.metadata != expected:
        raise CheckpointValidationError(
            f"checkpoint {path.name} identity or chain metadata mismatch"
        )
    if set(file_header.tensors) != {"hidden"}:
        raise CheckpointValidationError(
            f"checkpoint {path.name} must contain exactly one hidden tensor"
        )
    if tensor_header.dtype != "BF16" or tensor_header.shape != hidden_shape:
        raise CheckpointValidationError(
            f"checkpoint {path.name} hidden tensor contract mismatch"
        )
    if tensor_header.data_offsets != (0, len(raw_hidden)):
        raise CheckpointValidationError(
            f"checkpoint {path.name} hidden tensor offsets are not canonical"
        )
    physical_bytes = path.stat().st_size
    expected_bytes = file_header.payload_offset + file_header.declared_payload_bytes
    if physical_bytes != expected_bytes:
        raise CheckpointValidationError(
            f"checkpoint {path.name} physical extent mismatch"
        )
    hidden_sha256 = _sha256_bytes(raw_hidden)
    if hidden_sha256 != expected_hidden_sha256:
        raise CheckpointValidationError(
            f"checkpoint {path.name} hidden SHA-256 mismatch"
        )
    return _sha256_file(path), hidden_sha256


def _resume_checkpoint_chain(
    checkpoint_dir: Path,
    *,
    num_layers: int,
    hidden_shape: tuple[int, int, int],
    valid_mask_sha256: str,
    identities: CheckpointIdentities,
    genesis_sha256: str,
) -> tuple[int, mx.array | None, str, str]:
    ledger_genesis_sha256 = _ledger_genesis(genesis_sha256)
    if not checkpoint_dir.exists():
        return 0, None, genesis_sha256, ledger_genesis_sha256
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise CheckpointValidationError(
            "checkpoint_dir must be a regular non-symlink directory"
        )
    final_layers: list[int] = []
    for path in checkpoint_dir.iterdir():
        match = _CHECKPOINT_RE.fullmatch(path.name)
        if match is not None:
            final_layers.append(int(match.group("layer")))
    if not final_layers:
        if _read_checkpoint_ledger(checkpoint_dir):
            raise CheckpointValidationError(
                "checkpoint ledger exists without canonical final checkpoints"
            )
        return 0, None, genesis_sha256, ledger_genesis_sha256
    final_layers.sort()
    highest = final_layers[-1]
    if highest >= num_layers:
        raise CheckpointValidationError(
            f"checkpoint layer {highest} is outside the {num_layers}-layer model"
        )
    expected_layers = list(range(highest + 1))
    if final_layers != expected_layers:
        raise CheckpointValidationError(
            "final checkpoints do not form a contiguous chain from layer 0"
        )

    ledger_records = _read_checkpoint_ledger(checkpoint_dir)
    if len(ledger_records) != len(expected_layers):
        raise CheckpointValidationError(
            "canonical final checkpoints and ledger records are not one-to-one"
        )

    previous_sha256 = genesis_sha256
    previous_ledger_sha256 = ledger_genesis_sha256
    for layer_number, ledger_record in zip(
        expected_layers,
        ledger_records,
        strict=True,
    ):
        if set(ledger_record) != _LEDGER_KEYS:
            raise CheckpointValidationError(
                f"checkpoint ledger record {layer_number} keys do not match the strict schema"
            )
        record_body = dict(ledger_record)
        record_sha256 = record_body.pop("record_sha256", None)
        if record_sha256 != _canonical_sha256(record_body):
            raise CheckpointValidationError(
                f"checkpoint ledger record {layer_number} SHA-256 mismatch"
            )
        expected_ledger_fields = {
            "schema_version": 1,
            "record_type": "glm52_source_teacher_checkpoint_ledger_record_v1",
            "layer_number": layer_number,
            "checkpoint_filename": f"layer-{layer_number:05d}.safetensors",
            "previous_checkpoint_sha256": previous_sha256,
            "valid_mask_sha256": valid_mask_sha256,
            "identities_sha256": _canonical_sha256(identities.as_dict()),
            "previous_ledger_record_sha256": previous_ledger_sha256,
        }
        for key, expected_value in expected_ledger_fields.items():
            if ledger_record.get(key) != expected_value:
                raise CheckpointValidationError(
                    f"checkpoint ledger record {layer_number} identity or chain mismatch"
                )
        path = checkpoint_dir / f"layer-{layer_number:05d}.safetensors"
        highest_previous_checkpoint_sha256 = previous_sha256
        checkpoint_sha256, hidden_sha256 = _validate_layer_checkpoint(
            path,
            layer_number=layer_number,
            hidden_shape=hidden_shape,
            valid_mask_sha256=valid_mask_sha256,
            identities=identities,
            previous_checkpoint_sha256=previous_sha256,
        )
        if ledger_record.get("checkpoint_sha256") != checkpoint_sha256:
            raise CheckpointValidationError(
                f"checkpoint {path.name} whole-file SHA-256 does not match its ledger"
            )
        previous_sha256 = checkpoint_sha256
        previous_ledger_sha256 = str(record_sha256)
    try:
        hidden = read_safetensors_tensor_mlx(path, "hidden")
        mx.eval(hidden)
        loaded_hidden_sha256 = _sha256_bytes(_bf16_raw_bytes(hidden))
        (
            revalidated_checkpoint_sha256,
            revalidated_hidden_sha256,
        ) = _validate_layer_checkpoint(
            path,
            layer_number=highest,
            hidden_shape=hidden_shape,
            valid_mask_sha256=valid_mask_sha256,
            identities=identities,
            previous_checkpoint_sha256=highest_previous_checkpoint_sha256,
        )
    except Exception as error:
        raise CheckpointValidationError(
            f"checkpoint {path.name} changed between validation and load: {error}"
        ) from error
    if (
        revalidated_checkpoint_sha256 != checkpoint_sha256
        or revalidated_hidden_sha256 != hidden_sha256
        or loaded_hidden_sha256 != hidden_sha256
    ):
        raise CheckpointValidationError(
            f"checkpoint {path.name} changed between validation and load"
        )
    return highest + 1, hidden, previous_sha256, previous_ledger_sha256


def _scatter_valid_rows(
    valid_output: mx.array,
    *,
    valid_flat_indices: np.ndarray,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
) -> mx.array:
    # Row-wise scatter (axis=0), matching stream_selected_expert_moe's fix:
    # axis=None requires flattening `flat` (batch_size * sequence_length *
    # hidden_size elements), which overflows MLX's internal signed 32-bit
    # limit for long sessions. Row-wise scatter never flattens past
    # batch_size * sequence_length.
    flat = mx.zeros((batch_size * sequence_length, hidden_size), dtype=mx.bfloat16)
    row_indices = mx.broadcast_to(
        mx.array(valid_flat_indices)[:, None],
        (valid_flat_indices.size, hidden_size),
    )
    flat = mx.put_along_axis(flat, row_indices, valid_output, axis=0)
    return flat.reshape(batch_size, sequence_length, hidden_size)


def _prepare_predictor_batch(
    encoded_token_ids: tuple[tuple[int, ...], ...],
    *,
    vocab_size: int,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not encoded_token_ids:
        raise ValueError("at least one encoded prompt is required")
    for prompt_index, row in enumerate(encoded_token_ids):
        if len(row) < 2:
            raise ValueError(
                f"encoded prompt {prompt_index} must contain at least two tokens"
            )
        if any(
            type(token) is not int or token < 0 or token >= vocab_size
            for token in row
        ):
            raise ValueError(
                f"encoded prompt {prompt_index} contains an invalid token ID"
            )
    predictor_lengths = np.array(
        [len(row) - 1 for row in encoded_token_ids],
        dtype=np.int32,
    )
    max_length = int(predictor_lengths.max())
    token_batch = np.full(
        (len(encoded_token_ids), max_length),
        pad_token_id,
        dtype=np.int32,
    )
    valid_mask = np.zeros(token_batch.shape, dtype=np.bool_)
    for row_index, row in enumerate(encoded_token_ids):
        predictor = row[:-1]
        token_batch[row_index, : len(predictor)] = predictor
        valid_mask[row_index, : len(predictor)] = True
    right_padding = max_length - predictor_lengths
    return token_batch, valid_mask, right_padding


def _audit_source_teacher_model_precision(model: GLM52VQModel) -> None:
    mismatches: list[str] = []
    for parameter_name, value in tree_flatten(model.parameters()):
        if not mx.issubdtype(value.dtype, mx.floating):
            continue
        expected_dtype = (
            mx.float32
            if parameter_name.endswith("mlp.gate.e_score_correction_bias")
            else mx.bfloat16
        )
        if value.dtype != expected_dtype:
            mismatches.append(
                f"{parameter_name}={value.dtype}, expected={expected_dtype}"
            )
    if mismatches:
        raise ValueError(
            "GLM52 source-teacher model precision contract failed: "
            + "; ".join(mismatches)
        )


def _run_glm52_source_teacher_unlocked(
    model: GLM52VQModel,
    encoded_token_ids: (
        tuple[tuple[int, ...], ...] | list[list[int]] | tuple[list[int], ...]
    ),
    *,
    expert_resolver_for_layer: LayerExpertResolver,
    expected_num_layers: int,
    pad_token_id: int = 0,
    capture_route_trace: bool = False,
    checkpoint_dir: str | Path | None = None,
    checkpoint_identities: CheckpointIdentities | None = None,
    interrupt_after_layer: int | None = None,
) -> tuple[tuple[np.ndarray, ...], tuple[LayerRouteTrace, ...] | None]:
    """Run a right-padded, layer-major GLM-5.2 source-teacher forward pass."""

    if not isinstance(model, GLM52VQModel):
        raise TypeError("model must be a GLM52VQModel")
    if type(expected_num_layers) is not int or expected_num_layers <= 0:
        raise ValueError("expected_num_layers must be a positive integer")
    if model.args.num_hidden_layers != expected_num_layers:
        raise ValueError(
            f"num_hidden_layers={model.args.num_hidden_layers}, "
            f"expected_num_layers={expected_num_layers}"
        )
    if model.model.start_idx != 0 or model.model.end_idx != expected_num_layers:
        raise ValueError("source teacher requires the expected complete main-layer range")
    if len(model.model.layers) != expected_num_layers:
        raise ValueError("model layer inventory does not match expected_num_layers")
    prompts = tuple(tuple(row) for row in encoded_token_ids)
    if type(pad_token_id) is not int or not 0 <= pad_token_id < model.args.vocab_size:
        raise ValueError("pad_token_id must be a valid vocabulary index")
    token_batch, valid_mask, right_padding = _prepare_predictor_batch(
        prompts,
        vocab_size=model.args.vocab_size,
        pad_token_id=pad_token_id,
    )
    batch_size, sequence_length = token_batch.shape
    if sequence_length > model.args.index_topk:
        raise ValueError(
            "source-teacher checkpoints currently require short runs with absent "
            "IndexShare state"
        )
    _audit_source_teacher_model_precision(model)
    if interrupt_after_layer is not None:
        if (
            type(interrupt_after_layer) is not int
            or not 0 <= interrupt_after_layer < expected_num_layers
        ):
            raise ValueError("interrupt_after_layer is outside the main-layer range")
        if checkpoint_dir is None:
            raise ValueError("interrupt_after_layer requires checkpoint_dir")

    tokens = mx.array(token_batch)
    hidden = model.model.embed_tokens(tokens)
    if hidden.dtype != mx.bfloat16:
        raise ValueError("embedded source-teacher hidden state must use bfloat16")
    hidden = mx.contiguous(hidden)
    mx.eval(hidden)
    causal_mask = create_causal_mask(
        sequence_length,
        right_padding=mx.array(right_padding),
    )
    mx.eval(causal_mask)
    valid_flat_indices = np.flatnonzero(valid_mask.reshape(-1)).astype(np.int64)
    valid_mask_digest = _valid_mask_sha256(valid_mask)
    hidden_shape = (batch_size, sequence_length, model.args.hidden_size)

    start_layer = 0
    previous_checkpoint_sha256 = ""
    previous_ledger_record_sha256 = ""
    if checkpoint_dir is not None:
        if checkpoint_identities is None:
            raise ValueError("checkpoint_identities are required with checkpoint_dir")
        checkpoint_root = Path(checkpoint_dir)
        genesis_sha256 = _genesis_identity(
            identities=checkpoint_identities,
            encoded_token_ids=prompts,
            valid_mask_sha256=valid_mask_digest,
            embedded_hidden=hidden,
        )
        (
            start_layer,
            resumed_hidden,
            previous_checkpoint_sha256,
            previous_ledger_record_sha256,
        ) = _resume_checkpoint_chain(
            checkpoint_root,
            num_layers=expected_num_layers,
            hidden_shape=hidden_shape,
            valid_mask_sha256=valid_mask_digest,
            identities=checkpoint_identities,
            genesis_sha256=genesis_sha256,
        )
        if resumed_hidden is not None:
            if capture_route_trace:
                raise CheckpointValidationError(
                    "route traces cannot be reconstructed from hidden-only checkpoints"
                )
            hidden = resumed_hidden
    elif checkpoint_identities is not None:
        raise ValueError("checkpoint_identities require checkpoint_dir")

    route_trace: list[LayerRouteTrace] = []
    prev_topk_indices = None
    for layer_number in range(start_layer, expected_num_layers):
        layer = model.model.layers[layer_number]
        if int(layer.layer_idx) != layer_number:
            raise ValueError("decoder layer identity does not match its main-layer index")
        attention_output, next_topk_indices = layer.self_attn(
            layer.input_layernorm(hidden),
            causal_mask,
            None,
            prev_topk_indices,
        )
        if next_topk_indices is not None:
            raise ValueError("short source-teacher run must record IndexShare state absent")
        prev_topk_indices = next_topk_indices
        attention_hidden = hidden + attention_output
        mlp_hidden = layer.post_attention_layernorm(attention_hidden)

        if isinstance(layer.mlp, Glm52VQMoE):
            flat_mlp_hidden = mlp_hidden.reshape(-1, model.args.hidden_size)
            valid_hidden = mx.take(
                flat_mlp_hidden,
                mx.array(valid_flat_indices),
                axis=0,
            )
            if valid_hidden.dtype != mx.bfloat16:
                raise ValueError("routed source-teacher hidden state must use bfloat16")
            shared_expert = layer.mlp.get("shared_experts")
            streamed = stream_selected_expert_moe(
                valid_hidden,
                gate=layer.mlp.gate,
                expert_weight_resolver=expert_resolver_for_layer(layer_number),
                shared_expert=shared_expert,
                num_routed_experts=model.args.n_routed_experts,
            )
            expected_assignments = len(valid_flat_indices) * model.args.num_experts_per_tok
            if streamed.valid_assignment_count != expected_assignments:
                raise AssertionError(
                    "sparse-layer valid assignment count does not equal n_valid * top_k"
                )
            if streamed.padded_assignment_count != 0:
                raise AssertionError("padded rows entered sparse router accounting")
            sparse_output = _scatter_valid_rows(
                streamed.output,
                valid_flat_indices=valid_flat_indices,
                batch_size=batch_size,
                sequence_length=sequence_length,
                hidden_size=model.args.hidden_size,
            )
            hidden = attention_hidden + sparse_output
            if capture_route_trace:
                expert_ids_np = np.ascontiguousarray(
                    np.array(streamed.expert_indices)
                )
                scores_np = np.ascontiguousarray(
                    np.array(streamed.route_scores).astype(np.float32, copy=False)
                )
                route_trace.append(
                    LayerRouteTrace(
                        layer_index=layer_number,
                        expert_ids=expert_ids_np,
                        scores=scores_np,
                        valid_assignment_count=streamed.valid_assignment_count,
                        padded_assignment_count=streamed.padded_assignment_count,
                    )
                )
        else:
            hidden = attention_hidden + layer.mlp(mlp_hidden)

        if hidden.dtype != mx.bfloat16:
            raise ValueError("post-layer source-teacher hidden state must use bfloat16")
        hidden = mx.contiguous(hidden)
        mx.eval(hidden)
        if checkpoint_dir is not None:
            assert checkpoint_identities is not None
            (
                previous_checkpoint_sha256,
                previous_ledger_record_sha256,
            ) = _write_layer_checkpoint(
                Path(checkpoint_dir),
                layer_number=layer_number,
                layer_identity=f"model.layers.{layer_number}",
                hidden=hidden,
                valid_mask_sha256=valid_mask_digest,
                identities=checkpoint_identities,
                previous_checkpoint_sha256=previous_checkpoint_sha256,
                previous_ledger_record_sha256=previous_ledger_record_sha256,
            )
        del attention_output
        del attention_hidden
        del mlp_hidden
        if isinstance(layer.mlp, Glm52VQMoE):
            del flat_mlp_hidden
            del valid_hidden
            del streamed
            del sparse_output
            del shared_expert
        mx.clear_cache()
        if interrupt_after_layer == layer_number:
            raise SourceTeacherInterrupted(
                f"source-teacher interrupted after durable layer {layer_number} checkpoint"
            )

    normalized = model.model.norm(hidden)
    if normalized.dtype != mx.bfloat16:
        raise ValueError("final normalized source-teacher hidden state must use bfloat16")
    predictor_lengths = valid_mask.sum(axis=1).astype(np.int64)
    logits_by_prompt: list[np.ndarray] = []
    for prompt_index, predictor_length in enumerate(predictor_lengths):
        prompt_hidden = normalized[
            prompt_index : prompt_index + 1,
            : int(predictor_length),
            :,
        ]
        if prompt_hidden.dtype != mx.bfloat16:
            raise ValueError("LM-head input must use bfloat16")
        prompt_logits = model.lm_head(prompt_hidden).astype(mx.float32).reshape(
            int(predictor_length),
            model.args.vocab_size,
        )
        mx.eval(prompt_logits)
        stored_logits = np.ascontiguousarray(
            np.array(prompt_logits).astype(np.float32, copy=False)
        )
        if stored_logits.dtype != np.float32:
            raise AssertionError("stored source-teacher logits must use float32")
        logits_by_prompt.append(stored_logits)

    return (
        tuple(logits_by_prompt),
        tuple(route_trace) if capture_route_trace else None,
    )


def run_glm52_source_teacher(
    model: GLM52VQModel,
    encoded_token_ids: (
        tuple[tuple[int, ...], ...] | list[list[int]] | tuple[list[int], ...]
    ),
    *,
    expert_resolver_for_layer: LayerExpertResolver,
    expected_num_layers: int,
    pad_token_id: int = 0,
    capture_route_trace: bool = False,
    checkpoint_dir: str | Path | None = None,
    checkpoint_identities: CheckpointIdentities | None = None,
    interrupt_after_layer: int | None = None,
) -> tuple[tuple[np.ndarray, ...], tuple[LayerRouteTrace, ...] | None]:
    """Run with a run-specific lock; the caller retains the global lock duty."""

    lock = (
        _checkpoint_run_lock(Path(checkpoint_dir))
        if checkpoint_dir is not None
        else nullcontext()
    )
    with lock:
        return _run_glm52_source_teacher_unlocked(
            model,
            encoded_token_ids,
            expert_resolver_for_layer=expert_resolver_for_layer,
            expected_num_layers=expected_num_layers,
            pad_token_id=pad_token_id,
            capture_route_trace=capture_route_trace,
            checkpoint_dir=checkpoint_dir,
            checkpoint_identities=checkpoint_identities,
            interrupt_after_layer=interrupt_after_layer,
        )
