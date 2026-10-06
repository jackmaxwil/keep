"""Durable full-v2 teacher-signal production for the Teich corpus.

Produces the teacher signal consumed by :mod:`glm52_distill_loss`
(``topk_logit_ids`` / ``topk_logit_values`` / ``tail_mass``) for every
supervised position of the 257-session teich prompt pack, by running the
FP32-source GLM-5.2-REAP-504B teacher forward.

This is a *personal-model* production path.  It deliberately bypasses the
frozen 66-row eval-pack contract (which pins pack SHA-256, split counts, and
``tuning_eligible == (split == "selection")``) while keeping the useful
integrity work: source config/index/non-VQ identities are computed and
recorded in the run manifest, and every session output binds the prompt's
``token_ids_sha256``.

Differences from the committed eval producer
(:func:`glm52_teacher_cache_producer._run_glm52_source_teacher_to_sink`):

- **Long context.** The ``sequence_length <= index_topk`` guard is removed and
  the DSA IndexShare top-k indices are threaded across layers (the committed
  loop already carries the variable but forbids a non-absent state).  Every
  4th layer computes the sparse index; intermediate layers reuse it.
- **Chunked LM head + on-device top-K.** Logits are never materialised for a
  whole session; supervised positions are gathered and projected in bounded
  slices, top-K'd on device, and only ``[positions, K]`` reaches the host.
- **Four-layer durable checkpoints.** BF16 hidden state and DSA IndexShare
  state are authenticated together, so interruption after an index layer is
  resumable without changing the subsequent sharing layers.
- **Full v2 signal.** Supervised-position-only layer-77 CKA and worst-eight
  router targets are copied while their layer is resident; full-sequence probe
  tensors are never retained across later layers.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load as load_safetensors
from safetensors.numpy import save as save_safetensors

from mlx_vq.quality.glm52_teich_checkpoint import (
    DeadlinePolicy,
    LocalLMHeadSliceStore,
    LocalTeacherCheckpointStore,
    TeacherCheckpointConfig,
    should_checkpoint_after_layer,
)

__all__ = [
    "TeichSession",
    "load_teich_prompt_rows",
    "pack_sessions_into_chunks",
    "topk_capture_numpy",
    "validate_full_v2_capture",
    "load_teich_source_workload",
    "run_teich_teacher_production",
    "TeacherStopRequested",
]

_DEFAULT_TOP_K = 2048
_DEFAULT_LM_HEAD_SLICE = 4096
_SKIP_POST_LAYER_CLEAR_CACHE_ENV = "GLM52_TEICH_SKIP_POST_LAYER_CLEAR_CACHE"
_ROUTER_LAYERS = tuple(range(70, 78))


class TeacherStopRequested(RuntimeError):
    """The campaign reached an intentional, durably checkpointed stop boundary."""


def _clear_post_layer_cache(clear_cache: Callable[[], None]) -> None:
    """Clear the layer cache unless the CUDA compatibility rail is enabled."""

    if os.environ.get(_SKIP_POST_LAYER_CLEAR_CACHE_ENV) == "1":
        return
    clear_cache()


# ---------------------------------------------------------------------------
# Prompt pack loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeichSession:
    """One teich session: the prompt plus its supervised positions."""

    prompt: Any  # GLM52TeacherCachePrompt
    positions: np.ndarray  # int64 [P] — logits row p predicts token p+1
    target_token_ids: np.ndarray  # int64 [P]

    @property
    def prompt_id(self) -> str:
        return self.prompt.prompt_id

    @property
    def token_count(self) -> int:
        return self.prompt.token_count


def load_teich_prompt_rows(pack_path: str | Path) -> tuple[TeichSession, ...]:
    """Load and validate the teich prompt pack into sessions.

    Validates per row that ``target_token_ids[i] == encoded_token_ids[pos+1]``
    (the bridge's supervised-position convention) so a mis-built pack cannot
    silently produce misaligned teacher targets.
    """

    cache_api = importlib.import_module("mlx_vq.quality.glm52_teacher_cache")
    payload = json.loads(Path(pack_path).read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("teich pack must be a JSON object")
    rows = payload.get("prompt_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("teich pack must contain a non-empty prompt_rows list")
    declared = payload.get("prompt_row_count")
    if declared is not None and declared != len(rows):
        raise ValueError("teich pack prompt_row_count does not match prompt_rows")
    sessions: list[TeichSession] = []
    for index, row in enumerate(rows):
        prompt = cache_api.GLM52TeacherCachePrompt.from_prompt_pack_row(row)
        if row.get("token_ids_sha256") != prompt.token_ids_sha256:
            raise ValueError(f"prompt_rows[{index}] token_ids_sha256 mismatch")
        positions = np.asarray(row.get("positions"), dtype=np.int64)
        targets = np.asarray(row.get("target_token_ids"), dtype=np.int64)
        if positions.ndim != 1 or positions.shape != targets.shape or positions.size == 0:
            raise ValueError(f"prompt_rows[{index}] positions/targets malformed")
        if np.any(positions < 0) or np.any(positions >= prompt.token_count - 1):
            raise ValueError(f"prompt_rows[{index}] positions out of predictor range")
        if np.any(np.diff(positions) <= 0):
            raise ValueError(f"prompt_rows[{index}] positions must be strictly increasing")
        encoded = np.asarray(prompt.encoded_token_ids, dtype=np.int64)
        if not np.array_equal(encoded[positions + 1], targets):
            raise ValueError(
                f"prompt_rows[{index}] target_token_ids do not equal encoded[pos+1]"
            )
        sessions.append(
            TeichSession(prompt=prompt, positions=positions, target_token_ids=targets)
        )
    ids = [session.prompt_id for session in sessions]
    if len(set(ids)) != len(ids):
        raise ValueError("teich pack prompt IDs must be unique")
    return tuple(sessions)


def pack_sessions_into_chunks(
    sessions: Sequence[TeichSession],
    *,
    max_batch_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    """Group session indices into padded-batch chunks under a token budget.

    The layer-major forward pads every prompt in a chunk to the longest one,
    so the budget is ``batch_size * max_len``.  Sessions are packed longest
    first so similarly sized sessions share a chunk (minimising pad waste).
    Returns tuples of indices into ``sessions``.
    """

    if max_batch_tokens <= 0:
        raise ValueError("max_batch_tokens must be positive")
    order = sorted(range(len(sessions)), key=lambda i: -sessions[i].token_count)
    chunks: list[tuple[int, ...]] = []
    current: list[int] = []
    current_max = 0
    for index in order:
        length = sessions[index].token_count
        if length > max_batch_tokens:
            raise ValueError(
                f"session {sessions[index].prompt_id} ({length} tokens) exceeds "
                f"max_batch_tokens={max_batch_tokens}"
            )
        candidate_max = max(current_max, length)
        if current and candidate_max * (len(current) + 1) > max_batch_tokens:
            chunks.append(tuple(current))
            current = [index]
            current_max = length
        else:
            current.append(index)
            current_max = candidate_max
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


# ---------------------------------------------------------------------------
# Top-K capture
# ---------------------------------------------------------------------------


def topk_capture_numpy(
    logits: np.ndarray, *, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference top-K capture: (ids, values, logsumexp_full, tail_mass).

    ``ids`` are sorted by descending logit.  ``tail_mass`` is the exact
    teacher probability outside the top-K, computed against the full-vocab
    logsumexp — the quantity :func:`topk_aggregated_tail_kl_numpy` consumes.
    """

    if logits.ndim != 2:
        raise ValueError("logits must be [positions, vocab]")
    if not 0 < k <= logits.shape[1]:
        raise ValueError("k must be in (0, vocab]")
    work = logits.astype(np.float64)
    row_max = work.max(axis=1, keepdims=True)
    logsumexp = (np.log(np.exp(work - row_max).sum(axis=1)) + row_max[:, 0]).astype(
        np.float32
    )
    part = np.argpartition(-work, k - 1, axis=1)[:, :k]
    part_vals = np.take_along_axis(work, part, axis=1)
    order = np.argsort(-part_vals, axis=1, kind="stable")
    ids = np.take_along_axis(part, order, axis=1).astype(np.int32)
    values = np.take_along_axis(part_vals, order, axis=1)
    topk_mass = np.exp(values - logsumexp[:, None].astype(np.float64)).sum(axis=1)
    tail_mass = np.clip(1.0 - topk_mass, 0.0, None).astype(np.float32)
    return ids, values.astype(np.float32), logsumexp, tail_mass


def _topk_capture_mlx(logits_2d: Any, *, k: int) -> tuple[np.ndarray, ...]:
    """On-device top-K capture; returns host arrays matching the numpy ref."""

    mx = importlib.import_module("mlx.core")
    logits32 = logits_2d.astype(mx.float32)
    logsumexp = mx.logsumexp(logits32, axis=1)
    neg_sorted = mx.argsort(-logits32, axis=1)[:, :k]
    values = mx.take_along_axis(logits32, neg_sorted, axis=1)
    topk_mass = mx.exp(values - logsumexp[:, None]).sum(axis=1)
    tail = mx.maximum(1.0 - topk_mass, 0.0)
    mx.eval(neg_sorted, values, logsumexp, tail)
    return (
        np.array(neg_sorted).astype(np.int32),
        np.array(values).astype(np.float32),
        np.array(logsumexp).astype(np.float32),
        np.array(tail).astype(np.float32),
    )


def validate_full_v2_capture(
    capture: Mapping[str, np.ndarray],
    *,
    top_k: int,
    hidden_size: int,
) -> None:
    """Fail closed on the sparse full-v2 session tensor contract."""

    expected_dtypes = {
        "positions": np.dtype(np.int32),
        "target_token_ids": np.dtype(np.int32),
        "topk_logit_ids": np.dtype(np.int32),
        "topk_logit_values": np.dtype(np.float16),
        "logsumexp": np.dtype(np.float32),
        "tail_mass": np.dtype(np.float16),
        "layer_77_hidden_probe": np.dtype(np.float16),
        "router_top8_expert_ids": np.dtype(np.int32),
        "router_top8_normalized_weights": np.dtype(np.float16),
    }
    if set(capture) != set(expected_dtypes):
        raise ValueError("full-v2 capture tensor inventory mismatch")
    positions = capture["positions"]
    if positions.ndim != 1:
        raise ValueError("positions must be rank-1")
    p = positions.size
    expected_shapes = {
        "positions": (p,),
        "target_token_ids": (p,),
        "topk_logit_ids": (p, top_k),
        "topk_logit_values": (p, top_k),
        "logsumexp": (p,),
        "tail_mass": (p,),
        "layer_77_hidden_probe": (p, hidden_size),
        "router_top8_expert_ids": (len(_ROUTER_LAYERS), p, 8),
        "router_top8_normalized_weights": (len(_ROUTER_LAYERS), p, 8),
    }
    for name, dtype in expected_dtypes.items():
        value = capture[name]
        if value.dtype != dtype or value.shape != expected_shapes[name]:
            raise ValueError(
                f"{name} must have dtype {dtype} and shape {expected_shapes[name]}"
            )
    if p == 0 or np.any(np.diff(positions) <= 0):
        raise ValueError("positions must be non-empty and strictly increasing")
    if not np.all(np.isfinite(capture["logsumexp"])):
        raise ValueError("logsumexp must be finite")
    tail = capture["tail_mass"].astype(np.float32)
    if np.any(tail < 0) or np.any(tail >= 1) or not np.all(np.isfinite(tail)):
        raise ValueError("tail_mass must be finite and in [0, 1)")
    weights = capture["router_top8_normalized_weights"].astype(np.float32)
    if np.any(weights < 0) or not np.allclose(weights.sum(axis=-1), 1.0, atol=1e-3):
        raise ValueError("router_top8_normalized_weights must sum to one")


# ---------------------------------------------------------------------------
# Source workload (contract-bypass loader)
# ---------------------------------------------------------------------------


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_teich_source_workload(
    *,
    snapshot_dir: str | Path,
    non_vq_package_dir: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """Load the FP32-source workload without the frozen-pack contract.

    Reuses the hardened building blocks from the eval producer (strict non-VQ
    bind, authenticated source-blob inventory, NVFP4 expert resolver) and
    returns ``(workload, identities)`` where ``identities`` records the
    computed config/index/manifest SHA-256s for the run manifest.
    """

    producer_api = importlib.import_module(
        "mlx_vq.quality.glm52_teacher_cache_producer"
    )
    source_api = importlib.import_module("mlx_vq.models.glm52_source_teacher")
    mx = importlib.import_module("mlx.core")
    from mlx.utils import tree_map_with_path

    from mlx_vq.convert.stream_convert import load_safetensors_index
    from mlx_vq.models.glm52_vq_adapter import (
        GLM52VQModel,
        bind_glm52_non_vq_weights,
        glm52_vq_args_from_config,
    )

    snapshot = Path(snapshot_dir)
    non_vq_root = Path(non_vq_package_dir)
    config_bytes = (snapshot / "config.json").read_bytes()
    index_bytes = (snapshot / "model.safetensors.index.json").read_bytes()
    manifest_path = non_vq_root / "non-vq-manifest.json"
    identities: dict[str, Any] = {
        "config_sha256": _sha256_bytes(config_bytes),
        "index_sha256": _sha256_bytes(index_bytes),
        "non_vq_manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
        "snapshot_dir": str(snapshot),
        "non_vq_package_dir": str(non_vq_root),
    }
    non_vq_manifest = json.loads(manifest_path.read_text())
    identities["model_id"] = non_vq_manifest.get("model_id")
    identities["source_revision"] = non_vq_manifest.get("source_revision")

    config = json.loads(config_bytes)
    model = GLM52VQModel(glm52_vq_args_from_config(config))

    def cast_parameter(path: str, value: Any) -> Any:
        if model.cast_predicate(path) and mx.issubdtype(value.dtype, mx.floating):
            return value.astype(mx.bfloat16)
        return value

    model.update(tree_map_with_path(cast_parameter, model.parameters()))
    non_vq_index = load_safetensors_index(non_vq_root / "model.safetensors.index.json")
    bind_report = bind_glm52_non_vq_weights(model, non_vq_root, non_vq_index, strict=True)
    if bind_report.loaded_count != 1_272:
        raise ValueError("strict non-VQ bind did not load exactly 1272 runtime targets")
    model.eval()
    mx.eval(model.parameters())

    source_blobs = producer_api._open_authenticated_source_blob_inventory(
        snapshot, index_bytes
    )
    layer_resolver = producer_api._make_authenticated_layer_resolver(
        source_blobs,
        resolver_factory=source_api.make_modelopt_nvfp4_expert_weight_resolver,
    )
    workload = producer_api.GLM52SourceTeacherWorkload(
        model=model,
        expert_resolver_for_layer=layer_resolver,
        expected_num_layers=78,
        source_blob_inventory=source_blobs,
    )
    return workload, identities


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


class _Heartbeat:
    """Atomic JSON heartbeat + append-only progress log."""

    def __init__(self, path: Path, *, total_sessions: int, total_supervised: int):
        self.path = path
        self.log_path = path.with_suffix(".log")
        self.started = time.time()
        self.total_sessions = total_sessions
        self.total_supervised = total_supervised
        self.sessions_done = 0
        self.supervised_done = 0
        self.forward_tokens_done = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, **fields: Any) -> None:
        mx = importlib.import_module("mlx.core")
        elapsed = max(time.time() - self.started, 1e-9)
        tok_s = self.forward_tokens_done / elapsed
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(elapsed, 1),
            "sessions_done": self.sessions_done,
            "total_sessions": self.total_sessions,
            "supervised_done": self.supervised_done,
            "total_supervised": self.total_supervised,
            "forward_tokens_done": self.forward_tokens_done,
            "forward_tokens_per_s": round(tok_s, 1),
            "peak_gb": round(float(mx.get_peak_memory()) / 1024**3, 2),
            **fields,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, self.path)
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")


def _valid_mask_sha256(valid_mask: np.ndarray) -> str:
    if valid_mask.dtype != np.bool_ or valid_mask.ndim != 2:
        raise ValueError("valid mask must be rank-2 bool")
    return _sha256_bytes(
        json.dumps(list(valid_mask.shape), separators=(",", ":")).encode()
        + b"\0"
        + np.ascontiguousarray(valid_mask.astype(np.uint8)).tobytes()
    )


def _bf16_words(value: Any, mx: Any) -> np.ndarray:
    contiguous = mx.contiguous(value.astype(mx.bfloat16))
    mx.eval(contiguous)
    return np.ascontiguousarray(np.array(contiguous.view(mx.uint16)), dtype=np.uint16)


def _bf16_from_words(value: np.ndarray, mx: Any) -> Any:
    return mx.contiguous(mx.array(value, dtype=mx.uint16).view(mx.bfloat16))


def _signal_path(
    checkpoint_root: Path,
    *,
    prompt_id: str,
    kind: str,
    layer: int,
) -> Path:
    safe = prompt_id.replace("/", "__")
    return checkpoint_root / "resident-signals" / safe / f"{kind}-layer-{layer:05d}.safetensors"


def _write_resident_signal(
    path: Path,
    *,
    tensors: Mapping[str, np.ndarray],
    identity_sha256: str,
    prompt_id: str,
    token_ids_sha256: str,
    layer: int,
) -> None:
    normalized = {name: np.ascontiguousarray(value) for name, value in tensors.items()}
    raw = save_safetensors(
        normalized,
        metadata={
            "record_type": "glm52_teich_resident_signal_v1",
            "identity_sha256": identity_sha256,
            "prompt_id": prompt_id,
            "token_ids_sha256": token_ids_sha256,
            "layer": str(layer),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == raw:
            return
        raise ValueError(f"resident signal {path} conflicts with immutable authority")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
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


def _read_resident_signal(
    path: Path,
    *,
    expected_names: set[str],
    identity_sha256: str,
    prompt_id: str,
    token_ids_sha256: str,
    layer: int,
) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"resident signal is missing: {path}")
    try:
        tensors = load_safetensors(path.read_bytes())
    except Exception as error:
        raise ValueError(f"resident signal is malformed: {path}: {error}") from error
    if set(tensors) != expected_names:
        raise ValueError(f"resident signal tensor inventory mismatch: {path}")
    with safe_open(path, framework="np") as handle:
        metadata = handle.metadata()
    expected_metadata = {
        "record_type": "glm52_teich_resident_signal_v1",
        "identity_sha256": identity_sha256,
        "prompt_id": prompt_id,
        "token_ids_sha256": token_ids_sha256,
        "layer": str(layer),
    }
    if metadata != expected_metadata:
        raise ValueError(f"resident signal identity mismatch: {path}")
    return tensors


# ---------------------------------------------------------------------------
# Production forward
# ---------------------------------------------------------------------------


def _run_chunk_forward(
    workload: Any,
    chunk_sessions: Sequence[TeichSession],
    *,
    top_k: int,
    lm_head_slice: int,
    query_chunk_size: int = 512,
    on_layer: Callable[[int], None] | None = None,
    checkpoint_config: TeacherCheckpointConfig | None = None,
    checkpoint_key: str | None = None,
    requested_prompt_ids: frozenset[str] | None = None,
) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    """Run one padded chunk through the teacher; yield per-session captures.

    Mirrors the eval producer's layer loop (same MoE streaming, same scatter,
    same precision assertions) with long-context IndexShare threading and a
    sliced, supervised-positions-only LM head.

    Attention dispatches through ``dense_or_chunked_attention``: short
    sequences take the byte-identical dense reference path; sequences long
    enough to engage DSA use the memory-bounded, query-chunked gather path
    (see ``glm52_long_context_attention`` -- the dense indexer/attention
    score matmuls are O(sequence_length^2) with no bound, which crashes past
    ~37K tokens; verified numerically equivalent to the dense reference).
    """

    mx = importlib.import_module("mlx.core")
    source_api = importlib.import_module("mlx_vq.models.glm52_source_teacher")
    from mlx_lm.models.base import create_causal_mask
    from mlx_vq.models.glm52_long_context_attention import dense_or_chunked_attention
    from mlx_vq.models.glm52_vq_adapter import GLM52VQModel, Glm52VQMoE

    model = workload.model
    if not isinstance(model, GLM52VQModel):
        raise TypeError("workload model must be a GLM52VQModel")
    expected_num_layers = workload.expected_num_layers
    if len(model.model.layers) != expected_num_layers:
        raise ValueError("model layer inventory does not match expected_num_layers")

    encoded = tuple(tuple(session.prompt.encoded_token_ids) for session in chunk_sessions)
    token_batch, valid_mask, right_padding = source_api._prepare_predictor_batch(
        encoded,
        vocab_size=model.args.vocab_size,
        pad_token_id=workload.pad_token_id,
    )
    batch_size, sequence_length = token_batch.shape
    source_api._audit_source_teacher_model_precision(model)

    tokens = mx.array(token_batch)
    hidden = mx.contiguous(model.model.embed_tokens(tokens))
    if hidden.dtype != mx.bfloat16:
        raise ValueError("embedded source-teacher hidden state must use bfloat16")
    mx.eval(hidden)
    causal_mask = create_causal_mask(
        sequence_length,
        right_padding=mx.array(right_padding),
    )
    mx.eval(causal_mask)
    valid_flat_indices = np.flatnonzero(valid_mask.reshape(-1)).astype(np.int64)
    ordered_prompt_ids = tuple(session.prompt_id for session in chunk_sessions)
    ordered_token_hashes = tuple(
        session.prompt.token_ids_sha256 for session in chunk_sessions
    )
    valid_mask_sha256 = _valid_mask_sha256(valid_mask)
    generation_parameters: dict[str, object] = {
        "top_k": top_k,
        "lm_head_slice": lm_head_slice,
        "query_chunk_size": query_chunk_size,
        "router_layers": list(_ROUTER_LAYERS),
        "cka_probe_layer": 77,
    }
    chunk_root: Path | None = None
    forward_store: LocalTeacherCheckpointStore | None = None
    normalized_store: LocalTeacherCheckpointStore | None = None
    deadline_policy: DeadlinePolicy | None = None
    if checkpoint_config is not None:
        if checkpoint_key is None:
            checkpoint_key = _sha256_bytes(
                json.dumps(ordered_prompt_ids, separators=(",", ":")).encode()
            )[:20]
        chunk_root = (
            checkpoint_config.local_checkpoint_dir / "chunks" / checkpoint_key
        )
        forward_store = LocalTeacherCheckpointStore(
            chunk_root / "forward", identity=checkpoint_config.identity
        )
        normalized_store = LocalTeacherCheckpointStore(
            chunk_root / "normalized", identity=checkpoint_config.identity
        )
        if checkpoint_config.effective_deadline is not None:
            deadline_policy = DeadlinePolicy(
                capacity_block_end=checkpoint_config.capacity_block_deadline,
                execution_deadline=checkpoint_config.execution_deadline,
                stop_file=checkpoint_config.stop_file,
            )

    router_capture: list[dict[int, tuple[np.ndarray, np.ndarray]]] = [
        {} for _ in chunk_sessions
    ]
    cka_capture: list[np.ndarray | None] = [None for _ in chunk_sessions]
    valid_offsets = np.cumsum(
        [0, *(session.token_count for session in chunk_sessions[:-1])]
    ).astype(np.int64)

    def load_resident_signals(*, before_layer: int) -> None:
        if chunk_root is None or checkpoint_config is None:
            return
        for layer_number in _ROUTER_LAYERS:
            if layer_number >= before_layer:
                continue
            for row_index, session in enumerate(chunk_sessions):
                path = _signal_path(
                    chunk_root,
                    prompt_id=session.prompt_id,
                    kind="router",
                    layer=layer_number,
                )
                tensors = _read_resident_signal(
                    path,
                    expected_names={"expert_ids", "normalized_weights"},
                    identity_sha256=checkpoint_config.identity.sha256,
                    prompt_id=session.prompt_id,
                    token_ids_sha256=session.prompt.token_ids_sha256,
                    layer=layer_number,
                )
                router_capture[row_index][layer_number] = (
                    tensors["expert_ids"],
                    tensors["normalized_weights"],
                )
        if before_layer > 77:
            for row_index, session in enumerate(chunk_sessions):
                path = _signal_path(
                    chunk_root,
                    prompt_id=session.prompt_id,
                    kind="cka",
                    layer=77,
                )
                tensors = _read_resident_signal(
                    path,
                    expected_names={"hidden_probe"},
                    identity_sha256=checkpoint_config.identity.sha256,
                    prompt_id=session.prompt_id,
                    token_ids_sha256=session.prompt.token_ids_sha256,
                    layer=77,
                )
                cka_capture[row_index] = tensors["hidden_probe"]

    def validate_restored(restored: Any, *, state_kind: str) -> None:
        expected_parameters = {**generation_parameters, "state_kind": state_kind}
        if (
            restored.ordered_prompt_ids != ordered_prompt_ids
            or restored.ordered_token_hashes != ordered_token_hashes
            or restored.batch_shape != (batch_size, sequence_length)
            or restored.valid_mask_sha256 != valid_mask_sha256
            or restored.generation_parameters != expected_parameters
            or restored.hidden_bf16_bits.shape
            != (batch_size, sequence_length, model.args.hidden_size)
        ):
            raise ValueError("teacher checkpoint chunk identity or tensor contract drift")

    start_layer = 0
    prev_topk_indices = None
    normalized = None
    if (
        checkpoint_config is not None
        and checkpoint_config.resume
        and normalized_store is not None
        and normalized_store.latest_path.exists()
    ):
        restored = normalized_store.load_latest()
        validate_restored(restored, state_kind="normalized")
        if restored.next_layer != expected_num_layers:
            raise ValueError("normalized checkpoint must follow all decoder layers")
        normalized = _bf16_from_words(restored.hidden_bf16_bits, mx)
        mx.eval(normalized)
        start_layer = expected_num_layers
        load_resident_signals(before_layer=expected_num_layers)
    elif (
        checkpoint_config is not None
        and checkpoint_config.resume
        and forward_store is not None
        and forward_store.latest_path.exists()
    ):
        restored = forward_store.load_latest()
        validate_restored(restored, state_kind="forward")
        if restored.next_layer > expected_num_layers:
            raise ValueError("forward checkpoint is outside the decoder inventory")
        hidden = _bf16_from_words(restored.hidden_bf16_bits, mx)
        prev_topk_indices = (
            None
            if restored.prev_topk_indices is None
            else mx.array(restored.prev_topk_indices, dtype=mx.int32)
        )
        mx.eval(hidden)
        if isinstance(prev_topk_indices, mx.array):
            mx.eval(prev_topk_indices)
        start_layer = restored.next_layer
        load_resident_signals(before_layer=start_layer)

    for layer_number in range(start_layer, expected_num_layers):
        layer = model.model.layers[layer_number]
        if int(layer.layer_idx) != layer_number:
            raise ValueError("decoder layer identity does not match its index")
        normed_input = layer.input_layernorm(hidden)
        attention_output, next_topk_indices = dense_or_chunked_attention(
            layer.self_attn,
            normed_input,
            causal_mask,
            prev_topk_indices,
            query_chunk_size=query_chunk_size,
        )
        # Long-context DSA: index layers emit fresh top-k indices; sharing
        # layers return None and consume the previous state unchanged.
        if next_topk_indices is not None:
            prev_topk_indices = next_topk_indices
        attention_hidden = hidden + attention_output
        mlp_hidden = layer.post_attention_layernorm(attention_hidden)
        if isinstance(layer.mlp, Glm52VQMoE):
            flat_mlp_hidden = mlp_hidden.reshape(-1, model.args.hidden_size)
            valid_hidden = mx.take(
                flat_mlp_hidden, mx.array(valid_flat_indices), axis=0
            )
            streamed = source_api.stream_selected_expert_moe(
                valid_hidden,
                gate=layer.mlp.gate,
                expert_weight_resolver=workload.expert_resolver_for_layer(layer_number),
                shared_expert=layer.mlp.get("shared_experts"),
                num_routed_experts=model.args.n_routed_experts,
            )
            expected_assignments = (
                len(valid_flat_indices) * model.args.num_experts_per_tok
            )
            if streamed.valid_assignment_count != expected_assignments:
                raise AssertionError(
                    "valid sparse assignments do not equal n_valid * top_k"
                )
            if streamed.padded_assignment_count != 0:
                raise AssertionError("padded rows entered sparse router accounting")
            if layer_number in _ROUTER_LAYERS:
                if streamed.expert_indices.shape[1] != 8:
                    raise ValueError("full-v2 router targets require exactly top-8 experts")
                for row_index, session in enumerate(chunk_sessions):
                    selected_ordinals = valid_offsets[row_index] + session.positions
                    selected_ids = mx.take(
                        streamed.expert_indices,
                        mx.array(selected_ordinals),
                        axis=0,
                    )
                    selected_weights = mx.take(
                        streamed.route_scores,
                        mx.array(selected_ordinals),
                        axis=0,
                    )
                    mx.eval(selected_ids, selected_weights)
                    ids_np = np.ascontiguousarray(
                        np.array(selected_ids), dtype=np.int32
                    )
                    weights_np = np.ascontiguousarray(
                        np.array(selected_weights), dtype=np.float16
                    )
                    router_capture[row_index][layer_number] = (
                        ids_np,
                        weights_np,
                    )
                    if chunk_root is not None and checkpoint_config is not None:
                        _write_resident_signal(
                            _signal_path(
                                chunk_root,
                                prompt_id=session.prompt_id,
                                kind="router",
                                layer=layer_number,
                            ),
                            tensors={
                                "expert_ids": ids_np,
                                "normalized_weights": weights_np,
                            },
                            identity_sha256=checkpoint_config.identity.sha256,
                            prompt_id=session.prompt_id,
                            token_ids_sha256=session.prompt.token_ids_sha256,
                            layer=layer_number,
                        )
            sparse_output = source_api._scatter_valid_rows(
                streamed.output,
                valid_flat_indices=valid_flat_indices,
                batch_size=batch_size,
                sequence_length=sequence_length,
                hidden_size=model.args.hidden_size,
            )
            hidden = attention_hidden + sparse_output
            del flat_mlp_hidden, valid_hidden, streamed, sparse_output
        else:
            mlp_out = layer.mlp(mlp_hidden)
            hidden = attention_hidden + mlp_out
        hidden = mx.contiguous(hidden)
        if hidden.dtype != mx.bfloat16:
            raise ValueError("post-layer source-teacher hidden state must use bfloat16")
        mx.eval(hidden)
        if layer_number == 77:
            for row_index, session in enumerate(chunk_sessions):
                selected_hidden = mx.take(
                    hidden[row_index], mx.array(session.positions), axis=0
                )
                mx.eval(selected_hidden)
                probe = np.ascontiguousarray(
                    np.array(selected_hidden), dtype=np.float16
                )
                cka_capture[row_index] = probe
                if chunk_root is not None and checkpoint_config is not None:
                    _write_resident_signal(
                        _signal_path(
                            chunk_root,
                            prompt_id=session.prompt_id,
                            kind="cka",
                            layer=77,
                        ),
                        tensors={"hidden_probe": probe},
                        identity_sha256=checkpoint_config.identity.sha256,
                        prompt_id=session.prompt_id,
                        token_ids_sha256=session.prompt.token_ids_sha256,
                        layer=77,
                    )
        del attention_output, attention_hidden, mlp_hidden
        _clear_post_layer_cache(mx.clear_cache)
        stop_requested = bool(
            checkpoint_config is not None
            and (
                (
                    checkpoint_config.stop_file is not None
                    and checkpoint_config.stop_file.exists()
                )
                or (
                    deadline_policy is not None
                    and deadline_policy.stop_assigning()
                )
            )
        )
        next_layer = layer_number + 1
        if forward_store is not None and should_checkpoint_after_layer(
            next_layer,
            every_layers=checkpoint_config.checkpoint_every_layers,
            force=stop_requested or next_layer == expected_num_layers,
        ):
            dsa_np = (
                None
                if prev_topk_indices is None
                else np.ascontiguousarray(np.array(prev_topk_indices), dtype=np.int32)
            )
            forward_store.publish_forward(
                next_layer=next_layer,
                hidden_bf16_bits=_bf16_words(hidden, mx),
                prev_topk_indices=dsa_np,
                ordered_prompt_ids=ordered_prompt_ids,
                ordered_token_hashes=ordered_token_hashes,
                batch_shape=(batch_size, sequence_length),
                valid_mask_sha256=valid_mask_sha256,
                generation_parameters={**generation_parameters, "state_kind": "forward"},
            )
        if on_layer is not None:
            on_layer(layer_number)
        if stop_requested:
            raise TeacherStopRequested(
                f"teacher stopped after durable layer {next_layer} boundary"
            )

    if normalized is None:
        normalized = model.model.norm(hidden)
        if normalized.dtype != mx.bfloat16:
            raise ValueError("final normalized source-teacher hidden state must use bfloat16")
        mx.eval(normalized)
        if normalized_store is not None:
            normalized_store.publish_forward(
                next_layer=expected_num_layers,
                hidden_bf16_bits=_bf16_words(normalized, mx),
                prev_topk_indices=None,
                ordered_prompt_ids=ordered_prompt_ids,
                ordered_token_hashes=ordered_token_hashes,
                batch_shape=(batch_size, sequence_length),
                valid_mask_sha256=valid_mask_sha256,
                generation_parameters={
                    **generation_parameters,
                    "state_kind": "normalized",
                },
            )
        del hidden
        mx.clear_cache()

    for row_index, session in enumerate(chunk_sessions):
        if requested_prompt_ids is not None and session.prompt_id not in requested_prompt_ids:
            continue
        positions = session.positions
        slice_store = None
        if checkpoint_config is not None:
            slice_store = LocalLMHeadSliceStore(
                checkpoint_config.local_checkpoint_dir
                / "lm-head"
                / session.prompt_id.replace("/", "__"),
                identity=checkpoint_config.identity,
                prompt_id=session.prompt_id,
                token_ids_sha256=session.prompt.token_ids_sha256,
                positions=positions.astype(np.int32),
                target_token_ids=session.target_token_ids.astype(np.int32),
                top_k=top_k,
            )
            ranges = slice_store.missing_ranges(slice_size=lm_head_slice)
        else:
            ranges = tuple(
                (start, min(start + lm_head_slice, positions.size))
                for start in range(0, positions.size, lm_head_slice)
            )
        ids_parts: list[np.ndarray] = []
        values_parts: list[np.ndarray] = []
        logsumexp_parts: list[np.ndarray] = []
        tail_parts: list[np.ndarray] = []
        for start, stop in ranges:
            slice_positions = mx.array(positions[start:stop])
            slice_hidden = mx.take(normalized[row_index], slice_positions, axis=0)
            slice_logits = model.lm_head(slice_hidden[None, :, :]).reshape(
                stop - start, model.args.vocab_size
            )
            ids, values, logsumexp, tail = _topk_capture_mlx(slice_logits, k=top_k)
            slice_tensors = {
                "topk_logit_ids": ids.astype(np.int32, copy=False),
                "topk_logit_values": values.astype(np.float16),
                "logsumexp": logsumexp.astype(np.float32, copy=False),
                "tail_mass": tail.astype(np.float16),
            }
            if slice_store is not None:
                slice_store.publish_slice(start=start, end=stop, tensors=slice_tensors)
            else:
                ids_parts.append(slice_tensors["topk_logit_ids"])
                values_parts.append(slice_tensors["topk_logit_values"])
                logsumexp_parts.append(slice_tensors["logsumexp"])
                tail_parts.append(slice_tensors["tail_mass"])
            del slice_positions, slice_hidden, slice_logits
            mx.clear_cache()
            lm_stop_requested = bool(
                checkpoint_config is not None
                and (
                    (
                        checkpoint_config.stop_file is not None
                        and checkpoint_config.stop_file.exists()
                    )
                    or (
                        deadline_policy is not None
                        and deadline_policy.stop_assigning()
                    )
                )
            )
            if lm_stop_requested:
                raise TeacherStopRequested(
                    f"teacher stopped after durable LM-head slice {start}:{stop}"
                )
        if slice_store is not None:
            capture = slice_store.assemble()
        else:
            capture = {
                "positions": positions.astype(np.int32),
                "target_token_ids": session.target_token_ids.astype(np.int32),
                "topk_logit_ids": np.concatenate(ids_parts, axis=0),
                "topk_logit_values": np.concatenate(values_parts, axis=0),
                "logsumexp": np.concatenate(logsumexp_parts, axis=0),
                "tail_mass": np.concatenate(tail_parts, axis=0),
            }
        if cka_capture[row_index] is None or set(router_capture[row_index]) != set(
            _ROUTER_LAYERS
        ):
            raise ValueError("full-v2 resident CKA/router signal inventory is incomplete")
        capture["layer_77_hidden_probe"] = cka_capture[row_index]
        capture["router_top8_expert_ids"] = np.stack(
            [router_capture[row_index][layer][0] for layer in _ROUTER_LAYERS]
        )
        capture["router_top8_normalized_weights"] = np.stack(
            [router_capture[row_index][layer][1] for layer in _ROUTER_LAYERS]
        )
        validate_full_v2_capture(
            capture,
            top_k=top_k,
            hidden_size=model.args.hidden_size,
        )
        yield session.prompt_id, capture
        gc.collect()
    del normalized
    mx.clear_cache()


def _session_output_path(out_dir: Path, prompt_id: str) -> Path:
    safe = prompt_id.replace("/", "__")
    return out_dir / f"{safe}.npz"


def _atomic_savez(target: Path, **arrays) -> None:
    # np.savez_compressed appends ".npz" to any path that doesn't already
    # end with it, so the temp path must end in ".npz" itself (e.g.
    # "foo.tmp.npz", not "foo.npz.tmp") or the write lands at a different
    # path than the one os.replace expects, and the rename raises
    # FileNotFoundError.
    tmp = target.with_name(target.stem + ".tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        tmp.unlink(missing_ok=True)


def _audit_existing_session_capture(
    path: Path,
    *,
    session: TeichSession,
    top_k: int,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"existing session capture must be a regular file: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            capture = {name: archive[name] for name in archive.files}
    except Exception as error:
        raise ValueError(f"existing session capture is malformed: {path}: {error}") from error
    probe = capture.get("layer_77_hidden_probe")
    if not isinstance(probe, np.ndarray) or probe.ndim != 2:
        raise ValueError(f"existing session capture lacks the layer-77 probe: {path}")
    validate_full_v2_capture(capture, top_k=top_k, hidden_size=probe.shape[1])
    if not np.array_equal(capture["positions"], session.positions.astype(np.int32)):
        raise ValueError(f"existing session capture positions drifted: {path}")
    if not np.array_equal(
        capture["target_token_ids"], session.target_token_ids.astype(np.int32)
    ):
        raise ValueError(f"existing session capture targets drifted: {path}")


def run_teich_teacher_production(
    *,
    snapshot_dir: str | Path,
    non_vq_package_dir: str | Path,
    pack_path: str | Path,
    out_dir: str | Path,
    heartbeat_path: str | Path,
    top_k: int = _DEFAULT_TOP_K,
    max_batch_tokens: int = 131_072,
    lm_head_slice: int = _DEFAULT_LM_HEAD_SLICE,
    query_chunk_size: int = 512,
    session_filter: Callable[[TeichSession], bool] | None = None,
    workload_loader: Callable[..., tuple[Any, dict[str, Any]]] | None = None,
    checkpoint_config: TeacherCheckpointConfig | None = None,
) -> dict[str, Any]:
    """Produce full-v2 teacher captures for every unfinished Teich session.

    Existing outputs are skipped only after a fresh tensor/identity audit.
    With ``checkpoint_config`` the layer loop, normalized state, resident
    router/CKA signals, and every LM-head slice are independently resumable.
    """

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    sessions = load_teich_prompt_rows(pack_path)
    if session_filter is not None:
        sessions = tuple(s for s in sessions if session_filter(s))
    pending_ids: set[str] = set()
    for session in sessions:
        target = _session_output_path(out_root, session.prompt_id)
        if target.exists() or target.is_symlink():
            _audit_existing_session_capture(target, session=session, top_k=top_k)
        else:
            pending_ids.add(session.prompt_id)
    pending = tuple(s for s in sessions if s.prompt_id in pending_ids)
    heartbeat = _Heartbeat(
        Path(heartbeat_path),
        total_sessions=len(sessions),
        total_supervised=int(sum(s.positions.size for s in sessions)),
    )
    heartbeat.sessions_done = len(sessions) - len(pending)
    heartbeat.supervised_done = int(
        sum(s.positions.size for s in sessions) - sum(s.positions.size for s in pending)
    )
    heartbeat.update(phase="loading-source", pending_sessions=len(pending))
    if not pending:
        heartbeat.update(phase="complete")
        return {"sessions_produced": 0, "skipped": len(sessions)}

    loader = workload_loader or load_teich_source_workload
    workload, identities = loader(
        snapshot_dir=snapshot_dir, non_vq_package_dir=non_vq_package_dir
    )
    pack_sha256 = _sha256_bytes(Path(pack_path).read_bytes())
    generation_identity = {
        "record_type": "glm52_teich_full_v2_generation_config_v1",
        "top_k": top_k,
        "lm_head_slice": lm_head_slice,
        "query_chunk_size": query_chunk_size,
        "max_batch_tokens": max_batch_tokens,
        "router_layers": list(_ROUTER_LAYERS),
        "cka_probe_layers": [77],
    }
    generation_config_sha256 = _sha256_bytes(
        json.dumps(
            generation_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    model_sha256 = _sha256_bytes(
        json.dumps(
            {
                "config_sha256": identities.get("config_sha256"),
                "index_sha256": identities.get("index_sha256"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    if checkpoint_config is not None:
        expected = checkpoint_config.identity
        if expected.prompt_pack_sha256 != pack_sha256:
            raise ValueError("teacher checkpoint prompt-pack identity drift")
        if expected.generation_config_sha256 != generation_config_sha256:
            raise ValueError("teacher checkpoint generation-config identity drift")
        if expected.model_sha256 != model_sha256:
            raise ValueError("teacher checkpoint model identity drift")
        if expected.non_vq_package_sha256 != identities.get(
            "non_vq_manifest_sha256"
        ):
            raise ValueError("teacher checkpoint non-VQ package identity drift")
    manifest = {
        "record_type": "glm52_teich_full_v2_teacher_capture_run",
        "schema_version": 2,
        "top_k": top_k,
        "lm_head_slice": lm_head_slice,
        "query_chunk_size": query_chunk_size,
        "max_batch_tokens": max_batch_tokens,
        "pack_path": str(pack_path),
        "pack_sha256": pack_sha256,
        "generation_config": generation_identity,
        "generation_config_sha256": generation_config_sha256,
        "identities": identities,
        "checkpoint_identity": (
            checkpoint_config.identity.as_dict() if checkpoint_config is not None else None
        ),
        "sessions": {
            s.prompt_id: {
                "token_count": s.token_count,
                "supervised": int(s.positions.size),
                "token_ids_sha256": s.prompt.token_ids_sha256,
            }
            for s in sessions
        },
    }
    manifest_path = out_root / "run-manifest.json"
    manifest_tmp = out_root / ".run-manifest.tmp"
    manifest_tmp.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(manifest_tmp, manifest_path)

    # Chunk the complete filtered authority, not only pending rows.  This keeps
    # chunk identity stable when one session finished before a restart.
    all_chunks = pack_sessions_into_chunks(sessions, max_batch_tokens=max_batch_tokens)
    chunks = tuple(
        chunk
        for chunk in all_chunks
        if any(sessions[index].prompt_id in pending_ids for index in chunk)
    )
    produced = 0
    for chunk_index, chunk in enumerate(chunks):
        chunk_sessions = [sessions[i] for i in chunk]
        chunk_tokens = sum(s.token_count for s in chunk_sessions)
        chunk_requested = frozenset(
            session.prompt_id
            for session in chunk_sessions
            if session.prompt_id in pending_ids
        )
        chunk_key = (
            f"chunk-{all_chunks.index(chunk):05d}-"
            + _sha256_bytes(
                json.dumps(
                    [session.prompt_id for session in chunk_sessions],
                    separators=(",", ":"),
                ).encode()
            )[:16]
        )

        def on_layer(layer_number: int, _chunk=chunk_index, _n=len(chunks)) -> None:
            heartbeat.update(
                phase="layer-forward",
                chunk=f"{_chunk + 1}/{_n}",
                layer=f"{layer_number + 1}/78",
            )

        for prompt_id, capture in _run_chunk_forward(
            workload,
            chunk_sessions,
            top_k=top_k,
            lm_head_slice=lm_head_slice,
            query_chunk_size=query_chunk_size,
            on_layer=on_layer,
            checkpoint_config=checkpoint_config,
            checkpoint_key=chunk_key,
            requested_prompt_ids=chunk_requested,
        ):
            target = _session_output_path(out_root, prompt_id)
            _atomic_savez(target, **capture)
            produced += 1
            heartbeat.sessions_done += 1
            heartbeat.supervised_done += int(capture["positions"].size)
            heartbeat.update(phase="session-complete", last_session=prompt_id)
            pending_ids.remove(prompt_id)
        heartbeat.forward_tokens_done += chunk_tokens
        heartbeat.update(phase="chunk-complete", chunk=f"{chunk_index + 1}/{len(chunks)}")
    heartbeat.update(phase="complete")
    return {"sessions_produced": produced, "skipped": len(sessions) - len(pending)}
