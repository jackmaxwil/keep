"""Layer-sequential streaming teacher runner for DeepSeek-V4-Flash-0731.

The released checkpoint is 163 GB and its routed experts alone are 147 GB, so a
128 GB machine cannot hold the model. Wave 2 increment 2
(``.superpowers/sdd/wave2-increment2-report.md``) measured the way through and
this module builds it:

* **Residents stay resident.** All 43 layers' non-routed tensors (~6.7 GB of FP8
  codes decoded bit-exactly to bf16, plus the 2.1 GB embedding and LM head) are
  bound once at startup and never re-read.
* **Routed experts stream, one layer at a time.** Every one of the 43 layers
  keeps its entire routed-expert block inside exactly one shard as a 3.42 GB run
  of tensor bytes inside a 3.55 GB span (measured, §3.7), so a layer's experts
  are *one seek and one sequential read*, not a cache with a miss rate. Read at
  9.9 GB/s that is 0.36 s per layer against ~10 s of compute, and layer N+1's
  read is issued behind layer N's compute.
* **Experts are never dequantised.** The shipped ``I8`` FP4 code bytes are
  reinterpreted as ``uint32`` and the ``F8_E8M0`` scale bytes handed straight to
  ``mx.gather_qmm(mode="mxfp4", group_size=32, bits=4)`` -- 3.42 GB/layer instead
  of 12.9 GB/layer, and faster, because the routed matmuls are bandwidth-bound.

**The forward is layer-major, not chunk-major.** A chunk-major loop (chunk
outer, layer inner) would re-read all 43 layers' experts for every 1,024-token
chunk: 11,700 chunk-sweeps x 18 s = 58.8 h of pure I/O, and the run would be
I/O-bound. Layer-major (layer outer, chunk inner) reads each layer once per
session: 257 sweeps, 1.3 h, fully hidden behind compute. The two are
**bit-identical** -- layer i chunk j sees the same input and the same cache
state either way -- and :func:`tests/test_dsv4_teacher_runner.py` asserts that
against the chunk-major reference rather than assuming it.

Two behaviours pinned by increment 2 are load-bearing here:

1. **Chunked prefill is not bit-identical to single-shot prefill** (the two
   paths reduce over the same values in a different order). A teacher cache is
   therefore only reproducible against a pinned prefill chunk size, so
   ``prefill_chunk_tokens`` is recorded in every session artifact and in the run
   manifest, and a resumed run refuses session files written at a different
   chunk.
2. **``cache=None`` silently drops the pooled causal mask.** Every forward here
   passes a real cache from :meth:`DeepseekV4FlashVQModel.make_cache`.

Modes
-----
``calibration``
    imatrix-style activation statistics: per (layer, projection, expert) sums of
    ``activation^2`` over the rows that routed to that expert, plus the
    router-score-weighted variant. Accumulated **on device** (a one-hot
    ``count^T @ squared`` matmul per chunk) rather than in host NumPy, which at
    43 layers x 46 k tokens x 6 routes would be tens of GB of float64 traffic.
    :func:`finalize_dsv4_calibration_imatrix` merges the per-session files into
    the sidecar schema ``mlx_vq.quality.imatrix`` already consumes.
``logits``
    top-2048 logits at the pack's supervised positions, fp16, npz, following the
    :mod:`mlx_vq.quality.glm52_teich_teacher_producer` full-v2 schema minus the
    GLM-specific probes.
``mtp-targets``
    per supervised position, the DSpark drafter's top-K block logits and the
    drafter's final hidden states -- the ``(input, target)`` pair MTP-head
    recovery training needs. See :func:`mtp_target_capture` for the schema and
    for the two costs it carries (a second, drafter-shaped pass over the
    supervised positions, and a per-session artifact that is
    ``mtp_draft_width`` times a logits-mode one).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import numpy as np

__all__ = [
    "DSV4_PACK_RECORD_TYPE",
    "DSV4_TEACHER_MODES",
    "Dsv4RunHeartbeat",
    "Dsv4Session",
    "Dsv4StreamStats",
    "InMemoryExpertProvider",
    "LayerExpertSpan",
    "MtpDrafterUnavailable",
    "ShardStreamingExpertProvider",
    "build_dsv4_expert_span_index",
    "calibration_session_arrays",
    "finalize_dsv4_calibration_imatrix",
    "layer_major_prefill",
    "load_dsv4_streaming_workload",
    "load_dsv4_teich_pack",
    "mtp_target_capture",
    "routed_slot_columns",
    "run_dsv4_teacher_production",
    "session_output_path",
    "validate_dsv4_calibration_capture",
    "validate_dsv4_logits_capture",
    "validate_dsv4_mtp_targets_capture",
]

DSV4_PACK_RECORD_TYPE = "dsv4_coding_agent_corpus"
DSV4_PACK_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
DSV4_PACK_VOCAB_SIZE = 129_280
DSV4_TEACHER_MODES = ("calibration", "logits", "mtp-targets")

#: Increment 2 §3.4: chunk 1024 beats 2048 by 13-16% at three lengths and two
#: repeats, and wins on peak memory. Changing this changes the last digits of
#: every teacher logit -- see the module docstring.
DEFAULT_PREFILL_CHUNK_TOKENS = 1024
DEFAULT_TOP_K = 2048
DEFAULT_LM_HEAD_SLICE = 2048
DEFAULT_IO_THREADS = 8

#: Checkpoint projection name -> module name. w1=gate, w2=down, w3=up.
CHECKPOINT_PROJECTIONS: dict[str, str] = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}
#: The two distinct imatrix input spaces. ``gate_proj`` and ``up_proj`` consume
#: the *same* rows (the post-``ffn_norm`` hidden state) under the *same* routing,
#: so their imatrix vectors are identical by construction; only ``down_proj``
#: sees its own per-route GLU activation. Storing one copy per space instead of
#: one per projection halves every calibration artifact.
IMATRIX_SPACES = ("hidden", "down")
_SPACE_PROJECTIONS: dict[str, tuple[str, ...]] = {
    "hidden": ("gate_proj", "up_proj"),
    "down": ("down_proj",),
}

#: macOS ``fcntl`` command: bypass the unified buffer cache on this descriptor.
#: A 147 GB-per-session stream would otherwise evict everything MLX wants.
_F_NOCACHE = 48
#: Coalesce source reads separated by less than this into one sequential run.
_COALESCE_GAP_BYTES = 1 << 20


class MtpDrafterUnavailable(RuntimeError):
    """``--mode mtp-targets`` was asked for on a model with no usable drafter.

    Not "not implemented" any more -- the DSpark drafter forward landed in
    Wave 2 increment 3. This is the narrower, still-real failure: the loaded
    workload has ``mtp_drafter is None`` (built ``with_mtp=False``) or its
    stages have no routed experts bound.
    """


#: Draft-block width used when the caller does not pin one. ``None`` means
#: ``config.dspark_block_size`` (5 on this release) -- the widest block the
#: checkpoint can propose, and therefore the most supervision per position.
DEFAULT_MTP_DRAFT_WIDTH = None

#: How often ``mtp_target_capture`` returns MLX's buffer pool. Every draft block
#: allocates the same shapes, so clearing per position throws away exactly the
#: buffers the next one wants; clearing never lets a long session drift.
_MTP_CLEAR_CACHE_EVERY = 512


# ---------------------------------------------------------------------------
# Prompt pack
# ---------------------------------------------------------------------------


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Dsv4Session:
    """One retokenized teich session and its supervised positions."""

    prompt_id: str
    campaign_split: str
    token_ids: np.ndarray  # int32 [T]
    positions: np.ndarray  # int64 [P] -- logit row p predicts token p+1
    target_token_ids: np.ndarray  # int64 [P]
    token_ids_sha256: str

    @property
    def token_count(self) -> int:
        return int(self.token_ids.size)

    @property
    def supervised_count(self) -> int:
        return int(self.positions.size)


def _exact_json_int_list(value: Any, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not int for item in value)
    ):
        raise ValueError(f"{label} must be a non-empty list of JSON integers")
    return value


def load_dsv4_teich_pack(
    pack_path: str | Path,
    *,
    splits: Sequence[str] | None = None,
    prompt_ids: Sequence[str] | None = None,
    max_session_tokens: int | None = None,
    teacher_window: int | None = None,
) -> tuple[Dsv4Session, ...]:
    """Load and fully validate the retokenized DeepSeek-V4 teich pack.

    Every row is checked for the supervision convention the bridge promises
    (``target_token_ids[i] == encoded_token_ids[positions[i] + 1]``) and for its
    declared ``token_ids_sha256``, so a mis-built or partially-rewritten pack
    fails here rather than producing silently misaligned teacher targets.

    ``teacher_window`` (default
    :data:`ramp.models.deepseek_v4_flash_adapter.DEEPSEEK_V4_FLASH_TEACHER_WINDOW_TOKENS`)
    is an assertion, not a truncation: a session longer than the window would
    lose supervised positions off the end, so it raises.
    """

    from ramp.models.deepseek_v4_flash_adapter import (
        DEEPSEEK_V4_FLASH_TEACHER_WINDOW_TOKENS,
    )

    window = (
        DEEPSEEK_V4_FLASH_TEACHER_WINDOW_TOKENS
        if teacher_window is None
        else int(teacher_window)
    )
    payload = json.loads(Path(pack_path).read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("teich pack must be a JSON object")
    if payload.get("record_type") != DSV4_PACK_RECORD_TYPE:
        raise ValueError(
            f"teich pack record_type must be {DSV4_PACK_RECORD_TYPE!r}, "
            f"found {payload.get('record_type')!r}"
        )
    if payload.get("model_id") != DSV4_PACK_MODEL_ID:
        raise ValueError(
            f"teich pack model_id must be {DSV4_PACK_MODEL_ID!r}, "
            f"found {payload.get('model_id')!r}"
        )
    rows = payload.get("prompt_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("teich pack must contain a non-empty prompt_rows list")
    declared = payload.get("prompt_row_count")
    if type(declared) is not int:
        raise ValueError("teich pack prompt_row_count must be a JSON integer")
    if declared != len(rows):
        raise ValueError("teich pack prompt_row_count does not match prompt_rows")

    wanted_splits = None if splits is None else {str(s) for s in splits}
    wanted_ids = None if prompt_ids is None else {str(p) for p in prompt_ids}

    sessions: list[Dsv4Session] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"prompt_rows[{index}] must be an object")
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(
                f"prompt_rows[{index}] prompt_id must be a non-empty string"
            )
        if prompt_id in seen:
            raise ValueError(f"teich pack prompt IDs must be unique: {prompt_id}")
        seen.add(prompt_id)

        campaign_split = str(row.get("campaign_split") or "")
        if wanted_splits is not None and campaign_split not in wanted_splits:
            continue
        if wanted_ids is not None and prompt_id not in wanted_ids:
            continue

        raw_ids = _exact_json_int_list(
            row.get("encoded_token_ids"),
            label=f"prompt_rows[{index}] encoded_token_ids",
        )
        if len(raw_ids) < 2:
            raise ValueError(f"prompt_rows[{index}] encoded_token_ids malformed")
        if any(token < 0 or token >= DSV4_PACK_VOCAB_SIZE for token in raw_ids):
            raise ValueError(
                f"prompt_rows[{index}] encoded_token_ids must be within the pinned "
                f"{DSV4_PACK_VOCAB_SIZE}-token vocabulary"
            )
        token_ids = np.asarray(raw_ids, dtype=np.int64)
        if _canonical_sha256(token_ids.tolist()) != row.get("token_ids_sha256"):
            raise ValueError(f"prompt_rows[{index}] token_ids_sha256 mismatch")
        declared_count = row.get("token_count")
        if type(declared_count) is not int:
            raise ValueError(f"prompt_rows[{index}] token_count must be a JSON integer")
        if declared_count != len(raw_ids):
            raise ValueError(f"prompt_rows[{index}] token_count does not match ids")

        raw_positions = _exact_json_int_list(
            row.get("positions"), label=f"prompt_rows[{index}] positions"
        )
        raw_targets = _exact_json_int_list(
            row.get("target_token_ids"),
            label=f"prompt_rows[{index}] target_token_ids",
        )
        if len(raw_positions) != len(raw_targets):
            raise ValueError(f"prompt_rows[{index}] positions/targets malformed")
        if any(right <= left for left, right in pairwise(raw_positions)):
            raise ValueError(
                f"prompt_rows[{index}] positions must be strictly increasing"
            )
        if raw_positions[0] < 0 or raw_positions[-1] >= len(raw_ids) - 1:
            raise ValueError(f"prompt_rows[{index}] positions out of predictor range")
        if raw_targets != [raw_ids[position + 1] for position in raw_positions]:
            raise ValueError(
                f"prompt_rows[{index}] target_token_ids do not equal encoded[pos+1]"
            )
        positions = np.asarray(raw_positions, dtype=np.int64)
        targets = np.asarray(raw_targets, dtype=np.int64)
        if token_ids.size > window:
            raise ValueError(
                f"prompt_rows[{index}] {prompt_id} has {token_ids.size} tokens, past "
                f"the {window}-token teacher window; supervised positions would be "
                "dropped off the end"
            )

        if max_session_tokens is not None and token_ids.size > max_session_tokens:
            continue
        sessions.append(
            Dsv4Session(
                prompt_id=prompt_id,
                campaign_split=campaign_split,
                token_ids=token_ids.astype(np.int32),
                positions=positions,
                target_token_ids=targets,
                token_ids_sha256=str(row["token_ids_sha256"]),
            )
        )

    if wanted_ids is not None:
        missing = sorted(wanted_ids - {s.prompt_id for s in sessions})
        if missing:
            raise ValueError(f"unknown prompt ids in {pack_path}: {missing}")
    if not sessions:
        raise ValueError("no sessions selected from the teich pack")
    # Shortest first: a resumable run should bank cheap sessions early, and the
    # first completion lands in minutes rather than half an hour.
    return tuple(sorted(sessions, key=lambda s: (s.token_count, s.prompt_id)))


# ---------------------------------------------------------------------------
# Routed-expert streaming
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerExpertSpan:
    """Where one layer's routed experts live, and how to read them in runs.

    ``runs`` are the coalesced sequential byte ranges to read (file offsets,
    already past the safetensors header). ``reads`` maps each destination
    (projection, kind, expert) to its source offset and length. Splitting the
    two lets the reader issue large sequential ``preadv`` calls whose
    destinations are the *final* per-projection stacks, with no intermediate
    copy of 3.4 GB.
    """

    layer: int
    shard: str
    runs: tuple[tuple[int, int], ...]
    reads: tuple[tuple[str, str, int, int, int], ...]
    weight_shapes: Mapping[str, tuple[int, ...]]
    scale_shapes: Mapping[str, tuple[int, ...]]
    num_experts: int
    #: The checkpoint block this span belongs to -- ``"layers.7"`` for the
    #: backbone, ``"mtp.0"`` for a DSpark drafter stage. ``layer`` alone cannot
    #: distinguish them (``layers.0`` and ``mtp.0`` both have index 0), so
    #: anything that mixes the two families must key on this.
    block: str = ""

    @property
    def tensor_bytes(self) -> int:
        return sum(length for _, _, _, _, length in self.reads)

    @property
    def span_bytes(self) -> int:
        return sum(end - start for start, end in self.runs)


def build_dsv4_expert_span_index(
    checkpoint_dir: str | Path,
    *,
    layers: Sequence[int] | None = None,
    coalesce_gap_bytes: int = _COALESCE_GAP_BYTES,
) -> dict[int, LayerExpertSpan]:
    """Plan every backbone layer's routed-expert read off the shard headers.

    Backbone only, keyed by layer index -- the historic contract the streaming
    teacher runner is built on. :func:`build_dsv4_block_span_index` is the
    superset that also reaches the three DSpark drafter blocks at ``mtp.*``.
    """

    index = build_dsv4_block_span_index(
        checkpoint_dir,
        blocks=None if layers is None else [f"layers.{int(layer)}" for layer in layers],
        families=("layers",),
        coalesce_gap_bytes=coalesce_gap_bytes,
    )
    return {span.layer: span for span in index.values()}


def build_dsv4_block_span_index(
    checkpoint_dir: str | Path,
    *,
    blocks: Sequence[str] | None = None,
    families: Sequence[str] = ("layers", "mtp"),
    coalesce_gap_bytes: int = _COALESCE_GAP_BYTES,
) -> dict[str, LayerExpertSpan]:
    """Plan every MoE block's routed-expert read off the shard headers.

    A "block" is a checkpoint prefix that owns a full set of routed experts:
    ``layers.{0..42}`` for the backbone and ``mtp.{0,1,2}`` for the DSpark
    drafter, which the release ships as three complete MoE blocks with their
    own 256 routed FP4 experts each. Keyed by block name because ``layers.0``
    and ``mtp.0`` share an index and a materializer walks both.

    Refuses a block whose experts span more than one shard: on this release
    none do (measured -- the backbone in increment 2 §3.7, the three drafter
    blocks land one per shard in 46/47/48), and a block that did would silently
    turn one sequential run into two file opens.
    """

    from mlx_vq.io.source_safetensors import read_safetensors_file_header

    checkpoint = Path(checkpoint_dir)
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    allowed = tuple(str(family) for family in families)
    wanted = None if blocks is None else {str(block) for block in blocks}
    by_block: dict[str, set[str]] = {}
    for name, shard in weight_map.items():
        if ".ffn.experts." not in name:
            continue
        parts = name.split(".")
        if len(parts) < 2 or parts[0] not in allowed:
            continue
        block = f"{parts[0]}.{int(parts[1])}"
        if wanted is not None and block not in wanted:
            continue
        by_block.setdefault(block, set()).add(str(shard))

    if wanted is not None:
        missing = sorted(wanted - set(by_block))
        if missing:
            raise ValueError(f"checkpoint has no routed experts for blocks {missing}")
    if not by_block:
        raise ValueError(f"no routed-expert tensors found under {checkpoint}")
    spanning = {block: sorted(s) for block, s in by_block.items() if len(s) > 1}
    if spanning:
        raise ValueError(
            "layer-sequential streaming requires one shard per block; these span "
            f"more: {spanning}"
        )

    headers: dict[str, Any] = {}
    index: dict[str, LayerExpertSpan] = {}
    for block in sorted(
        by_block, key=lambda key: (key.split(".")[0], int(key.split(".")[1]))
    ):
        layer = int(block.split(".")[1])
        shard = next(iter(by_block[block]))
        if shard not in headers:
            headers[shard] = read_safetensors_file_header(checkpoint / shard)
        header = headers[shard]
        base = header.payload_offset
        prefix = f"{block}.ffn.experts."

        experts: set[int] = set()
        weight_shapes: dict[str, tuple[int, ...]] = {}
        scale_shapes: dict[str, tuple[int, ...]] = {}
        raw: list[tuple[int, str, str, int, int]] = []
        for name, tensor in header.tensors.items():
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix) :].split(".")
            if len(rest) != 3:
                raise ValueError(f"unexpected routed-expert tensor name {name!r}")
            expert_text, projection, kind = rest
            expert = int(expert_text)
            experts.add(expert)
            if projection not in CHECKPOINT_PROJECTIONS:
                raise ValueError(f"unexpected projection in {name!r}")
            if kind == "weight":
                if tensor.dtype != "I8":
                    raise ValueError(
                        f"{name}: expected I8 FP4 codes, got {tensor.dtype}"
                    )
                weight_shapes.setdefault(projection, tuple(tensor.shape))
                if tuple(tensor.shape) != weight_shapes[projection]:
                    raise ValueError(f"{name}: ragged expert weight shape")
            elif kind == "scale":
                if tensor.dtype != "F8_E8M0":
                    raise ValueError(
                        f"{name}: expected F8_E8M0 scales, got {tensor.dtype}"
                    )
                scale_shapes.setdefault(projection, tuple(tensor.shape))
                if tuple(tensor.shape) != scale_shapes[projection]:
                    raise ValueError(f"{name}: ragged expert scale shape")
            else:
                raise ValueError(f"unexpected routed-expert tensor kind in {name!r}")
            start, end = tensor.data_offsets
            raw.append((base + start, projection, kind, expert, end - start))

        num_experts = len(experts)
        if experts != set(range(num_experts)):
            raise ValueError(f"block {block} routed experts are not 0..N-1")
        if sorted(weight_shapes) != sorted(CHECKPOINT_PROJECTIONS) or sorted(
            scale_shapes
        ) != sorted(CHECKPOINT_PROJECTIONS):
            raise ValueError(f"block {block} is missing a routed projection")

        raw.sort()
        runs: list[list[int]] = []
        for offset, _projection, _kind, _expert, length in raw:
            if runs and offset - runs[-1][1] <= coalesce_gap_bytes:
                runs[-1][1] = max(runs[-1][1], offset + length)
            else:
                runs.append([offset, offset + length])
        index[block] = LayerExpertSpan(
            layer=layer,
            shard=shard,
            runs=tuple((start, end) for start, end in runs),
            reads=tuple(
                (projection, kind, expert, offset, length)
                for offset, projection, kind, expert, length in raw
            ),
            weight_shapes=dict(weight_shapes),
            scale_shapes=dict(scale_shapes),
            num_experts=num_experts,
            block=block,
        )
    return index


@dataclass
class Dsv4StreamStats:
    """Cumulative streaming accounting, reported in every run artifact.

    Three different times, deliberately not collapsed into one "read rate":

    ``read_span_seconds``
        submit-to-complete per layer. Once the prefetch is working this
        *includes* the previous layer's compute, so it is a span and not a
        throughput -- dividing bytes by it produces a number that looks like a
        slow disk and is actually a fast one. Named to make that unmistakable.
    ``io_busy_seconds``
        time actually spent inside ``preadv``, summed across worker threads.
        Divided by the pool width this is the wall time the disk was busy, and
        bytes over that is the honest device rate.
    ``read_wait_seconds``
        the only one that costs anything: how long compute blocked waiting for a
        layer's bytes. If the prefetch is doing its job this stays near zero.
    """

    layer_reads: int = 0
    bytes_read: int = 0
    read_span_seconds: float = 0.0
    io_busy_seconds: float = 0.0
    read_wait_seconds: float = 0.0
    convert_seconds: float = 0.0
    io_threads: int = DEFAULT_IO_THREADS

    def as_dict(self) -> dict[str, Any]:
        gb = self.bytes_read / 1e9
        busy_wall = self.io_busy_seconds / max(self.io_threads, 1)
        return {
            "layer_reads": self.layer_reads,
            "gb_read": round(gb, 3),
            "read_span_seconds": round(self.read_span_seconds, 2),
            "io_busy_seconds": round(self.io_busy_seconds, 2),
            "io_gb_per_s": round(gb / busy_wall, 2) if busy_wall > 0 else None,
            "read_wait_seconds": round(self.read_wait_seconds, 2),
            "convert_seconds": round(self.convert_seconds, 2),
        }


class ExpertProvider(Protocol):
    """Supplies one layer's routed experts as a bound ``switch_mlp`` module."""

    def start_prefetch(self, layer: int) -> None: ...

    def take(self, layer: int) -> Any: ...

    def release(self, layer: int) -> None: ...

    def close(self) -> None: ...


class _LayerBuffers:
    """Reusable host destination for one layer's expert bytes."""

    def __init__(self, span: LayerExpertSpan):
        self.weights = {
            projection: np.empty(
                (span.num_experts, *span.weight_shapes[projection]), dtype=np.uint8
            )
            for projection in CHECKPOINT_PROJECTIONS
        }
        self.scales = {
            projection: np.empty(
                (span.num_experts, *span.scale_shapes[projection]), dtype=np.uint8
            )
            for projection in CHECKPOINT_PROJECTIONS
        }
        self.layer: int | None = None

    def destination(self, projection: str, kind: str, expert: int) -> np.ndarray:
        table = self.weights if kind == "weight" else self.scales
        return table[projection][expert]


def _pread_exact(
    fd: int, destination: np.ndarray, offset: int, length: int
) -> tuple[int, float]:
    """Fill ``destination`` from ``offset``; return (bytes, seconds inside I/O).

    ``preadv`` releases the GIL, so the returned busy time is what the disk was
    actually doing rather than what Python was waiting on.
    """

    view = memoryview(destination).cast("B")
    if view.nbytes != length:
        raise ValueError(
            f"destination is {view.nbytes} bytes for a {length}-byte source tensor"
        )
    started = time.perf_counter()
    done = 0
    while done < length:
        got = os.preadv(fd, [view[done:]], offset + done)
        if got <= 0:
            raise OSError(f"short read at offset {offset + done}: {got}")
        done += got
    return done, time.perf_counter() - started


class ShardStreamingExpertProvider:
    """Double-buffered layer-sequential reader over the checkpoint's own bytes.

    Layer N+1's read is issued on a worker pool while layer N computes; the
    read is 0.36 s against ~10 s of compute, so it is fully hidden. Reads land
    **directly** in the per-projection expert stacks -- one ``preadv`` per
    (expert, projection) inside a coalesced sequential run -- so no 3.4 GB
    intermediate ever exists.

    ``F_NOCACHE`` is set by default: a run streams ~150 GB per session and the
    unified buffer cache would evict everything MLX wants in exchange for pages
    nothing reads twice.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        spans: Mapping[int, LayerExpertSpan],
        *,
        swiglu_limit: float,
        io_threads: int = DEFAULT_IO_THREADS,
        nocache: bool = True,
        stats: Dsv4StreamStats | None = None,
    ):
        self.checkpoint = Path(checkpoint_dir)
        self.spans = dict(spans)
        self.swiglu_limit = float(swiglu_limit)
        self.nocache = bool(nocache)
        self.stats = stats if stats is not None else Dsv4StreamStats()
        self.stats.io_threads = max(1, int(io_threads))
        self._pool = ThreadPoolExecutor(
            max_workers=self.stats.io_threads, thread_name_prefix="dsv4-pread"
        )
        self._fds: dict[str, int] = {}
        template = next(iter(self.spans.values()))
        self._buffers = [_LayerBuffers(template), _LayerBuffers(template)]
        self._inflight: dict[int, tuple[float, list[Future]]] = {}
        self._buffer_for: dict[int, _LayerBuffers] = {}
        self._resident: dict[int, Any] = {}

    # -- file descriptors ---------------------------------------------------

    def _fd(self, shard: str) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            fd = os.open(self.checkpoint / shard, os.O_RDONLY)
            if self.nocache:
                import fcntl

                fcntl.fcntl(fd, _F_NOCACHE, 1)
            self._fds[shard] = fd
        return fd

    # -- prefetch -----------------------------------------------------------

    def _free_buffer(self) -> _LayerBuffers:
        busy = {id(buffer) for buffer in self._buffer_for.values()}
        for buffer in self._buffers:
            if id(buffer) not in busy:
                return buffer
        raise RuntimeError("both expert buffers are in flight; prefetch depth is 1")

    def start_prefetch(self, layer: int) -> None:
        if layer in self._inflight or layer in self._resident:
            return
        span = self.spans.get(layer)
        if span is None:
            raise KeyError(f"no routed-expert span planned for layer {layer}")
        buffer = self._free_buffer()
        buffer.layer = layer
        self._buffer_for[layer] = buffer
        fd = self._fd(span.shard)
        # Every read is submitted flat onto the pool -- no task waits on another
        # task, so no depth of nesting can deadlock a bounded pool. Each
        # (expert, projection) weight read is ~4 MB and lands directly in its
        # slot of the final stacked array.
        started = time.perf_counter()
        futures = [
            self._pool.submit(
                _pread_exact,
                fd,
                buffer.destination(projection, kind, expert),
                offset,
                length,
            )
            for projection, kind, expert, offset, length in span.reads
        ]
        self._inflight[layer] = (started, futures)

    # -- materialise --------------------------------------------------------

    def take(self, layer: int) -> Any:
        resident = self._resident.get(layer)
        if resident is not None:
            return resident
        if layer not in self._inflight:
            self.start_prefetch(layer)
        wait_started = time.perf_counter()
        submitted, futures = self._inflight.pop(layer)
        wait(futures)
        results = [future.result() for future in futures]
        finished = time.perf_counter()
        self.stats.layer_reads += 1
        self.stats.bytes_read += sum(count for count, _busy in results)
        self.stats.io_busy_seconds += sum(busy for _count, busy in results)
        self.stats.read_span_seconds += finished - submitted
        self.stats.read_wait_seconds += finished - wait_started

        import mlx.core as mx

        from ramp.models.deepseek_v4_flash_adapter import LimitedSwiGLU

        started = time.perf_counter()
        buffer = self._buffer_for.pop(layer)
        linears = {}
        for projection, module_name in CHECKPOINT_PROJECTIONS.items():
            codes = mx.array(buffer.weights[projection]).view(mx.uint32)
            scales = mx.array(buffer.scales[projection])
            linears[module_name] = _MXFP4SwitchLinear(codes, scales)
        mx.eval([value.weight for value in linears.values()])
        mx.eval([value.scales for value in linears.values()])
        buffer.layer = None
        glu = _StatsSwitchGLU(
            gate_proj=linears["gate_proj"],
            up_proj=linears["up_proj"],
            down_proj=linears["down_proj"],
            activation=LimitedSwiGLU(self.swiglu_limit),
        )
        self.stats.convert_seconds += time.perf_counter() - started
        self._resident[layer] = glu
        return glu

    def release(self, layer: int) -> None:
        self._resident.pop(layer, None)

    def close(self) -> None:
        for _submitted, futures in list(self._inflight.values()):
            for future in futures:
                future.cancel()
        self._inflight.clear()
        self._buffer_for.clear()
        self._resident.clear()
        self._pool.shutdown(wait=False)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


class InMemoryExpertProvider:
    """Every layer's experts already built -- the test and tiny-model path."""

    def __init__(
        self, modules: Mapping[int, Any], stats: Dsv4StreamStats | None = None
    ):
        self._modules = dict(modules)
        self.stats = stats if stats is not None else Dsv4StreamStats()
        self.prefetched: list[int] = []

    def start_prefetch(self, layer: int) -> None:
        if layer in self._modules:
            self.prefetched.append(layer)

    def take(self, layer: int) -> Any:
        return self._modules[layer]

    def release(self, layer: int) -> None:
        return None

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# MLX modules: native-mxfp4 routed experts with an activation tap
# ---------------------------------------------------------------------------


def _switch_linear_classes():
    import mlx.core as mx
    from mlx import nn

    class MXFP4SwitchLinear(nn.Module):
        """``gather_qmm`` over the release's own FP4 bytes -- no dequantisation."""

        def __init__(self, weight_u32, scales_u8):
            super().__init__()
            self.weight = weight_u32
            self.scales = scales_u8
            self.group_size = 32
            self.bits = 4
            self.mode = "mxfp4"

        def __call__(self, x, indices, sorted_indices=False):
            return mx.gather_qmm(
                x,
                self["weight"],
                self["scales"],
                None,
                rhs_indices=indices,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                sorted_indices=sorted_indices,
            )

    return MXFP4SwitchLinear


def _stats_switch_glu_class():
    import mlx.core as mx
    from mlx import nn
    from mlx_lm.models.switch_layers import SwitchGLU, _scatter_unsort

    class StatsSwitchGLU(SwitchGLU):
        """``SwitchGLU.__call__`` verbatim, plus a tap on the down-proj input.

        Two departures from the base class, both deliberate:

        * ``__init__`` does not run ``SwitchGLU.__init__``, which builds three
          ``SwitchLinear`` layers out of ``mx.random.uniform``. At this model's
          shapes that is 8.6 GB of noise per projection, allocated and thrown
          away a line later.
        * ``__call__`` exposes ``activation(x_up, x_gate)`` -- the rows the
          ``down_proj`` actually consumes -- to an optional hook, together with
          the expert ids they are paired with and the permutation
          ``_gather_sort`` applied (``None`` when the routed slot count was
          below the sort threshold). The imatrix accumulator wants that exact
          pairing, so unsorting the activations first would be work undone
          immediately -- but it needs the permutation to line the router
          *scores* up, and re-deriving it by repeating the ``argsort`` would
          silently disagree the moment upstream changes the sort.

        With ``down_hook is None`` the body is the base class's expression
        unchanged; ``test_stats_switch_glu_matches_upstream`` asserts
        ``mx.array_equal`` against it in both states.
        """

        def __init__(self, gate_proj, up_proj, down_proj, activation):
            nn.Module.__init__(self)
            self.gate_proj = gate_proj
            self.up_proj = up_proj
            self.down_proj = down_proj
            self.activation = activation
            self.down_hook: Callable[[Any, Any, Any], None] | None = None

        def __call__(self, x, indices):
            x = mx.expand_dims(x, (-2, -3))
            do_sort = indices.size >= 64
            idx = indices
            inv_order = None
            order = None
            if do_sort:
                # ``_gather_sort`` inlined so the permutation the hook reports
                # is literally the one the projections consumed, not a second
                # ``argsort`` that merely ought to agree.
                width = indices.shape[-1]
                flat = indices.flatten()
                order = mx.argsort(flat)
                x = x.flatten(0, -3)[order // width]
                idx = flat[order]
                inv_order = mx.argsort(order)
            if self.training:
                idx = mx.stop_gradient(idx)
            x_up = self.up_proj(x, idx, sorted_indices=do_sort)
            x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
            activated = self.activation(x_up, x_gate)
            if self.down_hook is not None:
                self.down_hook(activated, idx, order)
            x = self.down_proj(activated, idx, sorted_indices=do_sort)
            if do_sort:
                x = _scatter_unsort(x, inv_order, indices.shape)
            return x.squeeze(-2)

    return StatsSwitchGLU


class _LazyClass:
    """Defer an MLX-importing class body until first use (headless import)."""

    def __init__(self, factory: Callable[[], type]):
        self._factory = factory
        self._cls: type | None = None

    def _resolve(self) -> type:
        if self._cls is None:
            self._cls = self._factory()
        return self._cls

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)

    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, self._resolve())


_MXFP4SwitchLinear = _LazyClass(_switch_linear_classes)
_StatsSwitchGLU = _LazyClass(_stats_switch_glu_class)


# ---------------------------------------------------------------------------
# Calibration accumulation -- on device
# ---------------------------------------------------------------------------


class _ImatrixAccumulator:
    """Per-(layer, space) sums of ``activation^2`` over routed rows, on device.

    The host-NumPy path in :mod:`mlx_vq.quality.imatrix` materialises
    ``[tokens * top_k, input_dim]`` float64; at 46 k tokens x 6 routes x 4096
    dims that is 18 GB per layer per session, and the calibration split would
    move a couple of TB through the host. The same quantity is a **scatter-add**
    on device: for each routed slot column, add the token's squared activation
    row into its expert's accumulator row.

    *Why scatter-add and not a one-hot ``count^T @ squared`` matmul,* which is
    the obvious GPU formulation: measured on this MLX/Metal build, a float32
    ``mx.matmul`` of that shape agrees with the NumPy reference to only
    **5.9e-4** relative -- roughly fp16 epsilon, and one-sided (it always
    undercounts), so it is a systematic bias rather than noise. The scatter-add
    formulation agrees to **1.8e-7**, i.e. exactly what fp32 accumulation should
    give, and costs less memory because nothing ``[rows, experts]`` is built.
    An importance matrix would probably survive 6e-4; shipping a silent,
    one-sided disagreement with the repo's own reference would not.

    Device sums are float32 and fold into float64 host accumulators once per
    layer, so cross-session and cross-layer accumulation carries no float32
    error at all.
    """

    def __init__(
        self, *, layers: Sequence[int], num_experts: int, dims: Mapping[str, int]
    ):
        self.layers = tuple(int(layer) for layer in layers)
        self.num_experts = int(num_experts)
        self.dims = dict(dims)
        self._layer_row = {layer: row for row, layer in enumerate(self.layers)}
        n = len(self.layers)
        self.importance = {
            space: np.zeros((n, self.num_experts, self.dims[space]), dtype=np.float64)
            for space in self.dims
        }
        self.affinity = {
            space: np.zeros((n, self.num_experts, self.dims[space]), dtype=np.float64)
            for space in self.dims
        }
        self.route_count = {
            space: np.zeros((n, self.num_experts), dtype=np.int64)
            for space in self.dims
        }
        self.affinity_score_sum = {
            space: np.zeros((n, self.num_experts), dtype=np.float64)
            for space in self.dims
        }
        self.total_route_count = {
            space: np.zeros(n, dtype=np.int64) for space in self.dims
        }
        self._device: dict[tuple[int, str], list[Any]] = {}

    def add_chunk(
        self,
        *,
        layer: int,
        space: str,
        squared_rows: Any,
        slot_columns: Sequence[tuple[Any, Any]],
    ) -> None:
        """Accumulate one chunk from ``(expert_ids, scores)`` column pairs.

        Every pair scatters the same ``squared_rows`` (one row per token, or one
        per routed slot in the down space) into the accumulator rows its expert
        ids name.
        """

        import mlx.core as mx

        if space not in self.dims:
            raise KeyError(f"unknown imatrix space {space!r}")
        key = (layer, space)
        bucket = self._device.get(key)
        if bucket is None:
            dims = self.dims[space]
            bucket = [
                mx.zeros((self.num_experts, dims), dtype=mx.float32),
                mx.zeros((self.num_experts, dims), dtype=mx.float32),
                mx.zeros((self.num_experts,), dtype=mx.float32),
                mx.zeros((self.num_experts,), dtype=mx.float32),
                0,
            ]
            self._device[key] = bucket

        rows = int(squared_rows.shape[0])
        ones = mx.ones((rows,), dtype=mx.float32)
        for expert_ids, scores in slot_columns:
            ids = expert_ids.reshape(-1)
            weights = scores.reshape(-1).astype(mx.float32)
            if int(ids.size) != rows:
                raise ValueError(
                    f"slot column has {int(ids.size)} entries for {rows} rows"
                )
            bucket[0] = bucket[0].at[ids].add(squared_rows)
            bucket[1] = bucket[1].at[ids].add(squared_rows * weights[:, None])
            bucket[2] = bucket[2].at[ids].add(ones)
            bucket[3] = bucket[3].at[ids].add(weights)
            bucket[4] += rows

    def eval_pending(self) -> None:
        """Realise the running device sums so the graph stays one chunk deep.

        Without this a 76-chunk session builds a 76-deep addition graph holding
        every chunk's ``[rows, dims]`` squared activations alive until the first
        ``eval``, which is exactly the memory the streaming design exists to
        avoid.
        """

        import mlx.core as mx

        for bucket in self._device.values():
            mx.eval(bucket[0], bucket[1], bucket[2], bucket[3])

    def flush_device(self) -> None:
        """Fold every device accumulator into the float64 host totals."""

        import mlx.core as mx

        for (layer, space), bucket in self._device.items():
            importance, affinity, counts, score_sum, total_rows = bucket
            mx.eval(importance, affinity, counts, score_sum)
            row = self._layer_row[layer]
            self.importance[space][row] += np.asarray(importance, dtype=np.float64)
            self.affinity[space][row] += np.asarray(affinity, dtype=np.float64)
            # Counts arrive as float32 scatter-added ones; they are exact small
            # integers there, but rint before the cast so a future non-integral
            # weighting truncates loudly instead of quietly.
            self.route_count[space][row] += np.rint(
                np.asarray(counts, dtype=np.float64)
            ).astype(np.int64)
            self.affinity_score_sum[space][row] += np.asarray(
                score_sum, dtype=np.float64
            )
            self.total_route_count[space][row] += int(total_rows)
        self._device.clear()

    def as_arrays(self) -> dict[str, np.ndarray]:
        self.flush_device()
        arrays: dict[str, np.ndarray] = {
            "layers": np.asarray(self.layers, dtype=np.int32),
            "num_experts": np.asarray(self.num_experts, dtype=np.int32),
        }
        for space in sorted(self.dims):
            arrays[f"importance_sum__{space}"] = self.importance[space].astype(
                np.float32
            )
            arrays[f"affinity_weighted_importance__{space}"] = self.affinity[
                space
            ].astype(np.float32)
            arrays[f"route_count__{space}"] = self.route_count[space].astype(np.int64)
            arrays[f"affinity_score_sum__{space}"] = self.affinity_score_sum[
                space
            ].astype(np.float64)
            arrays[f"total_route_count__{space}"] = self.total_route_count[
                space
            ].astype(np.int64)
        return arrays


def routed_slot_columns(indices: Any, scores: Any, *, per_row: bool):
    """Split routed slots into ``(expert_ids, scores)`` column pairs to scatter.

    ``per_row=False`` (the gate/up input space): ``indices`` is
    ``[tokens, top_k]`` and each token's row is consumed once per slot, so the
    slots become ``top_k`` pairs that all scatter the *same* activation rows --
    no ``[tokens * top_k, dims]`` repeat is ever materialised.

    ``per_row=True`` (the down input space): ``indices`` is ``[rows]``, already
    one entry per routed slot, so it is a single pair.
    """

    if per_row:
        return [(indices.reshape(-1), scores.reshape(-1))]
    width = int(indices.shape[-1])
    flat_inds = indices.reshape(-1, width)
    flat_scores = scores.reshape(-1, width)
    return [(flat_inds[:, slot], flat_scores[:, slot]) for slot in range(width)]


# ---------------------------------------------------------------------------
# Layer-major prefill
# ---------------------------------------------------------------------------


def layer_major_prefill(
    model: Any,
    token_ids: Any,
    *,
    chunk: int,
    expert_provider: ExpertProvider,
    accumulator: _ImatrixAccumulator | None = None,
    collect_spaces: Sequence[str] = (),
    on_layer: Callable[[int, int], None] | None = None,
    release_after_layer: bool = True,
    dspark_taps: dict[int, list[Any]] | None = None,
) -> list[Any]:
    """Prefill the whole backbone layer-outer, chunk-inner.

    Returns the per-chunk final hidden states (pre-``hc_head``), one array of
    shape ``[1, chunk, hc_mult, hidden]`` per chunk. Bit-identical to the
    chunk-outer loop: layer ``i`` chunk ``j`` sees the same input rows and the
    same cache state either way, and the attention mask is taken from the first
    layer's cache exactly as :meth:`DeepseekV4FlashBackbone.__call__` does.

    Each layer's KV and pooled caches are built, used, and dropped inside that
    layer's pass -- a prefill-only run never needs them again, and holding 43
    layers' pooled windows is what makes the chunk-major loop expensive.

    ``dspark_taps`` is the opt-in DSpark capture and is ``None`` on every
    existing caller. Pass a dict and it is filled with ``{layer_index:
    [per-chunk tap arrays]}`` for each ``config.dspark_target_layer_ids`` layer,
    using the adapter's own ``_dspark_tap`` so the runner cannot drift from
    :meth:`DeepseekV4FlashBackbone.__call__`. Layer-major order is why this has
    to be a dict of lists rather than one concatenated array: the taps for layer
    40 are complete before layer 41 has run at all.
    """

    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import CacheList, RotatingKVCache

    from ramp.models.deepseek_v4_flash_adapter import (
        PoolingCache,
        SparseCompressedAttention,
        _dspark_tap,
        hc_expand,
    )

    args = model.args
    tap_layers: set[int] = set()
    if dspark_taps is not None:
        target_ids = list(getattr(args, "dspark_target_layer_ids", []) or [])
        if not target_ids:
            raise ValueError(
                "dspark_taps was requested but the config has no "
                "dspark_target_layer_ids"
            )
        out_of_range = [i for i in target_ids if not 0 <= i < args.num_hidden_layers]
        if out_of_range:
            raise ValueError(
                f"dspark_target_layer_ids outside the backbone: {out_of_range}"
            )
        tap_layers = set(target_ids)
    total = int(token_ids.shape[1])
    n_chunks = max(1, math.ceil(total / chunk))
    id_chunks = [
        token_ids[:, start : start + chunk] for start in range(0, total, chunk)
    ]

    hidden: list[Any] = []
    for piece in id_chunks:
        embedded = model.model.embed_tokens(piece)
        hidden.append(
            mx.contiguous(
                mx.broadcast_to(
                    embedded[:, :, None, :],
                    (
                        embedded.shape[0],
                        embedded.shape[1],
                        args.hc_mult,
                        embedded.shape[2],
                    ),
                )
            )
        )
    mx.eval(hidden)

    masks: list[Any] = []
    spaces = tuple(collect_spaces)
    collect_hidden = "hidden" in spaces
    collect_down = "down" in spaces

    expert_provider.start_prefetch(0)
    for layer_index, layer in enumerate(model.model.layers):
        switch_mlp = expert_provider.take(layer_index)
        layer.ffn.switch_mlp = switch_mlp
        if layer_index + 1 < len(model.model.layers):
            expert_provider.start_prefetch(layer_index + 1)

        ratio = layer.attn.compress_ratio
        if ratio == 0:
            layer_cache: Any = RotatingKVCache(max_size=args.sliding_window)
        elif isinstance(layer.attn, SparseCompressedAttention):
            layer_cache = CacheList(
                RotatingKVCache(max_size=args.sliding_window),
                PoolingCache(ratio),
                PoolingCache(ratio),
            )
        else:
            layer_cache = CacheList(
                RotatingKVCache(max_size=args.sliding_window), PoolingCache(ratio)
            )

        capture: dict[str, Any] = {}
        if accumulator is not None and collect_down:

            def down_hook(activated, expert_ids, order, _capture=capture):
                _capture["down"] = (activated, expert_ids, order)

            switch_mlp.down_hook = down_hook
        else:
            switch_mlp.down_hook = None

        for chunk_index, piece in enumerate(id_chunks):
            h = hidden[chunk_index]
            if layer_index == 0:
                first = layer_cache
                masks.append(
                    create_attention_mask(
                        h[:, :, 0, :],
                        first[0] if isinstance(first, CacheList) else first,
                        window_size=args.sliding_window,
                        return_array=True,
                    )
                )
            mask = masks[chunk_index]

            if accumulator is None:
                hidden[chunk_index] = layer(h, mask, layer_cache, piece)
                if layer_index in tap_layers:
                    tap = _dspark_tap(hidden[chunk_index])
                    dspark_taps.setdefault(layer_index, []).append(tap)
                    mx.eval(tap)
                mx.eval(hidden[chunk_index])
                continue

            # The block body, re-expressed only far enough to see the routed
            # MoE input. Pinned bit-identical against DeepseekV4FlashBlock by
            # test_collecting_stats_does_not_change_the_forward.
            residual = h
            x, post, comb = layer.attn_hc(h)
            x = layer.attn(layer.attn_norm(x), mask=mask, cache=layer_cache)
            h = hc_expand(x, residual, post, comb)

            residual = h
            x, post, comb = layer.ffn_hc(h)
            moe_input = layer.ffn_norm(x)
            inds, scores = layer.ffn.gate(moe_input, piece)
            routed = layer.ffn.switch_mlp(moe_input, inds)
            y = (routed * scores[..., None].astype(routed.dtype)).sum(-2)
            y = y + layer.ffn.shared_experts(moe_input)
            hidden[chunk_index] = hc_expand(y, residual, post, comb)
            if layer_index in tap_layers:
                dspark_taps.setdefault(layer_index, []).append(
                    _dspark_tap(hidden[chunk_index])
                )

            flat_inds = inds.reshape(-1, inds.shape[-1])
            flat_scores = scores.reshape(-1, scores.shape[-1]).astype(mx.float32)
            if collect_hidden:
                rows = moe_input.reshape(-1, moe_input.shape[-1]).astype(mx.float32)
                accumulator.add_chunk(
                    layer=layer_index,
                    space="hidden",
                    squared_rows=rows * rows,
                    slot_columns=routed_slot_columns(
                        flat_inds, flat_scores, per_row=False
                    ),
                )
            if collect_down:
                activated, expert_ids, order = capture.pop("down")
                down_rows = activated.reshape(-1, activated.shape[-1]).astype(
                    mx.float32
                )
                row_scores = flat_scores.reshape(-1)
                if order is not None:
                    row_scores = row_scores[order]
                accumulator.add_chunk(
                    layer=layer_index,
                    space="down",
                    squared_rows=down_rows * down_rows,
                    slot_columns=routed_slot_columns(
                        expert_ids, row_scores, per_row=True
                    ),
                )
            mx.eval(hidden[chunk_index])
            accumulator.eval_pending()

        switch_mlp.down_hook = None
        layer.ffn.switch_mlp = None
        del layer_cache
        if release_after_layer:
            expert_provider.release(layer_index)
        if accumulator is not None:
            accumulator.flush_device()
        mx.clear_cache()
        if on_layer is not None:
            on_layer(layer_index, n_chunks)

    return hidden


# ---------------------------------------------------------------------------
# Top-K logit capture
# ---------------------------------------------------------------------------


def _topk_capture(logits_2d: Any, *, k: int) -> tuple[np.ndarray, ...]:
    """On-device top-K capture. Matches ``topk_capture_numpy`` semantics."""

    import mlx.core as mx

    logits32 = logits_2d.astype(mx.float32)
    logsumexp = mx.logsumexp(logits32, axis=1)
    order = mx.argsort(-logits32, axis=1)[:, :k]
    values = mx.take_along_axis(logits32, order, axis=1)
    topk_mass = mx.exp(values - logsumexp[:, None]).sum(axis=1)
    tail = mx.maximum(1.0 - topk_mass, 0.0)
    mx.eval(order, values, logsumexp, tail)
    return (
        np.array(order).astype(np.int32),
        np.array(values).astype(np.float32),
        np.array(logsumexp).astype(np.float32),
        np.array(tail).astype(np.float32),
    )


def _supervised_logit_capture(
    model: Any,
    hidden: Sequence[Any],
    session: Dsv4Session,
    *,
    chunk: int,
    top_k: int,
    lm_head_slice: int,
) -> dict[str, np.ndarray]:
    """Project only the supervised rows, in bounded slices, then top-K on device.

    ``hc_head`` and ``norm`` are strictly per-position, so gathering the
    supervised rows first is equivalent to projecting the whole session and
    slicing after -- and avoids a ``[46k, 129280]`` float32 tensor that does not
    fit.
    """

    import mlx.core as mx

    positions = session.positions
    chunk_of = positions // chunk
    within = positions % chunk
    # One gather per chunk, not one per position: a 13,587-position session
    # would otherwise build 13,587 graph nodes to collect 13,587 single rows.
    gathered = []
    for chunk_index in range(len(hidden)):
        rows_here = within[chunk_of == chunk_index]
        if rows_here.size == 0:
            continue
        gathered.append(
            mx.take(
                hidden[chunk_index][0], mx.array(rows_here.astype(np.int32)), axis=0
            )
        )
    rows = mx.concatenate(gathered, axis=0)  # [P, hc_mult, hidden]
    mx.eval(rows)
    if rows.shape[0] != positions.size:
        raise ValueError(
            f"gathered {rows.shape[0]} supervised rows for {positions.size} positions"
        )

    ids_out: list[np.ndarray] = []
    values_out: list[np.ndarray] = []
    logsumexp_out: list[np.ndarray] = []
    tail_out: list[np.ndarray] = []
    for start in range(0, rows.shape[0], lm_head_slice):
        piece = rows[start : start + lm_head_slice]
        collapsed = model.model.norm(model.model.hc_head(piece[None, ...]))
        logits = model.lm_head(collapsed)[0]
        ids, values, logsumexp, tail = _topk_capture(logits, k=top_k)
        ids_out.append(ids)
        values_out.append(values)
        logsumexp_out.append(logsumexp)
        tail_out.append(tail)
        del logits, collapsed, piece
        mx.clear_cache()

    return {
        "positions": positions.astype(np.int32),
        "target_token_ids": session.target_token_ids.astype(np.int32),
        "topk_logit_ids": np.concatenate(ids_out, axis=0),
        "topk_logit_values": np.concatenate(values_out, axis=0).astype(np.float16),
        "logsumexp": np.concatenate(logsumexp_out, axis=0),
        "tail_mass": np.concatenate(tail_out, axis=0).astype(np.float16),
    }


# ---------------------------------------------------------------------------
# MTP targets
# ---------------------------------------------------------------------------


def _fuse_dspark_taps(model: Any, taps: Mapping[int, Sequence[Any]]) -> Any:
    """``{layer: [chunk, ...]}`` -> one ``[1, T, hidden * n_targets]`` array.

    Concatenated over chunks first (restoring session order within a layer),
    then over ``dspark_target_layer_ids`` in *that list's* order -- which is the
    column order ``mtp.0.main_proj`` was trained against, not sorted order.
    """

    import mlx.core as mx

    target_ids = list(model.args.dspark_target_layer_ids)
    missing = [i for i in target_ids if i not in taps]
    if missing:
        raise RuntimeError(f"DSpark target tap missing for layer(s) {missing}")
    per_layer = [mx.concatenate(list(taps[i]), axis=1) for i in target_ids]
    fused = mx.concatenate(per_layer, axis=-1)
    mx.eval(fused)
    return fused


def mtp_target_capture(
    model: Any,
    taps: Mapping[int, Sequence[Any]],
    session: Dsv4Session,
    *,
    top_k: int,
    draft_width: int | None = None,
) -> dict[str, np.ndarray]:
    """Per supervised position: the drafter's block logits and final hiddens.

    **Schema** (mirrors the logits mode, one draft slot deeper). ``P`` is the
    session's supervised count, ``W`` the draft width, ``K`` the top-K:

    ==========================  ==============  =========================
    array                       dtype/shape     meaning
    ==========================  ==============  =========================
    ``positions``               i32 ``[P]``     supervised positions
    ``target_token_ids``        i32 ``[P]``     next token, as in ``logits``
    ``mtp_draft_width``         i32 scalar      ``W``
    ``mtp_target_token_ids``    i32 ``[P, W]``  tokens ``p+1 .. p+W``, -1 past the end
    ``mtp_target_valid``        bool ``[P, W]`` False where the session ran out
    ``mtp_topk_logit_ids``      i32 ``[P,W,K]`` drafter's top-K vocabulary ids
    ``mtp_topk_logit_values``   f16 ``[P,W,K]`` their logits
    ``mtp_logsumexp``           f32 ``[P, W]``  full-vocabulary normaliser
    ``mtp_tail_mass``           f16 ``[P, W]``  probability outside the top-K
    ``mtp_final_hidden``        f16 ``[P,W,H]`` drafter ``hc_head`` output
    ==========================  ==============  =========================

    ``mtp_final_hidden`` is the drafter's *final* hidden -- the exact input
    ``norm`` + ``lm_head`` consume -- so ``(mtp_final_hidden, mtp_topk_*)`` is a
    complete, self-contained recovery-training pair for the MTP head: no
    re-running the 163 GB teacher to regenerate the input side.

    **Position semantics -- the seam.** At supervised position ``p`` the
    committed context is the taps for ``0 .. p-1``, **not** ``0 .. p``, and the
    anchor is ``token_ids[p]``. So the ring ends at offset ``p``, the block's
    RoPE starts at ``p``, slot 0 holds token ``p`` and predicts ``p+1``. This is
    the convention
    :meth:`~ramp.models.deepseek_v4_flash_adapter.DeepseekV4FlashVQModel.dspark_calibration_forward`
    owns; committing the anchor's own tap as well runs, stays finite, and is
    wrong twice (every slot's RoPE shifted by one against the Wave-5 runtime,
    and the drafter conditioned on a backbone hidden it cannot have at
    inference). Slot 0's target is therefore ``target_token_ids[p]``, asserted
    rather than assumed.

    **Cost, stated rather than discovered later.** The context ring is appended
    forward once (positions are ascending), so the context work is linear in
    the session, but each position still runs a ``W``-wide block through every
    drafter stage. At the released shapes that is ``3 stages x W`` token-layers
    per supervised position against the backbone's 43, i.e. roughly a tenth of
    the backbone pass.

    Disk, per supervised position, at ``W=5``, ``K=2048``, ``H=4096`` -- every
    term, because the hidden is the one people forget and it is half the total:

    ==========================  ==========================  =========
    array                       arithmetic                  bytes
    ==========================  ==========================  =========
    ``mtp_topk_logit_ids``      ``5 x 2048 x 4`` (i32)      40,960
    ``mtp_final_hidden``        ``5 x 4096 x 2`` (f16)      40,960
    ``mtp_topk_logit_values``   ``5 x 2048 x 2`` (f16)      20,480
    everything else             targets/valid/lse/tail      55
    **total**                                               **~102 KB**
    ==========================  ==========================  =========

    At ~12.5 k supervised positions that is **~1.3 GB per session**, and over
    the 120-session ``mtp-train`` split **~154 GB**. The transient host cost is
    the same ~1.3 GB (the two fp16 arrays are cast at capture, not at the final
    stack -- holding them as float32 would make it ~2.0 GB). ``--mtp-draft-width``
    scales every one of those numbers linearly and is the knob to turn if disk
    binds; ``top_k`` scales the two logit terms only.
    """

    import mlx.core as mx

    if model.mtp_drafter is None:
        raise MtpDrafterUnavailable(
            "mode 'mtp-targets' needs a model built with_mtp=True; this one has "
            "no DSpark drafter"
        )
    unbound = [
        stage
        for stage, block in enumerate(model.mtp_drafter.blocks)
        if block.ffn.switch_mlp is None
    ]
    if unbound:
        raise MtpDrafterUnavailable(
            f"DSpark drafter stage(s) {unbound} have no routed experts bound"
        )

    args = model.args
    width = int(draft_width or args.dspark_block_size)
    if width <= 0:
        raise ValueError("mtp draft width must be positive")
    width = min(width, int(args.dspark_block_size))

    fused = _fuse_dspark_taps(model, taps)
    total = int(fused.shape[1])
    if total != int(session.token_count):
        raise ValueError(
            f"DSpark taps cover {total} tokens, session has {session.token_count}"
        )

    positions = session.positions.astype(np.int64)
    token_ids = session.token_ids.astype(np.int64)

    # Block targets, padded past the end of the session rather than truncated:
    # a ragged [P, W] would force every consumer to re-derive the validity.
    offsets = np.arange(1, width + 1, dtype=np.int64)
    wanted = positions[:, None] + offsets[None, :]
    valid = wanted < total
    block_targets = np.where(valid, token_ids[np.clip(wanted, 0, total - 1)], -1)
    if not np.array_equal(
        block_targets[:, 0], session.target_token_ids.astype(np.int64)
    ):
        raise ValueError(
            "MTP slot 0 must predict the same token the logits mode supervises; "
            "the pack's positions and target_token_ids disagree with its ids"
        )

    cache = model.make_mtp_cache()
    committed = 0
    ids_out: list[np.ndarray] = []
    values_out: list[np.ndarray] = []
    logsumexp_out: list[np.ndarray] = []
    tail_out: list[np.ndarray] = []
    hidden_out: list[np.ndarray] = []

    for index, position in enumerate(positions.tolist()):
        # Context through ``position - 1``, EXCLUDING the anchor's own tap --
        # the convention ``DeepseekV4FlashVQModel.dspark_calibration_forward``
        # owns and documents. Committing through ``position`` instead shifts
        # every draft slot's RoPE by one against the Wave-5 verify runtime and
        # conditions the drafter on a hidden state it cannot have at inference.
        # Positions ascend, so this walk is one pass over the session, not one
        # per supervised position.
        if position > committed:
            model.dspark_append_context(
                fused[:, committed:position],
                cache,
                start_offset=committed,
            )
            committed = position
        anchor = mx.array(token_ids[position : position + 1][None, :])
        logits, head_hidden = model.dspark_draft_block(
            anchor, cache, draft_length=width
        )
        mx.eval(logits, head_hidden)
        ids, values, logsumexp, tail = _topk_capture(logits[0], k=top_k)
        # Cast at capture, not at the final stack. The schema is fp16 for the
        # two big arrays anyway, and holding them as float32 for the whole
        # session doubles the transient: at 12.5 k supervised positions,
        # W=5, K=2048, H=4096 that is 2.0 GB instead of 1.3 GB per session.
        ids_out.append(ids)
        values_out.append(values.astype(np.float16))
        logsumexp_out.append(logsumexp)
        tail_out.append(tail.astype(np.float16))
        hidden_out.append(np.array(head_hidden[0].astype(mx.float16)))
        del logits, head_hidden
        # Not per position: at 12.5 k positions a per-iteration clear costs more
        # than the fragmentation it avoids, and the block-shaped allocations are
        # identical every step so the pool is exactly what should be reused.
        if (index + 1) % _MTP_CLEAR_CACHE_EVERY == 0:
            mx.clear_cache()
    mx.clear_cache()

    return {
        "positions": positions.astype(np.int32),
        "target_token_ids": session.target_token_ids.astype(np.int32),
        "mtp_draft_width": np.asarray(width, dtype=np.int32),
        "mtp_target_token_ids": block_targets.astype(np.int32),
        "mtp_target_valid": valid,
        "mtp_topk_logit_ids": np.stack(ids_out, axis=0),
        "mtp_topk_logit_values": np.stack(values_out, axis=0),
        "mtp_logsumexp": np.stack(logsumexp_out, axis=0).astype(np.float32),
        "mtp_tail_mass": np.stack(tail_out, axis=0),
        "mtp_final_hidden": np.stack(hidden_out, axis=0),
    }


def validate_dsv4_mtp_targets_capture(
    capture: Mapping[str, Any],
    *,
    top_k: int,
    supervised: int,
    width: int,
    hidden_size: int,
) -> None:
    """Fail closed on the MTP-targets session contract."""

    expected = {
        "positions": (np.dtype(np.int32), (supervised,)),
        "target_token_ids": (np.dtype(np.int32), (supervised,)),
        "mtp_target_token_ids": (np.dtype(np.int32), (supervised, width)),
        "mtp_target_valid": (np.dtype(np.bool_), (supervised, width)),
        "mtp_topk_logit_ids": (np.dtype(np.int32), (supervised, width, top_k)),
        "mtp_topk_logit_values": (np.dtype(np.float16), (supervised, width, top_k)),
        "mtp_logsumexp": (np.dtype(np.float32), (supervised, width)),
        "mtp_tail_mass": (np.dtype(np.float16), (supervised, width)),
        "mtp_final_hidden": (np.dtype(np.float16), (supervised, width, hidden_size)),
    }
    missing = sorted(set(expected) - set(capture))
    if missing:
        raise ValueError(f"mtp-targets capture is missing {missing}")
    for name, (dtype, shape) in expected.items():
        value = np.asarray(capture[name])
        if value.dtype != dtype or value.shape != shape:
            raise ValueError(
                f"{name} must have dtype {dtype} and shape {shape}; found "
                f"{value.dtype} {value.shape}"
            )
    if int(np.asarray(capture["mtp_draft_width"]).reshape(()).item()) != int(width):
        raise ValueError("mtp_draft_width disagrees with the array shapes")

    positions = np.asarray(capture["positions"])
    if positions.size == 0 or np.any(np.diff(positions.astype(np.int64)) <= 0):
        raise ValueError("positions must be non-empty and strictly increasing")

    valid = np.asarray(capture["mtp_target_valid"])
    if not np.all(valid[:, 0]):
        raise ValueError("slot 0 is the next token and is always supervised")
    # Validity is a prefix per row: a hole in the middle means the padding logic
    # drifted from the position arithmetic.
    if np.any(np.diff(valid.astype(np.int8), axis=1) > 0):
        raise ValueError("mtp_target_valid must be a prefix within each position")
    targets = np.asarray(capture["mtp_target_token_ids"])
    if not np.array_equal(targets[:, 0], np.asarray(capture["target_token_ids"])):
        raise ValueError("mtp slot 0 must match target_token_ids")
    if np.any(targets[valid] < 0) or np.any(targets[~valid] != -1):
        raise ValueError("mtp_target_token_ids must be -1 exactly where invalid")

    logsumexp = np.asarray(capture["mtp_logsumexp"])
    if not np.all(np.isfinite(logsumexp)):
        raise ValueError("mtp_logsumexp must be finite")
    values = np.asarray(capture["mtp_topk_logit_values"]).astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("mtp_topk_logit_values must be finite")
    # Same fp16-spacing tolerance as the logits mode: two float32 logits a hair
    # apart can round to the same or an inverted pair of fp16 values.
    tolerance = 1e-2 * np.maximum(np.abs(values[..., :-1]), 1.0)
    if np.any(np.diff(values, axis=-1) > tolerance):
        raise ValueError("mtp_topk_logit_values must be sorted descending")
    if np.any(values[..., 0] - logsumexp > 1e-2 * np.maximum(np.abs(logsumexp), 1.0)):
        raise ValueError("mtp_logsumexp must be at least the top logit")
    ids = np.asarray(capture["mtp_topk_logit_ids"])
    if np.any(ids < 0):
        raise ValueError("mtp_topk_logit_ids must be non-negative")
    for row in (0, ids.shape[0] - 1):
        if np.unique(ids[row, 0]).size != ids.shape[-1]:
            raise ValueError("mtp_topk_logit_ids must be distinct within a slot")
    tail = np.asarray(capture["mtp_tail_mass"]).astype(np.float32)
    if np.any(tail < 0) or np.any(tail >= 1) or not np.all(np.isfinite(tail)):
        raise ValueError("mtp_tail_mass must be finite and in [0, 1)")
    hidden = np.asarray(capture["mtp_final_hidden"]).astype(np.float32)
    if not np.all(np.isfinite(hidden)):
        raise ValueError("mtp_final_hidden must be finite")
    if float(np.std(hidden)) == 0.0:
        raise ValueError("mtp_final_hidden is constant; the drafter did not run")


# ---------------------------------------------------------------------------
# Artifacts: schema, atomic write, integrity
# ---------------------------------------------------------------------------


def session_output_path(out_dir: str | Path, prompt_id: str) -> Path:
    return Path(out_dir) / "sessions" / f"{prompt_id.replace('/', '__')}.npz"


def _atomic_savez(
    target: Path, *, compress: bool = False, **arrays: np.ndarray
) -> None:
    """Write an npz that either exists complete or does not exist.

    ``np.savez`` appends ``.npz`` to any path that does not already end in it,
    so the temp path must end in ``.npz`` itself (``foo.tmp.npz``, not
    ``foo.npz.tmp``) or the write lands somewhere the rename cannot find. Both
    the file and its parent directory are fsynced, so a kill between the write
    and the rename leaves the old file (or no file), never a truncated one.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.stem + ".tmp.npz")
    writer = np.savez_compressed if compress else np.savez
    try:
        with tmp.open("wb") as handle:
            writer(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        tmp.unlink(missing_ok=True)


def _identity_arrays(
    session: Dsv4Session, *, mode: str, chunk: int, generation_sha256: str
) -> dict[str, np.ndarray]:
    """Identity every artifact carries, chunk size included by design."""

    return {
        "record_type": np.asarray(f"dsv4_teacher_{mode.replace('-', '_')}_v1"),
        "schema_version": np.asarray(1, dtype=np.int32),
        "prompt_id": np.asarray(session.prompt_id),
        "campaign_split": np.asarray(session.campaign_split),
        "token_ids_sha256": np.asarray(session.token_ids_sha256),
        "token_count": np.asarray(session.token_count, dtype=np.int64),
        # Chunked prefill is NOT bit-identical to single-shot prefill and the
        # gap compounds with depth (increment 2 §1a). A cache is reproducible
        # only against a pinned chunk, so the chunk travels with the data.
        "prefill_chunk_tokens": np.asarray(chunk, dtype=np.int32),
        "generation_config_sha256": np.asarray(generation_sha256),
    }


def _identity_text(capture: Mapping[str, Any], key: str) -> str:
    value = capture[key]
    return str(np.asarray(value).reshape(()).item())


def _check_identity(
    capture: Mapping[str, Any],
    *,
    session: Dsv4Session,
    mode: str,
    chunk: int,
    generation_sha256: str,
    path: Path,
) -> None:
    expected = f"dsv4_teacher_{mode.replace('-', '_')}_v1"
    if _identity_text(capture, "record_type") != expected:
        raise ValueError(f"{path}: record_type is not {expected}")
    if int(np.asarray(capture["schema_version"]).reshape(()).item()) != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    if _identity_text(capture, "prompt_id") != session.prompt_id:
        raise ValueError(f"{path}: prompt_id drifted")
    if _identity_text(capture, "token_ids_sha256") != session.token_ids_sha256:
        raise ValueError(f"{path}: token_ids_sha256 drifted")
    if _identity_text(capture, "campaign_split") != session.campaign_split:
        raise ValueError(f"{path}: campaign_split drifted")
    if (
        int(np.asarray(capture["token_count"]).reshape(()).item())
        != session.token_count
    ):
        raise ValueError(f"{path}: token_count drifted")
    recorded = int(np.asarray(capture["prefill_chunk_tokens"]).reshape(()).item())
    if recorded != int(chunk):
        raise ValueError(
            f"{path}: written at prefill chunk {recorded}, this run uses {chunk}. "
            "Chunked prefill is not bit-identical across chunk sizes, so mixing "
            "them in one cache is silent corruption. Re-run this session or "
            "match the chunk."
        )
    if _identity_text(capture, "generation_config_sha256") != generation_sha256:
        raise ValueError(f"{path}: generation config drifted")


def validate_dsv4_logits_capture(
    capture: Mapping[str, Any], *, top_k: int, supervised: int
) -> None:
    """Fail closed on the top-K logits session contract."""

    expected = {
        "positions": (np.dtype(np.int32), (supervised,)),
        "target_token_ids": (np.dtype(np.int32), (supervised,)),
        "topk_logit_ids": (np.dtype(np.int32), (supervised, top_k)),
        "topk_logit_values": (np.dtype(np.float16), (supervised, top_k)),
        "logsumexp": (np.dtype(np.float32), (supervised,)),
        "tail_mass": (np.dtype(np.float16), (supervised,)),
    }
    missing = sorted(set(expected) - set(capture))
    if missing:
        raise ValueError(f"logits capture is missing {missing}")
    for name, (dtype, shape) in expected.items():
        value = np.asarray(capture[name])
        if value.dtype != dtype or value.shape != shape:
            raise ValueError(
                f"{name} must have dtype {dtype} and shape {shape}; found "
                f"{value.dtype} {value.shape}"
            )
    positions = np.asarray(capture["positions"])
    if positions.size == 0 or np.any(np.diff(positions.astype(np.int64)) <= 0):
        raise ValueError("positions must be non-empty and strictly increasing")
    if not np.all(np.isfinite(np.asarray(capture["logsumexp"]))):
        raise ValueError("logsumexp must be finite")
    values = np.asarray(capture["topk_logit_values"]).astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("topk_logit_values must be finite")
    # Descending in fp16: two float32 logits a hair apart can round to the same
    # or an inverted pair of fp16 values, so the bound has to scale with fp16's
    # spacing at that magnitude rather than being an absolute epsilon.
    tolerance = 1e-2 * np.maximum(np.abs(values[:, :-1]), 1.0)
    if np.any(np.diff(values, axis=1) > tolerance):
        raise ValueError("topk_logit_values must be sorted descending")
    logsumexp = np.asarray(capture["logsumexp"]).astype(np.float32)
    if np.any(values[:, 0] - logsumexp > 1e-2 * np.maximum(np.abs(logsumexp), 1.0)):
        raise ValueError("logsumexp must be at least the top logit")
    ids = np.asarray(capture["topk_logit_ids"])
    if np.any(ids < 0):
        raise ValueError("topk_logit_ids must be non-negative")
    for row in (0, ids.shape[0] - 1):
        if np.unique(ids[row]).size != ids.shape[1]:
            raise ValueError("topk_logit_ids must be distinct within a position")
    tail = np.asarray(capture["tail_mass"]).astype(np.float32)
    if np.any(tail < 0) or np.any(tail >= 1) or not np.all(np.isfinite(tail)):
        raise ValueError("tail_mass must be finite and in [0, 1)")


def validate_dsv4_calibration_capture(
    capture: Mapping[str, Any], *, layers: Sequence[int], num_experts: int
) -> None:
    """Fail closed on the activation-statistics session contract."""

    stored_layers = np.asarray(capture["layers"]).astype(np.int64)
    if stored_layers.tolist() != [int(layer) for layer in layers]:
        raise ValueError("calibration capture layer set drifted")
    if int(np.asarray(capture["num_experts"]).reshape(()).item()) != int(num_experts):
        raise ValueError("calibration capture expert count drifted")
    n = stored_layers.size
    for space in IMATRIX_SPACES:
        importance = capture.get(f"importance_sum__{space}")
        if importance is None:
            continue
        importance = np.asarray(importance)
        if importance.dtype != np.float32 or importance.ndim != 3:
            raise ValueError(f"importance_sum__{space} must be float32 [L, E, D]")
        if importance.shape[:2] != (n, int(num_experts)):
            raise ValueError(f"importance_sum__{space} has the wrong leading shape")
        if not np.all(np.isfinite(importance)):
            raise ValueError(f"importance_sum__{space} must be finite")
        if np.any(importance < 0):
            raise ValueError(f"importance_sum__{space} is a sum of squares; found < 0")
        affinity = np.asarray(capture[f"affinity_weighted_importance__{space}"])
        if affinity.shape != importance.shape or affinity.dtype != np.float32:
            raise ValueError(f"affinity_weighted_importance__{space} shape/dtype drift")
        if not np.all(np.isfinite(affinity)) or np.any(affinity < 0):
            raise ValueError(
                f"affinity_weighted_importance__{space} must be finite >= 0"
            )
        route_count = np.asarray(capture[f"route_count__{space}"])
        if route_count.shape != (n, int(num_experts)) or route_count.dtype != np.int64:
            raise ValueError(f"route_count__{space} shape/dtype drift")
        if np.any(route_count < 0):
            raise ValueError(f"route_count__{space} must be non-negative")
        total = np.asarray(capture[f"total_route_count__{space}"])
        if total.shape != (n,) or total.dtype != np.int64:
            raise ValueError(f"total_route_count__{space} shape/dtype drift")
        if np.any(total <= 0):
            raise ValueError(f"total_route_count__{space} must be positive")
        if np.any(route_count.sum(axis=1) != total):
            raise ValueError(
                f"route_count__{space} rows must sum to total_route_count__{space}"
            )


def calibration_session_arrays(
    accumulator: _ImatrixAccumulator,
) -> dict[str, np.ndarray]:
    return accumulator.as_arrays()


def _audit_existing_session(
    path: Path,
    *,
    session: Dsv4Session,
    mode: str,
    chunk: int,
    generation_sha256: str,
    top_k: int,
    layers: Sequence[int],
    num_experts: int,
    mtp_draft_width: int = 0,
    hidden_size: int = 0,
) -> None:
    """Every skip is earned: load the file and re-check it before trusting it."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"existing session output must be a regular file: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            capture = {name: archive[name] for name in archive.files}
    except Exception as error:
        raise ValueError(
            f"existing session output is malformed: {path}: {error}"
        ) from error
    _check_identity(
        capture,
        session=session,
        mode=mode,
        chunk=chunk,
        generation_sha256=generation_sha256,
        path=path,
    )
    if mode in ("logits", "mtp-targets"):
        if mode == "logits":
            validate_dsv4_logits_capture(
                capture, top_k=top_k, supervised=session.supervised_count
            )
        else:
            validate_dsv4_mtp_targets_capture(
                capture,
                top_k=top_k,
                supervised=session.supervised_count,
                width=mtp_draft_width,
                hidden_size=hidden_size,
            )
        if not np.array_equal(capture["positions"], session.positions.astype(np.int32)):
            raise ValueError(f"{path}: positions drifted from the pack")
        if not np.array_equal(
            capture["target_token_ids"], session.target_token_ids.astype(np.int32)
        ):
            raise ValueError(f"{path}: target_token_ids drifted from the pack")
    else:
        validate_dsv4_calibration_capture(
            capture, layers=layers, num_experts=num_experts
        )


# ---------------------------------------------------------------------------
# Heartbeat / progress
# ---------------------------------------------------------------------------


class Dsv4RunHeartbeat:
    """Atomic JSON status file plus an append-only per-session progress log.

    ``status.json`` is what a monitor polls; ``status.log`` is the append-only
    record with one line per session completion, both timestamped and both
    carrying tok/s so a thermal derate shows up as a trend rather than a
    surprise at the end.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str,
        chunk: int,
        total_sessions: int,
        total_tokens: int,
        total_supervised: int,
        extra: Mapping[str, Any] | None = None,
    ):
        self.path = Path(path)
        self.log_path = self.path.with_suffix(".log")
        self.mode = mode
        self.chunk = int(chunk)
        self.started = time.time()
        self.total_sessions = int(total_sessions)
        self.total_tokens = int(total_tokens)
        self.total_supervised = int(total_supervised)
        self.extra = dict(extra or {})
        self.sessions_done = 0
        self.sessions_skipped = 0
        self.tokens_done = 0
        self.supervised_done = 0
        self.compute_seconds = 0.0
        self.session_rates: list[float] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def payload(self, **fields: Any) -> dict[str, Any]:
        elapsed = max(time.time() - self.started, 1e-9)
        rate = (
            self.tokens_done / self.compute_seconds
            if self.compute_seconds > 0
            else None
        )
        remaining_tokens = max(self.total_tokens - self.tokens_done, 0)
        eta = remaining_tokens / rate if rate else None
        peak_gb = None
        try:
            import mlx.core as mx

            peak_gb = round(float(mx.get_peak_memory()) / 1e9, 2)
        except Exception:  # noqa: BLE001 -- headless / no-MLX monitors still work
            peak_gb = None
        return {
            "record_type": "dsv4_teacher_run_status_v1",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": self.mode,
            "prefill_chunk_tokens": self.chunk,
            "elapsed_s": round(elapsed, 1),
            "sessions_done": self.sessions_done,
            "sessions_skipped": self.sessions_skipped,
            "total_sessions": self.total_sessions,
            "tokens_done": self.tokens_done,
            "total_tokens": self.total_tokens,
            "supervised_done": self.supervised_done,
            "total_supervised": self.total_supervised,
            "compute_seconds": round(self.compute_seconds, 1),
            "tokens_per_s": round(rate, 1) if rate else None,
            "last_session_tokens_per_s": round(self.session_rates[-1], 1)
            if self.session_rates
            else None,
            "eta_s": round(eta, 0) if eta else None,
            "peak_gb": peak_gb,
            **self.extra,
            **fields,
        }

    def update(self, **fields: Any) -> dict[str, Any]:
        payload = self.payload(**fields)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, self.path)
        return payload

    def log(self, **fields: Any) -> dict[str, Any]:
        payload = self.update(**fields)
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload


# ---------------------------------------------------------------------------
# Workload loading
# ---------------------------------------------------------------------------


def _authenticate_dsv4_checkpoint(checkpoint_dir: str | Path) -> dict[str, str]:
    """Authenticate the pinned source before constructing the 163 GB model."""

    from mlx_vq.convert.dsv4_resident import (
        CONFIG_SHA256,
        INDEX_SHA256,
        MODEL_ID,
        REVISION,
        _validate_authority,
    )
    from mlx_vq.convert.stream_convert import load_safetensors_index

    checkpoint = Path(checkpoint_dir).resolve()
    index = load_safetensors_index(checkpoint / "model.safetensors.index.json")
    _validate_authority(
        checkpoint,
        index,
        model_id=MODEL_ID,
        revision=REVISION,
        config_sha256=CONFIG_SHA256,
        index_sha256=INDEX_SHA256,
        enforce_pinned_source=True,
    )
    return {
        "source_model_id": MODEL_ID,
        "source_revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "download_completion_sha256": _sha256_file(
            checkpoint / "_KEEP_DOWNLOAD_COMPLETE.json"
        ),
    }


def load_dsv4_streaming_workload(
    *,
    checkpoint_dir: str | Path,
    io_threads: int = DEFAULT_IO_THREADS,
    nocache: bool = True,
    num_hidden_layers: int | None = None,
    with_mtp: bool = False,
    stats: Dsv4StreamStats | None = None,
) -> tuple[Any, ExpertProvider, dict[str, Any]]:
    """Bind all residents and open the layer-sequential expert stream.

    Residents are every non-routed tensor of every layer plus the embedding and
    LM head: FP8 e4m3 codes decoded against their ue8m0 128x128 block scales
    into bf16, which increment 2 §3.2 verified is **bit-exact** on real layer-0
    tensors. Routed experts are never decoded at all, so a local teacher run is
    slow rather than approximate.

    ``with_mtp`` is what ``--mode mtp-targets`` needs. It binds the drafter's
    residents *and* dequantises its three stages' routed experts into dense
    bf16 stacks that stay resident for the whole run. That is deliberate and
    expensive (~13 GB per stage): the drafter is re-entered once per supervised
    position, so streaming its experts would re-read 3.4 GB thousands of times
    per session instead of the 43 reads the backbone pays. A native-mxfp4
    resident drafter would cost ~3.4 GB per stage instead of 13; it needs the
    span index generalised past the ``layers.`` prefix and is the obvious next
    optimisation, not a correctness gap.
    """

    import mlx.core as mx

    from ramp.models.deepseek_v4_flash_adapter import (
        DeepseekV4FlashVQModel,
        bind_deepseek_v4_flash_mtp_dense_experts,
        bind_deepseek_v4_flash_non_vq_weights,
        deepseek_v4_flash_args_from_config,
    )

    checkpoint = Path(checkpoint_dir)
    config = json.loads((checkpoint / "config.json").read_text())
    args = deepseek_v4_flash_args_from_config(config)
    if num_hidden_layers is not None:
        args = replace(
            args,
            num_hidden_layers=int(num_hidden_layers),
            compress_ratios=list(config["compress_ratios"]),
        )
    model = DeepseekV4FlashVQModel(args, with_mtp=with_mtp)

    started = time.perf_counter()
    report = bind_deepseek_v4_flash_non_vq_weights(
        model,
        checkpoint,
        layers=None,
        include_mtp=with_mtp,
        strict=False,
        compute_dtype=mx.bfloat16,
    )
    if with_mtp:
        bind_deepseek_v4_flash_mtp_dense_experts(
            model, checkpoint, compute_dtype=mx.bfloat16
        )
    mx.eval(model.parameters())
    resident_seconds = time.perf_counter() - started

    spans = build_dsv4_expert_span_index(
        checkpoint, layers=range(args.num_hidden_layers)
    )
    missing = sorted(set(range(args.num_hidden_layers)) - set(spans))
    if missing:
        raise ValueError(f"no routed-expert span for layer(s) {missing}")
    provider = ShardStreamingExpertProvider(
        checkpoint,
        spans,
        swiglu_limit=args.swiglu_limit,
        io_threads=io_threads,
        nocache=nocache,
        stats=stats,
    )
    identities = {
        "checkpoint_dir": str(checkpoint),
        "config_sha256": _sha256_file(checkpoint / "config.json"),
        "index_sha256": _sha256_file(checkpoint / "model.safetensors.index.json"),
        "num_hidden_layers": int(args.num_hidden_layers),
        "n_routed_experts": int(args.n_routed_experts),
        "num_experts_per_tok": int(args.num_experts_per_tok),
        "hidden_size": int(args.hidden_size),
        "moe_intermediate_size": int(args.moe_intermediate_size),
        "swiglu_limit": float(args.swiglu_limit),
        "compress_ratios": list(args.compress_ratios),
        "resident_bind_seconds": round(resident_seconds, 2),
        "with_mtp": bool(with_mtp),
        "resident_bound": int(report.bound_count),
        "resident_fp8_decoded": len(report.fp8_block_decoded_tensors),
        "resident_gb": round(
            sum(value.nbytes for _name, value in _flat_parameters(model)) / 1e9,
            3,
        ),
        "expert_bytes_per_layer": int(next(iter(spans.values())).tensor_bytes),
        "expert_span_bytes_per_layer": int(next(iter(spans.values())).span_bytes),
        "expert_mode": "native_mxfp4",
    }
    return model, provider, identities


def _flat_parameters(model: Any) -> list[tuple[str, Any]]:
    from mlx.utils import tree_flatten

    return tree_flatten(model.parameters())


_SOURCE_AUTHORITY_FIELDS = (
    "source_model_id",
    "source_revision",
    "config_sha256",
    "index_sha256",
    "download_completion_sha256",
)


def _source_authority(identities: Mapping[str, Any]) -> dict[str, str]:
    present = {name for name in _SOURCE_AUTHORITY_FIELDS if name in identities}
    if present and present != set(_SOURCE_AUTHORITY_FIELDS):
        raise ValueError(
            "source authority must contain every authenticated identity field"
        )
    return {
        name: str(identities[name])
        for name in _SOURCE_AUTHORITY_FIELDS
        if name in present
    }


def _session_manifest(sessions: Sequence[Dsv4Session]) -> dict[str, dict[str, Any]]:
    return {
        session.prompt_id: {
            "campaign_split": session.campaign_split,
            "token_count": session.token_count,
            "supervised": session.supervised_count,
            "token_ids_sha256": session.token_ids_sha256,
        }
        for session in sessions
    }


def _legacy_generation_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    legacy = {
        key: value
        for key, value in identity.items()
        if key not in {"source_authority", "pack_sha256", "session_inventory_sha256"}
    }
    legacy["record_type"] = "dsv4_teacher_generation_config_v1"
    return legacy


def _validate_existing_run_manifest(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> tuple[str, bool]:
    """Authenticate an existing cache before any manifest replacement."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"existing run manifest must be a regular file: {path}")
    try:
        current = json.loads(path.read_text())
    except Exception as error:
        raise ValueError(
            f"existing run manifest is malformed: {path}: {error}"
        ) from error
    if not isinstance(current, Mapping):
        raise ValueError(f"existing run manifest must be an object: {path}")
    for field in (
        "record_type",
        "schema_version",
        "mode",
        "pack_path",
        "pack_sha256",
        "splits",
        "session_count",
        "sessions",
    ):
        if current.get(field) != expected[field]:
            raise ValueError(f"existing run manifest {field} disagrees with this run")
    if not isinstance(current.get("identities"), Mapping) or _source_authority(
        current["identities"]
    ) != _source_authority(expected["identities"]):
        raise ValueError(
            "existing run manifest source authority disagrees with this run"
        )

    generation = current.get("generation_config")
    generation_sha256 = current.get("generation_config_sha256")
    if not isinstance(generation, Mapping) or generation_sha256 != _canonical_sha256(
        generation
    ):
        raise ValueError("existing run manifest generation identity is malformed")
    if generation.get("prefill_chunk_tokens") != expected["generation_config"].get(
        "prefill_chunk_tokens"
    ):
        raise ValueError(
            "existing run manifest prefill chunk differs; captures are not "
            "bit-identical across chunk sizes"
        )
    if generation == expected["generation_config"]:
        return str(generation_sha256), False
    if generation == _legacy_generation_identity(expected["generation_config"]):
        return str(generation_sha256), True
    raise ValueError(
        "existing run manifest generation identity disagrees with this run"
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_dsv4_teacher_production(
    *,
    mode: str,
    pack_path: str | Path,
    out_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    splits: Sequence[str] | None = None,
    prompt_ids: Sequence[str] | None = None,
    max_session_tokens: int | None = None,
    max_sessions: int | None = None,
    chunk: int = DEFAULT_PREFILL_CHUNK_TOKENS,
    top_k: int = DEFAULT_TOP_K,
    lm_head_slice: int = DEFAULT_LM_HEAD_SLICE,
    mtp_draft_width: int | None = DEFAULT_MTP_DRAFT_WIDTH,
    io_threads: int = DEFAULT_IO_THREADS,
    nocache: bool = True,
    compress_outputs: bool = False,
    status_path: str | Path | None = None,
    workload_loader: Callable[..., tuple[Any, ExpertProvider, dict[str, Any]]]
    | None = None,
    sessions: Sequence[Dsv4Session] | None = None,
    stop_file: str | Path | None = None,
) -> dict[str, Any]:
    """Produce per-session teacher artifacts for every unfinished session.

    Resumable: a session whose output exists is skipped only after the file is
    loaded and re-validated against the pack *and* against this run's prefill
    chunk size. A corrupt or chunk-mismatched file fails the run rather than
    being silently trusted or silently overwritten.
    """

    if mode not in DSV4_TEACHER_MODES:
        raise ValueError(f"mode must be one of {list(DSV4_TEACHER_MODES)}")
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    if top_k <= 0 or lm_head_slice <= 0 or io_threads <= 0:
        raise ValueError("top_k, lm_head_slice, and io_threads must be positive")

    out_root = Path(out_dir)
    (out_root / "sessions").mkdir(parents=True, exist_ok=True)
    stop = Path(stop_file) if stop_file is not None else None

    if sessions is None:
        sessions = load_dsv4_teich_pack(
            pack_path,
            splits=splits,
            prompt_ids=prompt_ids,
            max_session_tokens=max_session_tokens,
        )
    selected = tuple(sessions)
    if max_sessions is not None:
        selected = selected[: int(max_sessions)]
    pack_file = Path(pack_path)
    if not pack_file.is_file():
        raise ValueError(
            f"production DSV4 teacher runs require a pack file: {pack_file}"
        )
    pack_sha256 = _sha256_file(pack_file)
    session_manifest = _session_manifest(selected)
    session_inventory_sha256 = _canonical_sha256(session_manifest)

    source_authority: dict[str, str] = {}
    if workload_loader is None:
        if checkpoint_dir is None:
            raise ValueError("production DSV4 teacher runs require checkpoint_dir")
        source_authority = _authenticate_dsv4_checkpoint(checkpoint_dir)

    import mlx.core as mx

    stats = Dsv4StreamStats()
    # Reset before the residents bind so the reported peak covers the whole
    # process, residents included -- the memory-budget claim is about the total.
    mx.reset_peak_memory()
    loader = workload_loader or load_dsv4_streaming_workload
    model, provider, identities = loader(
        checkpoint_dir=checkpoint_dir,
        io_threads=io_threads,
        nocache=nocache,
        with_mtp=mode == "mtp-targets",
        stats=stats,
    )
    identities = {**identities, **source_authority}
    layers = tuple(range(int(model.args.num_hidden_layers)))
    num_experts = int(model.args.n_routed_experts)
    hidden_size = int(model.args.hidden_size)

    draft_width = 0
    if mode == "mtp-targets":
        if model.mtp_drafter is None:
            raise MtpDrafterUnavailable(
                "mode 'mtp-targets' needs a workload loaded with_mtp=True; the "
                "loader returned a model with no DSpark drafter. The drafter's "
                "three stages must also have routed experts bound -- they stay "
                "resident for the whole run rather than streaming, because the "
                "drafter is re-entered once per supervised position."
            )
        draft_width = int(mtp_draft_width or model.args.dspark_block_size)
        draft_width = min(draft_width, int(model.args.dspark_block_size))
        if draft_width <= 0:
            raise ValueError("mtp draft width must be positive")

    generation_identity = {
        "record_type": "dsv4_teacher_generation_config_v2",
        "mode": mode,
        "prefill_chunk_tokens": int(chunk),
        "batch_size": 1,
        "top_k": int(top_k) if mode in ("logits", "mtp-targets") else None,
        "lm_head_slice": (
            int(lm_head_slice) if mode in ("logits", "mtp-targets") else None
        ),
        "io_threads": int(io_threads),
        "nocache": bool(nocache),
        "compress_outputs": bool(compress_outputs),
        "imatrix_spaces": list(IMATRIX_SPACES) if mode == "calibration" else None,
        "mtp_draft_width": draft_width if mode == "mtp-targets" else None,
        "dspark_target_layer_ids": (
            list(model.args.dspark_target_layer_ids) if mode == "mtp-targets" else None
        ),
        "expert_mode": "native_mxfp4",
        "resident_dtype": "bfloat16",
        "num_hidden_layers": len(layers),
        "teacher_window_tokens": _teacher_window(),
        "source_authority": _source_authority(identities),
        "pack_sha256": pack_sha256,
        "session_inventory_sha256": session_inventory_sha256,
    }
    generation_sha256 = _canonical_sha256(generation_identity)

    manifest = {
        "record_type": "dsv4_teacher_capture_run",
        "schema_version": 1,
        "mode": mode,
        "pack_path": str(pack_path),
        "pack_sha256": pack_sha256,
        "generation_config": generation_identity,
        "generation_config_sha256": generation_sha256,
        "identities": identities,
        "splits": list(splits) if splits else None,
        "session_count": len(selected),
        "sessions": session_manifest,
    }
    manifest_path = out_root / "run-manifest.json"
    existing_count = sum(
        session_output_path(out_root, session.prompt_id).exists()
        or session_output_path(out_root, session.prompt_id).is_symlink()
        for session in selected
    )
    active_generation_sha256 = generation_sha256
    legacy_complete = False
    try:
        if manifest_path.exists() or manifest_path.is_symlink():
            active_generation_sha256, legacy = _validate_existing_run_manifest(
                manifest_path, expected=manifest
            )
            if legacy and existing_count not in (0, len(selected)):
                raise ValueError(
                    "partial legacy cache cannot be mixed with the source-bound "
                    "generation identity"
                )
            legacy_complete = legacy and existing_count == len(selected)
            if legacy and not legacy_complete:
                active_generation_sha256 = generation_sha256
        elif existing_count:
            raise ValueError(
                "existing session files require an authenticated run manifest"
            )
    except Exception:
        provider.close()
        raise

    pending: list[Dsv4Session] = []
    skipped = 0
    for session in selected:
        target = session_output_path(out_root, session.prompt_id)
        if target.exists() or target.is_symlink():
            _audit_existing_session(
                target,
                session=session,
                mode=mode,
                chunk=chunk,
                generation_sha256=active_generation_sha256,
                top_k=top_k,
                layers=layers,
                num_experts=num_experts,
                mtp_draft_width=draft_width,
                hidden_size=hidden_size,
            )
            skipped += 1
        else:
            pending.append(session)

    heartbeat = Dsv4RunHeartbeat(
        status_path or (out_root / "status.json"),
        mode=mode,
        chunk=chunk,
        total_sessions=len(selected),
        total_tokens=sum(s.token_count for s in selected),
        total_supervised=sum(s.supervised_count for s in selected),
        extra={
            "out_dir": str(out_root),
            "pack_path": str(pack_path),
            "generation_config_sha256": active_generation_sha256,
            "pid": os.getpid(),
        },
    )
    heartbeat.sessions_skipped = skipped
    heartbeat.update(phase="planned", pending_sessions=len(pending))

    if not legacy_complete:
        _atomic_json(manifest_path, manifest)

    if not pending:
        provider.close()
        heartbeat.log(phase="complete", sessions_produced=0)
        return {
            "mode": mode,
            "sessions_produced": 0,
            "sessions_skipped": skipped,
            "stream": stats.as_dict(),
        }

    produced = 0
    peak_gb = 0.0
    try:
        produced, peak_gb = _produce_sessions(
            pending,
            model=model,
            provider=provider,
            mode=mode,
            chunk=chunk,
            top_k=top_k,
            lm_head_slice=lm_head_slice,
            draft_width=draft_width,
            layers=layers,
            num_experts=num_experts,
            out_root=out_root,
            generation_sha256=generation_sha256,
            compress_outputs=compress_outputs,
            heartbeat=heartbeat,
            stats=stats,
            stop=stop,
        )
    finally:
        provider.close()

    summary = {
        "mode": mode,
        "sessions_produced": produced,
        "sessions_skipped": skipped,
        "sessions_selected": len(selected),
        "prefill_chunk_tokens": int(chunk),
        "generation_config_sha256": generation_sha256,
        "peak_mlx_gb": round(peak_gb, 2),
        "peak_host_rss_gb": _host_peak_rss_gb(),
        "tokens_per_s": round(heartbeat.tokens_done / heartbeat.compute_seconds, 1)
        if heartbeat.compute_seconds > 0
        else None,
        "stream": stats.as_dict(),
        "out_dir": str(out_root),
    }
    _atomic_json(out_root / "run-summary.json", summary)
    heartbeat.log(phase="complete", summary=summary)
    return summary


def _host_peak_rss_gb() -> float:
    """Peak host RSS in GB.

    MLX's peak counter does not see the NumPy landing buffer the expert stream
    reads into (3.42 GB), so a memory-budget claim that quotes only
    ``mx.get_peak_memory()`` understates the process by one whole expert layer.
    """

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes, Linux kilobytes.
    scale = 1e9 if os.uname().sysname == "Darwin" else 1e6
    return round(peak / scale, 2)


def _produce_sessions(
    pending: Sequence[Dsv4Session],
    *,
    model: Any,
    provider: ExpertProvider,
    mode: str,
    chunk: int,
    top_k: int,
    lm_head_slice: int,
    draft_width: int,
    layers: Sequence[int],
    num_experts: int,
    out_root: Path,
    generation_sha256: str,
    compress_outputs: bool,
    heartbeat: Dsv4RunHeartbeat,
    stats: Dsv4StreamStats,
    stop: Path | None,
) -> tuple[int, float]:
    import mlx.core as mx

    produced = 0
    peak_gb = 0.0
    for index, session in enumerate(pending):
        if stop is not None and stop.exists():
            heartbeat.log(phase="stopped", reason=f"stop file present: {stop}")
            break
        heartbeat.update(
            phase="session-start",
            session=session.prompt_id,
            session_index=index + 1,
            session_tokens=session.token_count,
        )
        started = time.perf_counter()
        token_ids = mx.array(session.token_ids.astype(np.int64)[None, :])
        accumulator = None
        collect: tuple[str, ...] = ()
        if mode == "calibration":
            accumulator = _ImatrixAccumulator(
                layers=layers,
                num_experts=num_experts,
                dims={
                    "hidden": int(model.args.hidden_size),
                    "down": int(model.args.moe_intermediate_size),
                },
            )
            collect = IMATRIX_SPACES

        def on_layer(layer_index: int, n_chunks: int, _s=session, _i=index) -> None:
            heartbeat.update(
                phase="layer-forward",
                session=_s.prompt_id,
                session_index=_i + 1,
                layer=f"{layer_index + 1}/{len(layers)}",
                chunks=n_chunks,
            )

        taps: dict[int, list[Any]] | None = {} if mode == "mtp-targets" else None
        hidden = layer_major_prefill(
            model,
            token_ids,
            chunk=chunk,
            expert_provider=provider,
            accumulator=accumulator,
            collect_spaces=collect,
            on_layer=on_layer,
            dspark_taps=taps,
        )

        if mode == "calibration":
            arrays = calibration_session_arrays(accumulator)
            validate_dsv4_calibration_capture(
                arrays, layers=layers, num_experts=num_experts
            )
        elif mode == "mtp-targets":
            # The backbone hidden states are not needed past this point; the
            # drafter consumes the taps, which are 1/hc_mult of their size.
            del hidden
            mx.clear_cache()
            heartbeat.update(
                phase="mtp-draft",
                session=session.prompt_id,
                session_index=index + 1,
                supervised=session.supervised_count,
            )
            arrays = mtp_target_capture(
                model, taps, session, top_k=top_k, draft_width=draft_width
            )
            validate_dsv4_mtp_targets_capture(
                arrays,
                top_k=top_k,
                supervised=session.supervised_count,
                width=draft_width,
                hidden_size=int(model.args.hidden_size),
            )
            hidden = None
        else:
            arrays = _supervised_logit_capture(
                model,
                hidden,
                session,
                chunk=chunk,
                top_k=top_k,
                lm_head_slice=lm_head_slice,
            )
            validate_dsv4_logits_capture(
                arrays, top_k=top_k, supervised=session.supervised_count
            )
        del hidden, taps
        mx.clear_cache()

        elapsed = time.perf_counter() - started
        arrays = {
            **arrays,
            **_identity_arrays(
                session, mode=mode, chunk=chunk, generation_sha256=generation_sha256
            ),
            "tokens_per_s": np.asarray(session.token_count / elapsed, dtype=np.float64),
            "session_seconds": np.asarray(elapsed, dtype=np.float64),
        }
        _atomic_savez(
            session_output_path(out_root, session.prompt_id),
            compress=compress_outputs,
            **arrays,
        )
        produced += 1
        peak_gb = max(peak_gb, float(mx.get_peak_memory()) / 1e9)
        heartbeat.sessions_done += 1
        heartbeat.tokens_done += session.token_count
        heartbeat.supervised_done += session.supervised_count
        heartbeat.compute_seconds += elapsed
        heartbeat.session_rates.append(session.token_count / elapsed)
        heartbeat.log(
            phase="session-complete",
            session=session.prompt_id,
            session_index=index + 1,
            session_tokens=session.token_count,
            session_seconds=round(elapsed, 2),
            session_tokens_per_s=round(session.token_count / elapsed, 1),
            peak_host_rss_gb=_host_peak_rss_gb(),
            stream=stats.as_dict(),
        )
        # Sessions differ in length by 20x; a per-session accumulator that the
        # next session's shapes will not reuse is dead weight in the cache.
        mx.clear_cache()
    return produced, peak_gb


def _teacher_window() -> int:
    from ramp.models.deepseek_v4_flash_adapter import (
        DEEPSEEK_V4_FLASH_TEACHER_WINDOW_TOKENS,
    )

    return int(DEEPSEEK_V4_FLASH_TEACHER_WINDOW_TOKENS)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Calibration finalize -> imatrix sidecars
# ---------------------------------------------------------------------------


def finalize_dsv4_calibration_imatrix(
    out_dir: str | Path,
    *,
    sidecar_dir: str | Path | None = None,
    prompt_set: str = "dsv4_teich_calibration_v1",
    projections: Sequence[str] = ("gate_proj", "up_proj", "down_proj"),
    layers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Merge per-session activation statistics into imatrix sidecars.

    Writes the schema ``mlx_vq.quality.imatrix`` already consumes -- one
    safetensors per (layer, projection, expert) plus a manifest -- so Wave 5's
    VQ fitting reads DeepSeek-V4 statistics through the same loader as
    GLM-4.5-Air's. ``gate_proj`` and ``up_proj`` share the ``hidden`` input
    space and are written from the same accumulated vector.

    Separate from the run on purpose: at 43 layers x 3 projections x 256
    experts this is 33,024 files, and a two-day streaming run should not be
    holding a lock while it writes them.
    """

    from mlx_vq.quality.imatrix import (
        ProjectionImatrixEntry,
        write_projection_imatrix_sidecars,
    )

    root = Path(out_dir)
    session_files = sorted((root / "sessions").glob("*.npz"))
    if not session_files:
        raise ValueError(f"no session outputs under {root / 'sessions'}")

    totals: dict[str, np.ndarray] = {}
    stored_layers: np.ndarray | None = None
    num_experts = 0
    prompt_ids: list[str] = []
    for path in session_files:
        with np.load(path, allow_pickle=False) as archive:
            capture = {name: archive[name] for name in archive.files}
        if _identity_text(capture, "record_type") != "dsv4_teacher_calibration_v1":
            raise ValueError(f"{path} is not a calibration capture")
        current = np.asarray(capture["layers"]).astype(np.int64)
        if stored_layers is None:
            stored_layers = current
            num_experts = int(np.asarray(capture["num_experts"]).reshape(()).item())
        elif not np.array_equal(stored_layers, current):
            raise ValueError(f"{path} layer set differs from the first session")
        prompt_ids.append(_identity_text(capture, "prompt_id"))
        for space in IMATRIX_SPACES:
            for key in (
                f"importance_sum__{space}",
                f"affinity_weighted_importance__{space}",
                f"route_count__{space}",
                f"affinity_score_sum__{space}",
                f"total_route_count__{space}",
            ):
                if key not in capture:
                    continue
                value = np.asarray(capture[key])
                promoted = value.astype(
                    np.int64 if value.dtype.kind in "iu" else np.float64
                )
                totals[key] = promoted if key not in totals else totals[key] + promoted

    assert stored_layers is not None
    layer_list = (
        stored_layers.tolist() if layers is None else [int(layer) for layer in layers]
    )
    layer_rows = {int(layer): row for row, layer in enumerate(stored_layers.tolist())}
    prompt_tuple = tuple(sorted(set(prompt_ids)))

    entries: list[ProjectionImatrixEntry] = []
    for layer in layer_list:
        row = layer_rows[layer]
        for space in IMATRIX_SPACES:
            key = f"importance_sum__{space}"
            if key not in totals:
                continue
            importance = totals[key][row]
            affinity = totals[f"affinity_weighted_importance__{space}"][row]
            route_count = totals[f"route_count__{space}"][row]
            score_sum = totals[f"affinity_score_sum__{space}"][row]
            total_route = int(totals[f"total_route_count__{space}"][row])
            input_dim = importance.shape[1]
            for projection in _SPACE_PROJECTIONS[space]:
                if projection not in projections:
                    continue
                for expert in range(num_experts):
                    count = int(route_count[expert])
                    sums = importance[expert]
                    entries.append(
                        ProjectionImatrixEntry(
                            layer=layer,
                            projection=projection,
                            expert=expert,
                            importance_sum=sums.astype(np.float32),
                            mean_importance=(
                                sums / count if count else np.zeros(input_dim)
                            ).astype(np.float32),
                            routing_weighted_importance=(
                                sums / total_route
                                if total_route
                                else np.zeros(input_dim)
                            ).astype(np.float32),
                            route_count=count,
                            total_route_count=total_route,
                            route_frequency=count / total_route if total_route else 0.0,
                            prompt_ids=prompt_tuple,
                            affinity_weighted_importance=affinity[expert].astype(
                                np.float32
                            ),
                            affinity_score_sum=float(score_sum[expert]),
                        )
                    )

    if not entries:
        raise ValueError("no imatrix entries produced; check --projections")
    target = Path(sidecar_dir) if sidecar_dir is not None else root / "imatrix-sidecars"
    manifest = write_projection_imatrix_sidecars(
        entries, output_dir=target, prompt_set=prompt_set
    )
    return {
        "record_type": "dsv4_calibration_imatrix_finalize",
        "session_count": len(session_files),
        "prompt_ids": list(prompt_tuple),
        "layers": layer_list,
        "projections": list(projections),
        "num_experts": num_experts,
        "entry_count": manifest["entry_count"],
        "sidecar_dir": str(target),
        "manifest_path": str(target / "imatrix-manifest.json"),
    }
