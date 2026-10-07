"""Teacher agreement for the DeepSeek-V4-Flash VQ artifact.

This is the Wave 5 quality gate: the first measurement in the campaign that is
*not* a proxy. Everything before it -- per-projection relative MSE, block cosine
through the clamped SwiGLU, roundtrip rel-MSE against the fit's own
reconstruction -- measures the fit against its own objective. This module
measures the artifact's **logits** against the teacher's logits on the same
tokens, which is the thing an RC decision is actually about.

Three deliberate choices, each of which the report has to be able to defend:

1. **The metric implementations are the GLM lineage's, not new ones.** Per
   position slice this module builds a GLM-shaped teacher row and calls
   :func:`keep.quality.teacher_cache.evaluate_teacher_cache_row` verbatim, so
   NLL, perplexity, ppl ratio, top-1 agreement, the KLD head sum and the p99.9
   token tail come out of the same code that produced every GLM-4.5-Air and
   GLM-5.2 number. What this module adds on top is only what that evaluator does
   not know how to do: the **explicit tail correction** for a truncated teacher,
   and top-5 / top-10 agreement.

2. **The truncated teacher is handled with bounds, not with a silent zero.** The
   capture stored the top 2048 of 129,280 logits plus the *exact* full-vocabulary
   ``logsumexp`` and the leftover ``tail_mass``. That is strictly more
   information than the GLM caches carried, and it is what makes an honest KLD
   possible -- see :func:`tail_corrected_kld` for the estimator, its bias
   direction and its bounds.

3. **The artifact is streamed layer-sequentially, the same shape as the teacher
   run.** Residents stay resident; each layer's three VQ projections are
   ``preadv``-ed into reusable host buffers with ``F_NOCACHE`` and released after
   the layer. The teacher run's own :class:`Dsv4StreamStats` counters are reused
   so the two runs' I/O lines are directly comparable, and a 70 GB-per-forward
   mmap walk never gets a chance to evict the residents and dirty the pageout
   counters that acceptance depends on.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import struct
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ramp.benchmark.metrics import (
    collect_metric_snapshot,
    collect_vm_stat_counts,
    reset_mlx_peak_memory,
)
from keep.quality.dsv4_teacher_runner import (
    DEFAULT_IO_THREADS,
    DEFAULT_PREFILL_CHUNK_TOKENS,
    Dsv4Session,
    Dsv4StreamStats,
    ExpertProvider,
    _pread_exact,
    layer_major_prefill,
    load_dsv4_streaming_workload,
    load_dsv4_teich_pack,
)
from keep.quality.teacher_cache import (
    evaluate_teacher_cache_row,
    summarize_teacher_cache_records,
)

__all__ = [
    "AGREEMENT_ROW_RECORD_TYPE",
    "AGREEMENT_STATUS_RECORD_TYPE",
    "DEFAULT_POSITION_SLICE",
    "TEACHER_LOGITS_RECORD_TYPE",
    "VQ_PROJECTIONS",
    "AgreementVerdict",
    "Dsv4TeacherSession",
    "Dsv4VqStreamingExpertProvider",
    "VqBlockSpan",
    "VqTensorSpan",
    "agreement_verdict",
    "build_dsv4_vq_block_span_index",
    "heavy_job_lock",
    "load_dsv4_teacher_session",
    "read_agreement_rows",
    "run_dsv4_teacher_agreement",
    "select_teacher_sessions",
    "stratified_indices",
    "summarize_agreement_records",
    "tail_corrected_kld",
    "teacher_session_paths",
    "vq_artifact_identity",
]

TEACHER_LOGITS_RECORD_TYPE = "dsv4_teacher_logits_v1"
VQ_MANIFEST_RECORD_TYPE = "dsv4_vq_artifact_manifest_v1"
AGREEMENT_ROW_RECORD_TYPE = "dsv4_teacher_agreement_row_v1"
AGREEMENT_STATUS_RECORD_TYPE = "dsv4_teacher_agreement_status_v1"

VQ_PROJECTIONS: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
DEFAULT_POSITION_SLICE = 256
DEFAULT_TOP_KS: tuple[int, ...] = (1, 5, 10)

# ``fcntl.F_NOCACHE`` on Darwin -- keep a 70 GB-per-forward stream out of the
# unified buffer cache so it cannot evict the residents.
_F_NOCACHE = 48

_SAFETENSORS_DTYPES: Mapping[str, np.dtype] = {
    "U8": np.dtype(np.uint8),
    "UINT8": np.dtype(np.uint8),
    "U16": np.dtype(np.uint16),
    "UINT16": np.dtype(np.uint16),
    "U32": np.dtype(np.uint32),
    "UINT32": np.dtype(np.uint32),
    "F16": np.dtype(np.float16),
    "FLOAT16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
    "FLOAT32": np.dtype(np.float32),
}


# ---------------------------------------------------------------------------
# Cooperative heavy-job lock (same file, same semantics as the materializer)
# ---------------------------------------------------------------------------


def heavy_job_lock(path: str | Path, *, enabled: bool = True, holder: str) -> Any:
    """The shared KEEP heavy-job lock: cooperative, released on process exit."""

    import contextlib

    @contextlib.contextmanager
    def _ctx() -> Iterator[None]:
        if not enabled:
            yield
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                handle.truncate()
                handle.write(f"{os.getpid()}\n{holder}\n")
                handle.flush()
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    return _ctx()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_u32_bytes(values: np.ndarray) -> str:
    """Same hashing convention as :mod:`keep.vq.e8`."""

    return hashlib.sha256(np.asarray(values, dtype="<u4").tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Teacher-side: the per-session top-K logit capture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dsv4TeacherSession:
    """One session's teacher capture, loaded from its ``.npz``.

    ``topk_logits`` are the raw float32-widened stored logits; ``logsumexp`` is
    the *full-vocabulary* normalizer, so ``exp(topk_logits - logsumexp)`` is the
    teacher's true probability on the retained support with no renormalization,
    and ``tail_mass`` is exactly the probability the capture threw away.
    """

    prompt_id: str
    campaign_split: str
    token_count: int
    token_ids_sha256: str
    prefill_chunk_tokens: int
    generation_config_sha256: str
    path: Path
    positions: np.ndarray
    target_token_ids: np.ndarray
    topk_ids: np.ndarray
    topk_logits: np.ndarray
    logsumexp: np.ndarray
    tail_mass: np.ndarray

    @property
    def supervised_count(self) -> int:
        return int(self.positions.size)

    @property
    def top_k(self) -> int:
        return int(self.topk_ids.shape[1])

    def topk_logprobs(self) -> np.ndarray:
        """Teacher log-probabilities on the retained support, un-renormalized."""

        return self.topk_logits - self.logsumexp[:, None]

    def slice(self, start: int, stop: int) -> Dsv4TeacherSession:
        return Dsv4TeacherSession(
            prompt_id=self.prompt_id,
            campaign_split=self.campaign_split,
            token_count=self.token_count,
            token_ids_sha256=self.token_ids_sha256,
            prefill_chunk_tokens=self.prefill_chunk_tokens,
            generation_config_sha256=self.generation_config_sha256,
            path=self.path,
            positions=self.positions[start:stop],
            target_token_ids=self.target_token_ids[start:stop],
            topk_ids=self.topk_ids[start:stop],
            topk_logits=self.topk_logits[start:stop],
            logsumexp=self.logsumexp[start:stop],
            tail_mass=self.tail_mass[start:stop],
        )


def load_dsv4_teacher_session(path: str | Path) -> Dsv4TeacherSession:
    """Load and validate one teacher-capture ``.npz``."""

    target = Path(path)
    with np.load(target, allow_pickle=False) as payload:
        record_type = str(payload["record_type"])
        if record_type != TEACHER_LOGITS_RECORD_TYPE:
            raise ValueError(
                f"{target.name}: record_type must be {TEACHER_LOGITS_RECORD_TYPE!r}, "
                f"found {record_type!r}"
            )
        positions = np.asarray(payload["positions"], dtype=np.int64)
        targets = np.asarray(payload["target_token_ids"], dtype=np.int64)
        topk_ids = np.asarray(payload["topk_logit_ids"], dtype=np.int64)
        topk_logits = np.asarray(payload["topk_logit_values"], dtype=np.float64)
        logsumexp = np.asarray(payload["logsumexp"], dtype=np.float64)
        tail_mass = np.asarray(payload["tail_mass"], dtype=np.float64)
        session = Dsv4TeacherSession(
            prompt_id=str(payload["prompt_id"]),
            campaign_split=str(payload["campaign_split"]),
            token_count=int(payload["token_count"]),
            token_ids_sha256=str(payload["token_ids_sha256"]),
            prefill_chunk_tokens=int(payload["prefill_chunk_tokens"]),
            generation_config_sha256=str(payload["generation_config_sha256"]),
            path=target,
            positions=positions,
            target_token_ids=targets,
            topk_ids=topk_ids,
            topk_logits=topk_logits,
            logsumexp=logsumexp,
            tail_mass=tail_mass,
        )
    count = session.supervised_count
    if count == 0:
        raise ValueError(f"{target.name}: no supervised positions")
    for name, array, ndim in (
        ("target_token_ids", targets, 1),
        ("topk_logit_ids", topk_ids, 2),
        ("topk_logit_values", topk_logits, 2),
        ("logsumexp", logsumexp, 1),
        ("tail_mass", tail_mass, 1),
    ):
        if array.ndim != ndim or array.shape[0] != count:
            raise ValueError(
                f"{target.name}: {name} must lead with {count} positions, "
                f"found shape {array.shape}"
            )
    if topk_ids.shape != topk_logits.shape:
        raise ValueError(f"{target.name}: topk ids and values must have the same shape")
    if np.any(np.diff(positions) <= 0):
        raise ValueError(f"{target.name}: positions must be strictly increasing")
    if np.any(tail_mass < 0.0) or np.any(tail_mass > 1.0):
        raise ValueError(f"{target.name}: tail_mass outside [0, 1]")
    return session


def teacher_session_paths(cache_dir: str | Path) -> dict[str, Path]:
    root = Path(cache_dir) / "sessions"
    if not root.is_dir():
        raise ValueError(f"no sessions/ directory under {cache_dir}")
    return {path.stem: path for path in sorted(root.glob("*.npz"))}


def stratified_indices(count: int, picks: int) -> tuple[int, ...]:
    """Evenly spaced indices over ``range(count)``, endpoints included.

    Deterministic by construction so a subset run is reproducible and its span
    over the length distribution is inspectable rather than sampled.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if picks <= 0:
        raise ValueError("picks must be positive")
    if picks >= count:
        return tuple(range(count))
    if picks == 1:
        return (count // 2,)
    step = (count - 1) / (picks - 1)
    return tuple(sorted({int(round(index * step)) for index in range(picks)}))


def select_teacher_sessions(
    cache_dir: str | Path,
    *,
    split: str | None = "report",
    prompt_ids: Sequence[str] | None = None,
    stratify: int | None = None,
    max_sessions: int | None = None,
) -> tuple[Dsv4TeacherSession, ...]:
    """Pick teacher sessions, shortest first, optionally length-stratified."""

    paths = teacher_session_paths(cache_dir)
    wanted = None if prompt_ids is None else {str(value) for value in prompt_ids}
    sessions: list[Dsv4TeacherSession] = []
    for prompt_id, path in paths.items():
        if wanted is not None and prompt_id not in wanted:
            continue
        session = load_dsv4_teacher_session(path)
        if split is not None and session.campaign_split != split:
            continue
        sessions.append(session)
    if wanted is not None:
        missing = sorted(wanted - {s.prompt_id for s in sessions})
        if missing:
            raise ValueError(f"unknown or split-filtered prompt ids: {missing}")
    if not sessions:
        raise ValueError(f"no teacher sessions selected from {cache_dir}")
    sessions.sort(key=lambda s: (s.token_count, s.prompt_id))
    if stratify is not None:
        keep = stratified_indices(len(sessions), int(stratify))
        sessions = [sessions[index] for index in keep]
    if max_sessions is not None:
        sessions = sessions[: int(max_sessions)]
    return tuple(sessions)


# ---------------------------------------------------------------------------
# The KLD estimator for a truncated teacher
# ---------------------------------------------------------------------------


def tail_corrected_kld(
    *,
    teacher_topk_logprobs: np.ndarray,
    student_topk_logprobs: np.ndarray,
    teacher_tail_mass: np.ndarray,
    student_head_mass: np.ndarray,
    teacher_min_head_logprob: np.ndarray,
    student_min_logprob: np.ndarray,
) -> dict[str, np.ndarray]:
    r"""Per-position KL(teacher || student) with the truncated tail made explicit.

    The teacher stored the top ``K`` of ``V`` logits plus the exact
    full-vocabulary normalizer, so with ``S`` the retained support:

    .. math::
        \mathrm{KL} = \underbrace{\sum_{i\in S} p_i\log\frac{p_i}{q_i}}_{\text{head, exact}}
                    + \underbrace{\sum_{i\notin S} p_i\log\frac{p_i}{q_i}}_{T,\ \text{unobserved}}

    The head term is computed exactly -- both ``p`` (teacher logit minus stored
    ``logsumexp``) and ``q`` (student log-softmax over the full vocabulary) are
    known on ``S``, and neither side is renormalized. **It is the term this
    function does not have to estimate, and it is the term that dominates.**

    ``T`` is bounded from both sides using quantities we do have:

    * **Lower.** By the log-sum inequality,
      :math:`T \ge P_{\text{tail}}\log(P_{\text{tail}}/Q_{\text{tail}})`, with
      :math:`P_{\text{tail}}` the stored ``tail_mass`` and
      :math:`Q_{\text{tail}} = 1 - \sum_{i\in S} q_i` computed from the student's
      own full-vocabulary logits. Equality holds exactly when the teacher's tail
      *shape* equals the student's tail shape, so this bound is simultaneously
      the plug-in estimate under a proportional-tail model -- which is why it is
      reported as the primary number.
    * **Upper.** Every tail probability is at most
      :math:`\tau = \min_{i\in S} p_i` (otherwise it would have been retained),
      and every :math:`-\log q_i` is at most :math:`-\log q_{\min}` over the
      whole vocabulary, so
      :math:`T \le P_{\text{tail}}\log(\tau / q_{\min})`. This is adversarial: it
      places the entire tail mass on the single token the student disbelieves
      most.

    Bias direction, stated once and repeated in the report: ``kld_lower`` is a
    **rigorous lower bound**, i.e. biased *low*, i.e. flattering to the artifact.
    ``kld_upper`` is a rigorous but loose upper bound. The honest statement about
    a gate is therefore "passes / fails / undetermined", decided by which side of
    the threshold the interval falls on.

    Note also what this makes explicit about the GLM lineage's
    ``teacher_topk_lower_bound``: the head sum alone is *not* a lower bound on
    the KL. It is one whenever the student holds no more tail mass than the
    teacher (:math:`Q_{\text{tail}} \le P_{\text{tail}}`), and it is an
    over-estimate otherwise.
    """

    teacher_probs = np.exp(teacher_topk_logprobs)
    head = np.sum(teacher_probs * (teacher_topk_logprobs - student_topk_logprobs), axis=-1)

    tail_p = np.clip(teacher_tail_mass, 0.0, 1.0)
    tail_q = np.clip(1.0 - student_head_mass, 1e-300, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tail_lower = np.where(
            tail_p > 0.0,
            tail_p * (np.log(np.maximum(tail_p, 1e-300)) - np.log(tail_q)),
            0.0,
        )
        tail_upper = np.where(
            tail_p > 0.0,
            tail_p * (teacher_min_head_logprob - student_min_logprob),
            0.0,
        )
    # The adversarial bound can only ever be an upper bound; if the arithmetic
    # ever lands below the proportional-tail term the interval is degenerate and
    # the wider of the two is the honest one.
    tail_upper = np.maximum(tail_upper, tail_lower)
    return {
        "head": head,
        "tail_lower": tail_lower,
        "tail_upper": tail_upper,
        "kld_lower": head + tail_lower,
        "kld_upper": head + tail_upper,
        "teacher_head_mass": 1.0 - tail_p,
        "student_head_mass": student_head_mass,
    }


# ---------------------------------------------------------------------------
# Artifact-side: span index over the VQ safetensors, and the streaming provider
# ---------------------------------------------------------------------------


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path.name}: truncated safetensors length prefix")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(handle.read(header_length))
    return header, 8 + header_length


@dataclass(frozen=True)
class VqTensorSpan:
    """Where one projection's codes and scales live inside its own file."""

    path: Path
    codes_offset: int
    codes_shape: tuple[int, ...]
    codes_dtype: np.dtype
    scales_offset: int
    scales_shape: tuple[int, ...]
    scales_dtype: np.dtype
    sha256: str | None

    @property
    def codes_bytes(self) -> int:
        return int(np.prod(self.codes_shape)) * self.codes_dtype.itemsize

    @property
    def scales_bytes(self) -> int:
        return int(np.prod(self.scales_shape)) * self.scales_dtype.itemsize


@dataclass(frozen=True)
class VqBlockSpan:
    """One MoE block's three VQ projections, ready to ``preadv``."""

    layer: int
    projections: Mapping[str, VqTensorSpan]
    num_experts: int
    input_dims: int
    hidden_dims: int
    group_size: int
    code_bits: int

    @property
    def tensor_bytes(self) -> int:
        return sum(
            span.codes_bytes + span.scales_bytes for span in self.projections.values()
        )


def _tensor_entry(header: Mapping[str, Any], name: str, path: Path) -> tuple[int, tuple[int, ...], np.dtype]:
    entry = header.get(name)
    if not isinstance(entry, Mapping):
        raise ValueError(f"{path.name}: missing tensor {name!r}")
    dtype_name = str(entry["dtype"]).upper()
    if dtype_name not in _SAFETENSORS_DTYPES:
        raise ValueError(f"{path.name}: unsupported dtype {dtype_name!r} for {name!r}")
    start, end = (int(value) for value in entry["data_offsets"])
    shape = tuple(int(value) for value in entry["shape"])
    dtype = _SAFETENSORS_DTYPES[dtype_name]
    if end - start != int(np.prod(shape)) * dtype.itemsize:
        raise ValueError(f"{path.name}: {name!r} data_offsets disagree with shape/dtype")
    return start, shape, dtype


def build_dsv4_vq_block_span_index(
    artifact_dir: str | Path,
    *,
    layers: Sequence[int],
    manifest: Mapping[str, Any] | None = None,
) -> dict[int, VqBlockSpan]:
    """Parse the artifact's own safetensors headers into read plans.

    Sizes are cross-checked against the manifest's per-file inventory when one
    is available. Re-hashing 75 GB per eval run is not affordable, so this
    verifies size and structure and the run record carries the manifest's own
    per-file SHA-256 list by reference -- see :func:`vq_artifact_identity`.
    """

    root = Path(artifact_dir)
    manifest_sizes: dict[str, int] = {}
    manifest_shas: dict[str, str] = {}
    if manifest is not None:
        for block in manifest.get("blocks", []):
            for entry in block.get("files", []):
                manifest_sizes[str(entry["path"])] = int(entry["bytes"])
                manifest_shas[str(entry["path"])] = str(entry["sha256"])

    spans: dict[int, VqBlockSpan] = {}
    for layer in layers:
        projections: dict[str, VqTensorSpan] = {}
        geometry: dict[str, tuple[int, ...]] = {}
        for projection in VQ_PROJECTIONS:
            filename = f"layer-{int(layer):05d}-{projection}.safetensors"
            path = root / filename
            if not path.is_file():
                raise ValueError(f"missing VQ artifact file {path}")
            actual_size = path.stat().st_size
            expected_size = manifest_sizes.get(filename)
            if expected_size is not None and actual_size != expected_size:
                raise ValueError(
                    f"{filename}: {actual_size} bytes on disk, manifest claims "
                    f"{expected_size}"
                )
            header, data_start = _read_safetensors_header(path)
            prefix = f"model.layers.{int(layer)}.ffn.switch_mlp.{projection}"
            codes_start, codes_shape, codes_dtype = _tensor_entry(
                header, f"{prefix}.codes", path
            )
            scales_start, scales_shape, scales_dtype = _tensor_entry(
                header, f"{prefix}.scales", path
            )
            if len(codes_shape) != 3 or len(scales_shape) != 3:
                raise ValueError(f"{filename}: codes and scales must both be 3D")
            projections[projection] = VqTensorSpan(
                path=path,
                codes_offset=data_start + codes_start,
                codes_shape=codes_shape,
                codes_dtype=codes_dtype,
                scales_offset=data_start + scales_start,
                scales_shape=scales_shape,
                scales_dtype=scales_dtype,
                sha256=manifest_shas.get(filename),
            )
            geometry[projection] = codes_shape

        gate = projections["gate_proj"]
        down = projections["down_proj"]
        num_experts = int(gate.codes_shape[0])
        hidden_dims = int(gate.codes_shape[1])
        input_dims = int(gate.codes_shape[2]) * 8
        code_bits = 16 if gate.codes_dtype == np.dtype(np.uint16) else 8
        group_size = input_dims // int(gate.scales_shape[2])
        if int(down.codes_shape[1]) != input_dims or int(down.codes_shape[2]) * 8 != hidden_dims:
            raise ValueError(
                f"layer {layer}: down_proj geometry {down.codes_shape} does not "
                f"transpose gate_proj {gate.codes_shape}"
            )
        spans[int(layer)] = VqBlockSpan(
            layer=int(layer),
            projections=projections,
            num_experts=num_experts,
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            group_size=group_size,
            code_bits=code_bits,
        )
    return spans


def read_vq_codebook(path: str | Path, *, tensor: str = "model.vq_codebook.e8") -> np.ndarray:
    """Read the artifact's own packed E8 codebook out of one projection file.

    Read directly rather than through ``mx.load``, which would pull the whole
    1.64 GB file to recover 1 KB, or through
    :func:`~keep.io.source_safetensors.read_safetensors_tensor_mlx`, which does
    not carry a ``U32`` case because no source checkpoint has one.
    """

    target = Path(path)
    header, data_start = _read_safetensors_header(target)
    offset, shape, dtype = _tensor_entry(header, tensor, target)
    with target.open("rb") as handle:
        handle.seek(data_start + offset)
        raw = handle.read(int(np.prod(shape)) * dtype.itemsize)
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def vq_artifact_identity(artifact_dir: str | Path) -> dict[str, Any]:
    """Provenance for the artifact under eval, without re-hashing 75 GB.

    ``file_inventory_sha256`` is the SHA-256 of the manifest's own canonicalized
    ``(path, bytes, sha256)`` inventory. It changes if any file's recorded hash
    changes, so quoting it pins the artifact to the tree the materializer
    verified, and the eval separately re-checks every file's size on disk.
    """

    root = Path(artifact_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"no manifest.json under {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("record_type") != VQ_MANIFEST_RECORD_TYPE:
        raise ValueError(
            f"manifest record_type must be {VQ_MANIFEST_RECORD_TYPE!r}, "
            f"found {manifest.get('record_type')!r}"
        )
    inventory = sorted(
        (str(entry["path"]), int(entry["bytes"]), str(entry["sha256"]))
        for block in manifest.get("blocks", [])
        for entry in block.get("files", [])
    )
    return {
        "artifact_dir": str(root),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "file_inventory_sha256": _canonical_sha256(inventory),
        "file_count": len(inventory),
        "artifact_payload_bytes": int(
            manifest.get("audit", {}).get("artifact_payload_bytes", 0)
        ),
        "family": manifest.get("family"),
        "converter": manifest.get("converter"),
        "created_utc": manifest.get("created_utc"),
        "policy": manifest.get("policy"),
        "codebook": manifest.get("codebook"),
        "source": manifest.get("source"),
        "geometry": manifest.get("geometry"),
        "manifest": manifest,
    }


class _VqBlockBuffers:
    """Reusable host destination for one block's codes and scales."""

    def __init__(self, span: VqBlockSpan):
        self.codes = {
            projection: np.empty(
                tensor.codes_shape, dtype=tensor.codes_dtype
            )
            for projection, tensor in span.projections.items()
        }
        self.scales = {
            projection: np.empty(tensor.scales_shape, dtype=tensor.scales_dtype)
            for projection, tensor in span.projections.items()
        }
        self.layer: int | None = None


class Dsv4VqStreamingExpertProvider:
    """Layer-sequential double-buffered reader over the VQ artifact.

    Deliberately the same shape as the teacher run's
    :class:`~keep.quality.dsv4_teacher_runner.ShardStreamingExpertProvider`,
    including its counters, so the eval's I/O line is comparable to the
    capture's. Per block this reads 1.636 GB (three files, two coalesced tensors
    each) against the teacher's 3.42 GB, which is the whole point of the
    artifact.
    """

    def __init__(
        self,
        artifact_dir: str | Path,
        spans: Mapping[int, VqBlockSpan],
        *,
        swiglu_limit: float,
        codebook: Any | None = None,
        io_threads: int = DEFAULT_IO_THREADS,
        nocache: bool = True,
        stats: Dsv4StreamStats | None = None,
    ):
        if not spans:
            raise ValueError("no VQ block spans planned")
        self.artifact_dir = Path(artifact_dir)
        self.spans = dict(spans)
        self.swiglu_limit = float(swiglu_limit)
        self.nocache = bool(nocache)
        self.stats = stats if stats is not None else Dsv4StreamStats()
        self.stats.io_threads = max(1, int(io_threads))
        self._pool = ThreadPoolExecutor(
            max_workers=self.stats.io_threads, thread_name_prefix="dsv4-vq-pread"
        )
        self._fds: dict[Path, int] = {}
        template = next(iter(self.spans.values()))
        self._buffers = [_VqBlockBuffers(template), _VqBlockBuffers(template)]
        self._inflight: dict[int, tuple[float, list[Any]]] = {}
        self._buffer_for: dict[int, _VqBlockBuffers] = {}
        self._resident: dict[int, Any] = {}
        self._codebook = codebook
        self.blocks_taken = 0

    # -- file descriptors ---------------------------------------------------

    def _fd(self, path: Path) -> int:
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            if self.nocache:
                fcntl.fcntl(fd, _F_NOCACHE, 1)
            self._fds[path] = fd
        return fd

    # -- prefetch -----------------------------------------------------------

    def _free_buffer(self) -> _VqBlockBuffers:
        busy = {id(buffer) for buffer in self._buffer_for.values()}
        for buffer in self._buffers:
            if id(buffer) not in busy:
                return buffer
        raise RuntimeError("both VQ block buffers are in flight; prefetch depth is 1")

    def start_prefetch(self, layer: int) -> None:
        if layer in self._inflight or layer in self._resident:
            return
        span = self.spans.get(layer)
        if span is None:
            raise KeyError(f"no VQ block span planned for layer {layer}")
        buffer = self._free_buffer()
        buffer.layer = layer
        self._buffer_for[layer] = buffer
        started = time.perf_counter()
        futures = []
        for projection, tensor in span.projections.items():
            fd = self._fd(tensor.path)
            futures.append(
                self._pool.submit(
                    _pread_exact,
                    fd,
                    buffer.codes[projection],
                    tensor.codes_offset,
                    tensor.codes_bytes,
                )
            )
            futures.append(
                self._pool.submit(
                    _pread_exact,
                    fd,
                    buffer.scales[projection],
                    tensor.scales_offset,
                    tensor.scales_bytes,
                )
            )
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

        from ramp.models.glm4_moe_adapter import QuantizedVQSwitchGLU
        from ramp.nn.switch_linear import QuantizedVQSwitchLinear
        from ramp.models.deepseek_v4_flash_adapter import LimitedSwiGLU

        span = self.spans[layer]
        started = time.perf_counter()
        buffer = self._buffer_for.pop(layer)
        linears: dict[str, Any] = {}
        for projection, tensor in span.projections.items():
            output_dims = int(tensor.codes_shape[1])
            input_dims = int(tensor.codes_shape[2]) * 8
            linears[projection] = QuantizedVQSwitchLinear(
                input_dims=input_dims,
                output_dims=output_dims,
                num_experts=int(tensor.codes_shape[0]),
                codes=mx.array(buffer.codes[projection]),
                scales=mx.array(buffer.scales[projection]),
                codebook=self._codebook,
                group_size=input_dims // int(tensor.scales_shape[2]),
                code_bits=span.code_bits,
            )
        mx.eval([linear.codes for linear in linears.values()])
        mx.eval([linear.scales for linear in linears.values()])
        buffer.layer = None
        glu = QuantizedVQSwitchGLU(
            gate_proj=linears["gate_proj"],
            up_proj=linears["up_proj"],
            down_proj=linears["down_proj"],
            activation=LimitedSwiGLU(self.swiglu_limit),
        )
        self.stats.convert_seconds += time.perf_counter() - started
        self._resident[layer] = glu
        self.blocks_taken += 1
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


def load_dsv4_vq_streaming_workload(
    *,
    checkpoint_dir: str | Path,
    artifact_dir: str | Path,
    io_threads: int = DEFAULT_IO_THREADS,
    nocache: bool = True,
    num_hidden_layers: int | None = None,
    stats: Dsv4StreamStats | None = None,
) -> tuple[Any, ExpertProvider, dict[str, Any]]:
    """Residents from the checkpoint, routed experts from the VQ artifact.

    The residents are bound by exactly the same call the teacher run used --
    ``bind_deepseek_v4_flash_non_vq_weights`` with block-FP8 decode into bf16 --
    so the only difference between this workload and the teacher's is the routed
    experts. That is what makes the resulting delta attributable to the VQ fit.
    """

    import mlx.core as mx

    from mlx.utils import tree_flatten

    from keep.vq.e8 import E8_1BIT_PACKED_SHA256, E8P_PACKED_ABS_SHA256

    from ramp.models.deepseek_v4_flash_adapter import (
        DeepseekV4FlashVQModel,
        bind_deepseek_v4_flash_non_vq_weights,
        deepseek_v4_flash_args_from_config,
    )

    checkpoint = Path(checkpoint_dir)
    identity = vq_artifact_identity(artifact_dir)
    codebook_sha = str((identity["codebook"] or {}).get("sha256") or "")
    if codebook_sha and codebook_sha not in {E8_1BIT_PACKED_SHA256, E8P_PACKED_ABS_SHA256}:
        raise ValueError(
            f"artifact E8 codebook hash {codebook_sha!r} is not supported by this runtime"
        )

    config = json.loads((checkpoint / "config.json").read_text())
    args = deepseek_v4_flash_args_from_config(config)
    if num_hidden_layers is not None:
        from dataclasses import replace

        args = replace(
            args,
            num_hidden_layers=int(num_hidden_layers),
            compress_ratios=list(config["compress_ratios"]),
        )
    model = DeepseekV4FlashVQModel(args, with_mtp=False)

    started = time.perf_counter()
    report = bind_deepseek_v4_flash_non_vq_weights(
        model,
        checkpoint,
        layers=None,
        include_mtp=False,
        strict=False,
        compute_dtype=mx.bfloat16,
    )
    mx.eval(model.parameters())
    resident_seconds = time.perf_counter() - started

    spans = build_dsv4_vq_block_span_index(
        artifact_dir,
        layers=range(args.num_hidden_layers),
        manifest=identity["manifest"],
    )
    codebook_path = spans[0].projections["gate_proj"].path
    codebook_values = read_vq_codebook(codebook_path)
    codebook_measured_sha = _sha256_u32_bytes(codebook_values)
    if codebook_sha and codebook_measured_sha != codebook_sha:
        raise ValueError(
            f"{codebook_path.name} codebook hashes to {codebook_measured_sha!r} but "
            f"the manifest claims {codebook_sha!r}"
        )
    codebook = mx.array(codebook_values)
    mx.eval(codebook)

    provider = Dsv4VqStreamingExpertProvider(
        artifact_dir,
        spans,
        swiglu_limit=args.swiglu_limit,
        codebook=codebook,
        io_threads=io_threads,
        nocache=nocache,
        stats=stats,
    )
    template = spans[0]
    identities = {
        "checkpoint_dir": str(checkpoint),
        "config_sha256": hashlib.sha256(
            (checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        "num_hidden_layers": int(args.num_hidden_layers),
        "n_routed_experts": int(args.n_routed_experts),
        "num_experts_per_tok": int(args.num_experts_per_tok),
        "hidden_size": int(args.hidden_size),
        "moe_intermediate_size": int(args.moe_intermediate_size),
        "swiglu_limit": float(args.swiglu_limit),
        "compress_ratios": list(args.compress_ratios),
        "resident_bind_seconds": round(resident_seconds, 2),
        "with_mtp": False,
        "resident_bound": int(report.bound_count),
        "resident_fp8_decoded": len(report.fp8_block_decoded_tensors),
        "resident_gb": round(
            sum(value.nbytes for _name, value in tree_flatten(model.parameters())) / 1e9,
            3,
        ),
        "expert_bytes_per_layer": int(template.tensor_bytes),
        "expert_span_bytes_per_layer": int(template.tensor_bytes),
        "expert_mode": "vq_e8p_g512",
        "vq_code_bits": int(template.code_bits),
        "vq_group_size": int(template.group_size),
        "vq_artifact": {
            key: value for key, value in identity.items() if key != "manifest"
        },
    }
    return model, provider, identities


# ---------------------------------------------------------------------------
# Student-side logits at the supervised positions
# ---------------------------------------------------------------------------


def _student_logit_slices(
    model: Any,
    hidden: Sequence[Any],
    positions: np.ndarray,
    *,
    chunk: int,
    position_slice: int,
) -> Iterator[tuple[int, Any]]:
    """Yield ``(start, logits[slice, vocab])`` over the supervised positions.

    The gather is the teacher capture's own
    :func:`~keep.quality.dsv4_teacher_runner._supervised_logit_capture`
    arithmetic: ``hc_head`` and ``norm`` are strictly per-position, so gathering
    the supervised rows and projecting them is equal to projecting everything
    and slicing after -- and it avoids a ``[T, 129280]`` tensor that does not
    fit. The difference is only that this yields the dense logits instead of
    reducing them to a top-K on device, because the *student* side of a KLD
    needs the full normalizer and the probability of the teacher's support.
    """

    import mlx.core as mx

    chunk_of = positions // chunk
    within = positions % chunk
    gathered = []
    for chunk_index in range(len(hidden)):
        rows_here = within[chunk_of == chunk_index]
        if rows_here.size == 0:
            continue
        gathered.append(
            mx.take(hidden[chunk_index][0], mx.array(rows_here.astype(np.int32)), axis=0)
        )
    rows = mx.concatenate(gathered, axis=0)
    mx.eval(rows)
    if rows.shape[0] != positions.size:
        raise ValueError(
            f"gathered {rows.shape[0]} supervised rows for {positions.size} positions"
        )
    for start in range(0, rows.shape[0], position_slice):
        piece = rows[start : start + position_slice]
        collapsed = model.model.norm(model.model.hc_head(piece[None, ...]))
        logits = model.lm_head(collapsed)[0]
        mx.eval(logits)
        yield start, logits
        del logits, collapsed, piece
        mx.clear_cache()


def _student_compact_summary(
    logits: Any, teacher_topk_ids: np.ndarray, *, top_k: int
) -> dict[str, np.ndarray]:
    """On-device reductions the tail correction and top-K agreement need.

    Everything here is a reduction of a ``[positions, 129280]`` tensor down to
    ``[positions, K]`` or ``[positions]``, so it runs on the GPU in float32 and
    the host never sees a second dense copy. The dense float64 copy the house
    evaluator makes is unavoidable and is the reason position slices are small.
    """

    import mlx.core as mx

    logits32 = logits.astype(mx.float32)
    logsumexp = mx.logsumexp(logits32, axis=1)
    log_probs_at_support = mx.take_along_axis(
        logits32, mx.array(teacher_topk_ids.astype(np.int32)), axis=1
    ) - logsumexp[:, None]
    head_mass = mx.exp(log_probs_at_support).sum(axis=1)
    min_logprob = mx.min(logits32, axis=1) - logsumexp
    student_top_ids = mx.argsort(-logits32, axis=1)[:, :top_k]
    mx.eval(log_probs_at_support, head_mass, min_logprob, student_top_ids)
    return {
        "log_probs_at_support": np.asarray(log_probs_at_support, dtype=np.float64),
        "head_mass": np.asarray(head_mass, dtype=np.float64),
        "min_logprob": np.asarray(min_logprob, dtype=np.float64),
        "student_top_ids": np.asarray(student_top_ids, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# One session's row
# ---------------------------------------------------------------------------


def _teacher_row_for_slice(
    teacher: Dsv4TeacherSession,
    *,
    censor_logprob: np.ndarray,
) -> dict[str, Any]:
    """A GLM-shaped teacher-cache row over one position slice.

    ``target_logprobs`` is the one field the truncated capture cannot always
    supply: when the supervised target is outside the retained top-K the
    teacher's probability for it is censored. The substituted value is the
    smallest retained log-probability, which is a rigorous **upper** bound on the
    censored value -- so the teacher NLL computed from it is a lower bound, and
    ``nll_delta = vq_nll - teacher_nll`` is an upper bound. The count of censored
    positions rides along on the row so the effect is auditable rather than
    buried.
    """

    topk_logprobs = teacher.topk_logprobs()
    hits = teacher.topk_ids == teacher.target_token_ids[:, None]
    found = np.any(hits, axis=1)
    first = np.argmax(hits, axis=1)
    target_logprobs = np.where(
        found,
        np.take_along_axis(topk_logprobs, first[:, None], axis=1)[:, 0],
        censor_logprob,
    )
    return {
        "prompt_id": teacher.prompt_id,
        "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "teacher_kind": "dsv4_local_source_topk",
        "positions": [int(value) for value in teacher.positions.tolist()],
        "target_token_ids": [int(value) for value in teacher.target_token_ids.tolist()],
        "target_logprobs": target_logprobs,
        "teacher_top1_ids": teacher.topk_ids[:, 0],
        "topk_ids": teacher.topk_ids,
        "topk_logprobs": topk_logprobs,
        "full_logits_available": False,
        "_target_in_topk": found,
    }


@dataclass
class _SessionAccumulator:
    """Position-exact accumulation of one session's metrics across slices."""

    teacher: Dsv4TeacherSession
    top_ks: tuple[int, ...] = DEFAULT_TOP_KS

    def __post_init__(self) -> None:
        self.positions_done = 0
        self.vq_target_logprobs: list[np.ndarray] = []
        self.teacher_target_logprobs: list[np.ndarray] = []
        self.target_in_topk: list[np.ndarray] = []
        self.kld_head: list[np.ndarray] = []
        self.kld_lower: list[np.ndarray] = []
        self.kld_upper: list[np.ndarray] = []
        self.tail_p: list[np.ndarray] = []
        self.tail_q: list[np.ndarray] = []
        self.top1_hit: list[np.ndarray] = []
        self.topk_hit: dict[int, list[np.ndarray]] = {k: [] for k in self.top_ks}
        self.slice_records: list[dict[str, Any]] = []
        self.head_crosscheck_drift = 0.0

    def add(self, start: int, logits: Any) -> None:
        count = int(logits.shape[0])
        stop = start + count
        teacher_slice = self.teacher.slice(start, stop)
        topk_logprobs = teacher_slice.topk_logprobs()
        censor = topk_logprobs.min(axis=1)
        row = _teacher_row_for_slice(teacher_slice, censor_logprob=censor)
        target_in_topk = row.pop("_target_in_topk")

        # The house evaluator, verbatim: NLL, perplexity, ppl ratio, top-1
        # agreement, the exact KLD head sum and its token-level tail.
        record = evaluate_teacher_cache_row(
            row,
            vq_logits=logits,
            cache_root=self.teacher.path.parent,
            extra={"slice_start": start},
        )
        compact = _student_compact_summary(
            logits, teacher_slice.topk_ids, top_k=max(self.top_ks)
        )
        tail = tail_corrected_kld(
            teacher_topk_logprobs=topk_logprobs,
            student_topk_logprobs=compact["log_probs_at_support"],
            teacher_tail_mass=teacher_slice.tail_mass,
            student_head_mass=compact["head_mass"],
            teacher_min_head_logprob=censor,
            student_min_logprob=compact["min_logprob"],
        )
        # The house evaluator's head sum and ours must agree. They are computed
        # from the same float32 student logits by two independent paths (numpy
        # float64 log-softmax versus an MLX float32 logsumexp), so they differ
        # only by float32 accumulation over 129,280 terms. A real gather or
        # normalizer bug moves this by O(0.1), not O(1e-5), so the tolerance is
        # loose enough to be quiet and tight enough to catch that.
        house_head = np.asarray(record["token_klds"], dtype=np.float64)
        drift = float(np.max(np.abs(house_head - tail["head"]))) if count else 0.0
        if not math.isfinite(drift) or drift > 1e-3:
            raise ValueError(
                f"{self.teacher.prompt_id}: KLD head sum disagrees between the "
                f"house evaluator and the tail estimator by {drift:.3e}"
            )
        self.head_crosscheck_drift = max(self.head_crosscheck_drift, drift)

        vq_top1 = np.asarray(record["vq_top1_ids"], dtype=np.int64)
        teacher_top1 = teacher_slice.topk_ids[:, 0]
        student_topk = compact["student_top_ids"]
        for k in self.top_ks:
            self.topk_hit[k].append(
                np.any(student_topk[:, :k] == teacher_top1[:, None], axis=1)
            )
        self.top1_hit.append(vq_top1 == teacher_top1)
        self.vq_target_logprobs.append(
            np.asarray(record["vq_target_logprobs"], dtype=np.float64)
        )
        self.teacher_target_logprobs.append(
            np.asarray(record["teacher_target_logprobs"], dtype=np.float64)
        )
        self.target_in_topk.append(target_in_topk)
        self.kld_head.append(tail["head"])
        self.kld_lower.append(tail["kld_lower"])
        self.kld_upper.append(tail["kld_upper"])
        self.tail_p.append(np.clip(teacher_slice.tail_mass, 0.0, 1.0))
        self.tail_q.append(1.0 - compact["head_mass"])
        self.positions_done += count
        self.slice_records.append(
            {
                "slice_start": start,
                "positions": count,
                "mean_kld_head": float(np.mean(tail["head"])),
                "top1_agreement": record["top1_agreement"],
            }
        )

    def token_arrays(self) -> dict[str, np.ndarray]:
        return {
            "positions": self.teacher.positions.astype(np.int32),
            "target_token_ids": self.teacher.target_token_ids.astype(np.int32),
            "kld_head": np.concatenate(self.kld_head).astype(np.float32),
            "kld_lower": np.concatenate(self.kld_lower).astype(np.float32),
            "kld_upper": np.concatenate(self.kld_upper).astype(np.float32),
            "vq_target_logprobs": np.concatenate(self.vq_target_logprobs).astype(np.float32),
            "teacher_target_logprobs": np.concatenate(self.teacher_target_logprobs).astype(
                np.float32
            ),
            "top1_hit": np.concatenate(self.top1_hit).astype(np.uint8),
            "teacher_tail_mass": np.concatenate(self.tail_p).astype(np.float32),
            "student_tail_mass": np.concatenate(self.tail_q).astype(np.float32),
        }

    def row(self, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.positions_done != self.teacher.supervised_count:
            raise ValueError(
                f"{self.teacher.prompt_id}: accumulated {self.positions_done} of "
                f"{self.teacher.supervised_count} supervised positions"
            )
        vq_logprobs = np.concatenate(self.vq_target_logprobs)
        teacher_logprobs = np.concatenate(self.teacher_target_logprobs)
        in_topk = np.concatenate(self.target_in_topk)
        head = np.concatenate(self.kld_head)
        lower = np.concatenate(self.kld_lower)
        upper = np.concatenate(self.kld_upper)
        tail_p = np.concatenate(self.tail_p)
        tail_q = np.concatenate(self.tail_q)
        top1 = np.concatenate(self.top1_hit)

        vq_nll = float(-np.mean(vq_logprobs))
        teacher_nll = float(-np.mean(teacher_logprobs))
        nll_delta = vq_nll - teacher_nll
        supported = bool(np.any(in_topk))
        row: dict[str, Any] = {
            "record_type": AGREEMENT_ROW_RECORD_TYPE,
            "schema_version": 1,
            "prompt_id": self.teacher.prompt_id,
            "split": self.teacher.campaign_split,
            "teacher_path": str(self.teacher.path),
            "teacher_generation_config_sha256": self.teacher.generation_config_sha256,
            "teacher_token_ids_sha256": self.teacher.token_ids_sha256,
            "teacher_prefill_chunk_tokens": self.teacher.prefill_chunk_tokens,
            "teacher_top_k": self.teacher.top_k,
            "token_count": self.teacher.token_count,
            "supervised_positions": int(self.positions_done),
            # NLL / perplexity
            "vq_nll": vq_nll,
            "teacher_nll": teacher_nll,
            "nll_delta": nll_delta,
            "vq_perplexity": float(math.exp(vq_nll)) if vq_nll < 700 else None,
            "teacher_perplexity": float(math.exp(teacher_nll)) if teacher_nll < 700 else None,
            "ppl_ratio": float(math.exp(nll_delta)) if abs(nll_delta) < 700 else None,
            "vq_nll_in_teacher_support": (
                float(-np.mean(vq_logprobs[in_topk])) if supported else None
            ),
            "teacher_nll_in_teacher_support": (
                float(-np.mean(teacher_logprobs[in_topk])) if supported else None
            ),
            "target_in_teacher_topk_fraction": float(np.mean(in_topk)),
            "teacher_nll_censored_positions": int(np.count_nonzero(~in_topk)),
            # KLD
            "kld_estimator": "head_exact_plus_proportional_tail",
            "kld_mode": "dsv4_topk_logsumexp_tail_bounded",
            "mean_kld": float(np.mean(lower)),
            "mean_kld_head": float(np.mean(head)),
            "mean_kld_lower": float(np.mean(lower)),
            "mean_kld_upper": float(np.mean(upper)),
            "p999_kld": float(np.quantile(lower, 0.999)),
            "p999_kld_head": float(np.quantile(head, 0.999)),
            "p999_kld_upper": float(np.quantile(upper, 0.999)),
            "max_kld": float(np.max(lower)),
            "mean_teacher_tail_mass": float(np.mean(tail_p)),
            "max_teacher_tail_mass": float(np.max(tail_p)),
            "mean_student_tail_mass": float(np.mean(tail_q)),
            "teacher_topk_probability_mass": float(np.mean(1.0 - tail_p)),
            # agreement
            "top1_agreement": float(np.mean(top1)),
            "topk_agreement_definition": (
                "fraction of supervised positions whose teacher top-1 token is "
                "inside the student's top-k"
            ),
        }
        for k in self.top_ks:
            row[f"top{k}_agreement"] = float(np.mean(np.concatenate(self.topk_hit[k])))
        row["slice_count"] = len(self.slice_records)
        row["kld_head_crosscheck_max_abs_drift"] = self.head_crosscheck_drift
        if extra:
            row.update(extra)
        json.dumps(row, sort_keys=True)
        return row


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_agreement_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if row.get("record_type") != AGREEMENT_ROW_RECORD_TYPE:
            raise ValueError(
                f"{path}: row record_type must be {AGREEMENT_ROW_RECORD_TYPE!r}, "
                f"found {row.get('record_type')!r}"
            )
        rows.append(row)
    return rows


class AgreementHeartbeat:
    """Atomic ``status.json`` plus append-only ``status.log``.

    Same shape as the teacher runner's and the materializer's heartbeat so one
    ``monitor`` habit covers all three.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        total_sessions: int,
        total_tokens: int,
        total_positions: int,
        extra: Mapping[str, Any] | None = None,
    ):
        self.path = Path(path)
        self.log_path = self.path.with_suffix(".log")
        self.started = time.time()
        self.total_sessions = int(total_sessions)
        self.total_tokens = int(total_tokens)
        self.total_positions = int(total_positions)
        self.extra = dict(extra or {})
        self.sessions_done = 0
        self.sessions_skipped = 0
        self.tokens_done = 0
        self.positions_done = 0
        self.compute_seconds = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _eta(self) -> tuple[float | None, str]:
        remaining = self.total_tokens - self.tokens_done
        if self.tokens_done <= 0 or self.compute_seconds <= 0:
            return None, "no completed session yet"
        rate = self.tokens_done / self.compute_seconds
        return (
            remaining / rate,
            f"{self.tokens_done} tokens at {rate:.1f} tok/s over "
            f"{self.sessions_done} session(s)",
        )

    def payload(self, **fields: Any) -> dict[str, Any]:
        eta, basis = self._eta()
        return {
            "record_type": AGREEMENT_STATUS_RECORD_TYPE,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            "elapsed_s": round(time.time() - self.started, 1),
            "sessions_done": self.sessions_done,
            "sessions_skipped": self.sessions_skipped,
            "total_sessions": self.total_sessions,
            "tokens_done": self.tokens_done,
            "total_tokens": self.total_tokens,
            "positions_done": self.positions_done,
            "total_positions": self.total_positions,
            "tokens_per_s": round(self.tokens_done / self.compute_seconds, 1)
            if self.compute_seconds > 0
            else None,
            "eta_s": round(eta, 0) if eta else None,
            "eta_h": round(eta / 3600.0, 2) if eta else None,
            "eta_basis": basis,
            **self.extra,
            **fields,
        }

    def update(self, **fields: Any) -> dict[str, Any]:
        payload = self.payload(**fields)
        _atomic_json(self.path, payload)
        return payload

    def log(self, **fields: Any) -> dict[str, Any]:
        payload = self.update(**fields)
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload


def run_dsv4_teacher_agreement(
    *,
    teacher_dir: str | Path,
    pack_path: str | Path,
    out_dir: str | Path,
    artifact_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    engine: str = "vq_e8p_streamed",
    split: str | None = "report",
    prompt_ids: Sequence[str] | None = None,
    stratify: int | None = None,
    max_sessions: int | None = None,
    chunk: int = DEFAULT_PREFILL_CHUNK_TOKENS,
    position_slice: int = DEFAULT_POSITION_SLICE,
    io_threads: int = DEFAULT_IO_THREADS,
    nocache: bool = True,
    append_jsonl: str | Path | None = None,
    status_path: str | Path | None = None,
    save_token_arrays: bool = True,
    workload_loader: Callable[..., tuple[Any, ExpertProvider, dict[str, Any]]] | None = None,
    teacher_sessions: Sequence[Dsv4TeacherSession] | None = None,
    pack_sessions: Sequence[Dsv4Session] | None = None,
    stop_file: str | Path | None = None,
) -> dict[str, Any]:
    """Measure teacher agreement for one engine over the selected sessions.

    ``engine`` is ``vq_e8p_streamed`` (the artifact) or ``source_mxfp4_streamed``
    (the control: the unmodified source experts, streamed by the teacher run's
    own loader). The control exists because "KLD 0.2" is not interpretable on its
    own -- the source-versus-teacher number is not exactly zero either, and only
    measuring it says how much of the artifact's number is the artifact.
    """

    import mlx.core as mx

    if engine not in ("vq_e8p_streamed", "source_mxfp4_streamed"):
        raise ValueError(
            "engine must be 'vq_e8p_streamed' or 'source_mxfp4_streamed', "
            f"found {engine!r}"
        )
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    if position_slice <= 0:
        raise ValueError("position_slice must be positive")
    if engine == "vq_e8p_streamed" and artifact_dir is None:
        raise ValueError("engine 'vq_e8p_streamed' needs an artifact_dir")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows_path = Path(append_jsonl) if append_jsonl is not None else out_root / "rows.jsonl"
    stop = Path(stop_file) if stop_file is not None else None
    token_array_dir = out_root / "token-arrays"

    if teacher_sessions is None:
        teacher_sessions = select_teacher_sessions(
            teacher_dir,
            split=split,
            prompt_ids=prompt_ids,
            stratify=stratify,
            max_sessions=max_sessions,
        )
    selected = tuple(teacher_sessions)

    # Prefill chunk parity is a refusal, not a note: a student forwarded at a
    # different chunk size is a different computation from the teacher and the
    # resulting KLD is not a measurement of the artifact.
    mismatched = sorted(
        {int(session.prefill_chunk_tokens) for session in selected} - {int(chunk)}
    )
    if mismatched:
        raise ValueError(
            "prefill chunk mismatch: this run uses "
            f"{chunk} but the teacher capture used {mismatched}. The numbers "
            "would not be comparable; re-run with --chunk "
            f"{mismatched[0]} or re-capture the teacher."
        )
    teacher_config_shas = sorted(
        {session.generation_config_sha256 for session in selected}
    )
    if len(teacher_config_shas) != 1:
        raise ValueError(
            f"selected sessions span {len(teacher_config_shas)} teacher generation "
            f"configs: {teacher_config_shas}"
        )

    if pack_sessions is None:
        pack_sessions = load_dsv4_teich_pack(
            pack_path, prompt_ids=[session.prompt_id for session in selected]
        )
    pack_by_id = {session.prompt_id: session for session in pack_sessions}

    for teacher in selected:
        source = pack_by_id.get(teacher.prompt_id)
        if source is None:
            raise ValueError(f"teich pack has no session {teacher.prompt_id}")
        if source.token_ids_sha256 != teacher.token_ids_sha256:
            raise ValueError(
                f"{teacher.prompt_id}: teich pack token_ids_sha256 does not match "
                "the teacher capture"
            )
        if source.token_count != teacher.token_count:
            raise ValueError(f"{teacher.prompt_id}: token_count mismatch")
        if not np.array_equal(
            source.positions.astype(np.int64), teacher.positions.astype(np.int64)
        ):
            raise ValueError(f"{teacher.prompt_id}: supervised positions mismatch")
        if not np.array_equal(
            source.target_token_ids.astype(np.int64),
            teacher.target_token_ids.astype(np.int64),
        ):
            raise ValueError(f"{teacher.prompt_id}: supervised targets mismatch")

    done_ids: set[str] = set()
    if rows_path.is_file():
        for row in read_agreement_rows(rows_path):
            if row.get("engine") == engine:
                done_ids.add(str(row["prompt_id"]))
    pending = [session for session in selected if session.prompt_id not in done_ids]

    stats = Dsv4StreamStats()
    mx.reset_peak_memory()
    trace_vm_start = collect_vm_stat_counts()

    def _default_loader(**kwargs: Any) -> tuple[Any, ExpertProvider, dict[str, Any]]:
        if engine == "vq_e8p_streamed":
            return load_dsv4_vq_streaming_workload(artifact_dir=artifact_dir, **kwargs)
        return load_dsv4_streaming_workload(with_mtp=False, **kwargs)

    loader = workload_loader if workload_loader is not None else _default_loader
    model, provider, identities = loader(
        checkpoint_dir=checkpoint_dir,
        io_threads=io_threads,
        nocache=nocache,
        stats=stats,
    )

    heartbeat = AgreementHeartbeat(
        status_path or (out_root / "status.json"),
        total_sessions=len(selected),
        total_tokens=sum(session.token_count for session in pending),
        total_positions=sum(session.supervised_count for session in pending),
        extra={
            "engine": engine,
            "out_dir": str(out_root),
            "rows_jsonl": str(rows_path),
            "prefill_chunk_tokens": int(chunk),
            "teacher_generation_config_sha256": teacher_config_shas[0],
            "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
        },
    )
    heartbeat.sessions_skipped = len(selected) - len(pending)
    heartbeat.update(phase="planned", pending_sessions=len(pending))

    run_identity = {
        "record_type": "dsv4_teacher_agreement_run_v1",
        "schema_version": 1,
        "engine": engine,
        "split": split,
        "prefill_chunk_tokens": int(chunk),
        "position_slice": int(position_slice),
        "teacher_dir": str(teacher_dir),
        "teacher_generation_config_sha256": teacher_config_shas[0],
        "pack_path": str(pack_path),
        "identities": identities,
        "selected_sessions": [
            {
                "prompt_id": session.prompt_id,
                "split": session.campaign_split,
                "token_count": session.token_count,
                "supervised": session.supervised_count,
                "teacher_top_k": session.top_k,
            }
            for session in selected
        ],
    }
    run_identity["run_config_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in run_identity.items()
            if key in ("engine", "prefill_chunk_tokens", "position_slice", "identities")
        }
    )
    _atomic_json(out_root / f"run-{engine}.json", run_identity)

    records: list[dict[str, Any]] = []
    bind_proof: dict[str, Any] | None = None
    try:
        bind_proof = _probe_bind_proof(model, provider, engine)
        _atomic_json(out_root / f"bind-proof-{engine}.json", bind_proof)
        for index, teacher in enumerate(pending):
            if stop is not None and stop.exists():
                heartbeat.log(phase="stopped", stop_file=str(stop))
                break
            source = pack_by_id[teacher.prompt_id]
            before_vm = collect_vm_stat_counts()
            reset_mlx_peak_memory()
            heartbeat.update(
                phase="forward",
                prompt_id=teacher.prompt_id,
                session_index=index,
                session_tokens=teacher.token_count,
                session_positions=teacher.supervised_count,
            )
            started = time.perf_counter()
            token_ids = mx.array(source.token_ids.astype(np.int32)[None, :])
            reads_before = int(stats.layer_reads)
            hidden = layer_major_prefill(
                model,
                token_ids,
                chunk=chunk,
                expert_provider=provider,
            )
            forward_seconds = time.perf_counter() - started
            layers_streamed = int(stats.layer_reads) - reads_before

            accumulator = _SessionAccumulator(teacher)
            metric_started = time.perf_counter()
            for start, logits in _student_logit_slices(
                model,
                hidden,
                teacher.positions,
                chunk=chunk,
                position_slice=position_slice,
            ):
                accumulator.add(start, logits)
            metric_seconds = time.perf_counter() - metric_started
            del hidden
            mx.clear_cache()

            memory = collect_metric_snapshot(previous_vm_stat_counts=before_vm)
            token_arrays_name = None
            if save_token_arrays:
                token_array_dir.mkdir(parents=True, exist_ok=True)
                token_arrays_name = f"{engine}-{teacher.prompt_id}.npz"
                np.savez(
                    token_array_dir / token_arrays_name, **accumulator.token_arrays()
                )
            row = accumulator.row(
                extra={
                    "engine": engine,
                    "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
                    "artifact_file_inventory_sha256": (
                        identities.get("vq_artifact", {}).get("file_inventory_sha256")
                    ),
                    "artifact_manifest_sha256": (
                        identities.get("vq_artifact", {}).get("manifest_sha256")
                    ),
                    "checkpoint_config_sha256": identities.get("config_sha256"),
                    "expert_mode": identities.get("expert_mode"),
                    "prefill_chunk_tokens": int(chunk),
                    "position_slice": int(position_slice),
                    "run_config_sha256": run_identity["run_config_sha256"],
                    "forward_seconds": round(forward_seconds, 2),
                    "metric_seconds": round(metric_seconds, 2),
                    "tokens_per_s": round(teacher.token_count / forward_seconds, 1),
                    "expert_layer_reads": layers_streamed,
                    "token_arrays_npz": token_arrays_name,
                    "bind_proof": bind_proof,
                    "stream": stats.as_dict(),
                    **memory,
                    "memory_clean": (
                        memory.get("pageouts_delta") == 0
                        and memory.get("swapouts_delta") == 0
                    ),
                }
            )
            records.append(row)
            _append_jsonl(rows_path, row)
            heartbeat.sessions_done += 1
            heartbeat.tokens_done += teacher.token_count
            heartbeat.positions_done += teacher.supervised_count
            heartbeat.compute_seconds += forward_seconds + metric_seconds
            heartbeat.log(
                phase="session_done",
                prompt_id=teacher.prompt_id,
                mean_kld=row["mean_kld"],
                mean_kld_head=row["mean_kld_head"],
                top1_agreement=row["top1_agreement"],
                vq_nll=row["vq_nll"],
                memory_clean=row["memory_clean"],
            )
    finally:
        provider.close()

    all_rows = [
        row for row in read_agreement_rows(rows_path) if row.get("engine") == engine
    ]
    summary = summarize_agreement_records(
        all_rows, token_array_dir=token_array_dir if save_token_arrays else None
    )
    summary.update(
        {
            "engine": engine,
            "split": split,
            "rows_jsonl": str(rows_path),
            "out_dir": str(out_root),
            "prefill_chunk_tokens": int(chunk),
            "teacher_generation_config_sha256": teacher_config_shas[0],
            "run_config_sha256": run_identity["run_config_sha256"],
            "sessions_selected": len(selected),
            "sessions_produced": len(records),
            "sessions_skipped": len(selected) - len(pending),
            "peak_mlx_gb": round(mx.get_peak_memory() / 1e9, 2),
            "stream": stats.as_dict(),
            "identities": identities,
            "bind_proof": bind_proof,
            "process_memory": collect_metric_snapshot(
                previous_vm_stat_counts=trace_vm_start
            ),
        }
    )
    _atomic_json(out_root / f"summary-{engine}.json", summary)
    heartbeat.log(phase="complete", summary_path=str(out_root / f"summary-{engine}.json"))
    return summary


def _probe_bind_proof(model: Any, provider: ExpertProvider, engine: str) -> dict[str, Any]:
    """Answer the audit predicates from a genuinely bound layer.

    Under layer-sequential streaming the whole-model
    ``has_unbound_deepseek_v4_flash_vq_experts`` predicate is vacuously ``True``
    -- 42 of 43 layers are deliberately unbound at any instant -- so quoting it
    would be dishonest and this record says so instead of pretending otherwise.
    What *is* checkable, and is checked here, is that while a layer is bound no
    routed-expert parameter is stored densely anywhere in the model, and that
    the bound module is the quantized VQ type carrying the ``LimitedSwiGLU``
    clamp rather than a silently dequantized or plain-SwiGLU stand-in.

    The probe takes layer 0 through the provider's own path, reads the
    predicates, then unbinds and releases it, so the session forwards that
    follow start from the same state they would have started from anyway.
    """

    from ramp.models.glm4_moe_adapter import QuantizedVQSwitchGLU
    from ramp.nn.switch_linear import QuantizedVQSwitchLinear
    from ramp.models.deepseek_v4_flash_adapter import (
        dense_deepseek_v4_flash_routed_parameter_names,
    )

    layer = model.model.layers[0]
    bound = provider.take(0)
    layer.ffn.switch_mlp = bound
    try:
        dense = dense_deepseek_v4_flash_routed_parameter_names(model)
        proof: dict[str, Any] = {
            "record_type": "dsv4_teacher_agreement_bind_proof_v1",
            "checked_at": "layer_0_bound_probe_before_first_forward",
            "engine": engine,
            "bound_module": type(bound).__name__,
            "dense_routed_parameter_names": list(dense),
            "dense_routed_experts": bool(dense),
            "activation": type(getattr(bound, "activation", None)).__name__,
            "whole_model_unbound_check": (
                "not applicable under layer-sequential streaming; 42 of 43 layers "
                "are deliberately unbound at any instant"
            ),
        }
        if engine == "vq_e8p_streamed":
            gate = getattr(bound, "gate_proj", None)
            proof.update(
                {
                    "is_quantized_vq_switch_glu": isinstance(bound, QuantizedVQSwitchGLU),
                    "is_quantized_vq_switch_linear": isinstance(
                        gate, QuantizedVQSwitchLinear
                    ),
                    "route_backend": getattr(gate, "route_backend", None),
                    "code_bits": getattr(gate, "code_bits", None),
                    "group_size": getattr(gate, "group_size", None),
                    "num_experts": getattr(gate, "num_experts", None),
                    "file_discovery": (
                        "every backbone block's three artifact files were opened "
                        "and header-parsed by build_dsv4_vq_block_span_index, and "
                        "each file's size was re-checked against the manifest"
                    ),
                }
            )
    finally:
        layer.ffn.switch_mlp = None
        provider.release(0)
    return proof


# ---------------------------------------------------------------------------
# Summary and verdict
# ---------------------------------------------------------------------------


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if array.size == 0 or weight.sum() <= 0:
        return None
    return float(np.sum(array * weight) / np.sum(weight))


def _pooled_token_kld(
    rows: Sequence[Mapping[str, Any]], token_array_dir: str | Path | None
) -> dict[str, Any]:
    """Pool the per-token KLD across rows for a true p99.9.

    The house summarizer pools ``token_klds`` carried inline on each row. These
    rows deliberately do not carry them -- 316,617 supervised positions inline
    would be ~90 MB of JSONL -- so the arrays live in per-row ``.npz`` sidecars
    and the pooled tail is recomputed from those. When the sidecars are absent
    the pooled figure is reported as ``None`` rather than silently degraded to a
    mean of per-row tails, which is a different statistic.
    """

    if token_array_dir is None:
        return {"p999_kld_pooled": None, "p999_kld_source": "unavailable"}
    root = Path(token_array_dir)
    pieces: list[np.ndarray] = []
    for row in rows:
        name = row.get("token_arrays_npz")
        if not name:
            return {"p999_kld_pooled": None, "p999_kld_source": "row_missing_sidecar"}
        path = root / str(name)
        if not path.is_file():
            return {"p999_kld_pooled": None, "p999_kld_source": f"missing {name}"}
        with np.load(path) as payload:
            pieces.append(np.asarray(payload["kld_lower"], dtype=np.float64))
    pooled = np.concatenate(pieces)
    return {
        "p999_kld_pooled": float(np.quantile(pooled, 0.999)),
        "p99_kld_pooled": float(np.quantile(pooled, 0.99)),
        "max_token_kld_pooled": float(np.max(pooled)),
        "pooled_token_count": int(pooled.size),
        "p999_kld_source": "token-array sidecars",
    }


def summarize_agreement_records(
    records: Sequence[Mapping[str, Any]],
    *,
    token_array_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute the split summary from the JSONL rows.

    The per-row means come from the house summarizer so the row-level
    distributions match the GLM lineage exactly. The **position-weighted**
    aggregates are added on top and are the headline for this family: report
    sessions run from 1,298 to 27,060 supervised positions, so an unweighted mean
    over sessions is a mean over sessions, not over tokens, and the two differ.
    """

    if not records:
        raise ValueError("cannot summarize an empty agreement result set")
    rows = [dict(record) for record in records]
    house = summarize_teacher_cache_records(rows)
    weights = [float(row["supervised_positions"]) for row in rows]
    clean = [row for row in rows if row.get("memory_clean")]
    top_k_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("top") and key.endswith("_agreement")
        }
    )
    weighted: dict[str, float | None] = {}
    for field in (
        "vq_nll",
        "teacher_nll",
        "nll_delta",
        "mean_kld",
        "mean_kld_head",
        "mean_kld_lower",
        "mean_kld_upper",
        "mean_teacher_tail_mass",
        "mean_student_tail_mass",
        "target_in_teacher_topk_fraction",
        *top_k_fields,
    ):
        present = [(row[field], weight) for row, weight in zip(rows, weights) if row.get(field) is not None]
        weighted[field] = _weighted_mean(
            [value for value, _ in present], [weight for _, weight in present]
        )
    weighted_nll_delta = weighted.get("nll_delta")
    return {
        "record_type": "dsv4_teacher_agreement_summary_v1",
        "schema_version": 1,
        "row_count": len(rows),
        "clean_row_count": len(clean),
        "all_memory_clean": len(clean) == len(rows),
        "total_supervised_positions": int(sum(weights)),
        "total_tokens": int(sum(int(row["token_count"]) for row in rows)),
        "kld_estimators": sorted({str(row.get("kld_estimator")) for row in rows}),
        "prefill_chunk_tokens": sorted({int(row["prefill_chunk_tokens"]) for row in rows}),
        "engines": sorted({str(row.get("engine")) for row in rows}),
        "position_weighted": {
            **weighted,
            "ppl_ratio": (
                float(math.exp(weighted_nll_delta))
                if weighted_nll_delta is not None and abs(weighted_nll_delta) < 700
                else None
            ),
        },
        "per_row_mean": {
            "mean_kld": house.get("mean_kld"),
            "mean_top1_agreement": house.get("mean_top1_agreement"),
            "mean_nll_delta": house.get("mean_nll_delta"),
            "max_nll_delta": house.get("max_nll_delta"),
            "mean_ppl_ratio": house.get("mean_ppl_ratio"),
            "max_ppl_ratio": house.get("max_ppl_ratio"),
        },
        "token_tail": {
            "max_row_p999_kld": max(float(row["p999_kld"]) for row in rows),
            "max_row_max_kld": max(float(row["max_kld"]) for row in rows),
            **_pooled_token_kld(rows, token_array_dir),
        },
        "row_distributions": house.get("row_distributions"),
        "worst_rows": {
            "lowest_top1": sorted(
                (
                    {"prompt_id": row["prompt_id"], "top1_agreement": row["top1_agreement"]}
                    for row in rows
                ),
                key=lambda entry: entry["top1_agreement"],
            )[:5],
            "highest_mean_kld": sorted(
                (
                    {"prompt_id": row["prompt_id"], "mean_kld": row["mean_kld"]}
                    for row in rows
                ),
                key=lambda entry: -entry["mean_kld"],
            )[:5],
        },
    }


@dataclass(frozen=True)
class AgreementVerdict:
    """A threshold check that is allowed to say "undetermined"."""

    name: str
    value: float | None
    threshold: float
    direction: str
    passed: bool | None
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "direction": self.direction,
            "passed": self.passed,
            "detail": self.detail,
        }


def agreement_verdict(
    summary: Mapping[str, Any],
    *,
    mean_kld_max: float = 0.30,
    p999_kld_max: float = 3.0,
    top1_min: float = 0.85,
) -> dict[str, Any]:
    """Check the GLM-lineage community-wow thresholds against a split summary.

    The KLD check is decided on the **interval**, not the point estimate: the
    artifact clears the threshold only when the rigorous upper bound clears it,
    fails only when the rigorous lower bound exceeds it, and is otherwise
    reported as undetermined. Anything else would be reading a bound as a
    measurement.
    """

    weighted = summary.get("position_weighted", {})
    tail = summary.get("token_tail") or {}
    lower = weighted.get("mean_kld_lower")
    upper = weighted.get("mean_kld_upper")
    top1 = weighted.get("top1_agreement")
    p999 = tail.get("p999_kld_pooled")
    p999_source = tail.get("p999_kld_source")
    if p999 is None:
        p999 = tail.get("max_row_p999_kld")
        p999_source = "worst per-row p99.9 (pooled arrays unavailable)"

    if lower is None or upper is None:
        kld_passed: bool | None = None
        detail = "no KLD bounds available"
    elif upper <= mean_kld_max:
        kld_passed = True
        detail = f"upper bound {upper:.4f} <= {mean_kld_max}"
    elif lower > mean_kld_max:
        kld_passed = False
        detail = f"lower bound {lower:.4f} > {mean_kld_max}"
    else:
        kld_passed = None
        detail = (
            f"interval [{lower:.4f}, {upper:.4f}] straddles {mean_kld_max}; the "
            "truncated teacher cannot decide this threshold"
        )

    checks = [
        AgreementVerdict(
            name="mean_kld_le_threshold",
            value=lower,
            threshold=mean_kld_max,
            direction="<=",
            passed=kld_passed,
            detail=detail,
        ),
        AgreementVerdict(
            name="p999_kld_le_threshold",
            value=p999,
            threshold=p999_kld_max,
            direction="<=",
            passed=None if p999 is None else bool(p999 <= p999_kld_max),
            detail=f"p99.9 of the per-token proportional-tail KLD ({p999_source})",
        ),
        AgreementVerdict(
            name="top1_ge_threshold",
            value=top1,
            threshold=top1_min,
            direction=">=",
            passed=None if top1 is None else bool(top1 >= top1_min),
            detail="position-weighted top-1 agreement with the teacher",
        ),
    ]
    decided = [check.passed for check in checks]
    return {
        "record_type": "dsv4_teacher_agreement_verdict_v1",
        "schema_version": 1,
        "thresholds": {
            "mean_kld_max": mean_kld_max,
            "p999_kld_max": p999_kld_max,
            "top1_min": top1_min,
            # Frozen identifier recorded in existing artifacts; keeps its pre-rename module path.
            "provenance": (
                "mlx_vq.quality.rc_gates.COMMUNITY_WOW_TARGETS, inherited from the "
                "GLM-4.5-Air / GLM-5.2 lineage"
            ),
        },
        "checks": [check.as_dict() for check in checks],
        "all_passed": all(value is True for value in decided),
        "any_failed": any(value is False for value in decided),
        "undetermined": [
            check.name for check in checks if check.passed is None
        ],
    }
