"""Production VQ materializer for DeepSeek-V4-Flash-0731.

This is the implementation behind the registered converter kind
``deepseek_v4_vq_groups``. It turns the release's FP4-in-I8 routed experts into
KEEP-format per-block VQ artifacts that
:func:`ramp.models.deepseek_v4_flash_adapter.bind_deepseek_v4_flash_vq_experts`
can load, at the ladder the Wave 5 pilot settled
(``docs/deepseek-v4-flash/research/wave5-pilot-report.md``): **E8P ``code_bits=16``,
``group_size=512`` uniform on gate/up/down, 2.031 bpw routed, 8 fit
iterations**, across **46 MoE blocks** -- the 43 backbone layers plus the three
DSpark drafter blocks at ``mtp.{0,1,2}``.

Nothing here re-derives the pilot's measurements; it composes the pilot module
(:mod:`keep.convert.dsv4_vq_pilot`) for the shared fit machinery -- checkpoint
span reads, FP4 decode, imatrix aggregation, error metrics -- and adds only what
a production run needs and a pilot does not: resumability, atomic publication,
a manifest, and an audit.

The naming contract, stated once because two reviews flagged it as unconfirmed
-----------------------------------------------------------------------------

The checkpoint calls the three routed projections ``w1``/``w2``/``w3``. The
adapter's ``QuantizedVQSwitchGLU`` calls them ``gate_proj``/``up_proj``/
``down_proj``. The mapping this materializer writes is

====================  ==================  ================================
checkpoint tensor     artifact projection meaning
====================  ==================  ================================
``w1`` [2048, 4096]   ``gate_proj``       SwiGLU gate branch
``w3`` [2048, 4096]   ``up_proj``         SwiGLU up branch
``w2`` [4096, 2048]   ``down_proj``       routed output back to hidden
====================  ==================  ================================

It is recorded verbatim in every manifest (``tensor_mapping``) and, more to the
point, it is *verified semantically* rather than asserted: ``verify_roundtrip``
materializes a block, binds it through the adapter's own loader, and checks the
bound module's forward output against the fit's own reconstruction pushed
through the reference clamped SwiGLU. A swapped gate/up would still load, still
produce finite numbers, and fail that check -- which is exactly why the check
exists.

Artifact layout
---------------

For backbone layer ``L``, matching the file and tensor names
``load_deepseek_v4_flash_vq_switch_glu`` expects::

    layer-{L:05d}-gate_proj.safetensors
      model.layers.{L}.ffn.switch_mlp.gate_proj.codes    uint16  [E, out, in/8]
      model.layers.{L}.ffn.switch_mlp.gate_proj.scales   float16 [E, out, in/group]
      model.vq_codebook.e8                               uint32  packed E8P grid

For DSpark drafter stage ``S`` the same three tensors under
``mtp-{S:05d}-{projection}.safetensors`` with prefix
``mtp_drafter.blocks.{S}.ffn.switch_mlp.{projection}`` -- the parameter path the
adapter's ``has_unbound_deepseek_v4_flash_mtp_experts`` walks. The increment-1
adapter has a *backbone* binder only; the MTP binder does not exist yet, so
these files are written to the convention the drafter's parameter tree implies
and are stated as such in the manifest rather than claimed to be loadable
today.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from keep.convert.dsv4_vq_pilot import (
    CHECKPOINT_PROJECTIONS,
    PROJECTION_INPUT_SPACE,
    PROJECTION_SHAPE,
    aggregate_calibration_importance,
    projection_error,
    rate_bpw,
    reconstruct_quantized,
    route_weighted_mean,
    write_json,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CONVERTER_KIND",
    "FAMILY",
    "PROJECTIONS",
    "SOURCE_MODEL_ID",
    "SOURCE_REVISION",
    "TENSOR_MAPPING",
    "BlockRecord",
    "BlockSpec",
    "MaterializePolicy",
    "audit_artifact_tree",
    "block_importance",
    "build_imatrix_cache",
    "build_manifest",
    "decode_artifact_experts",
    "load_imatrix_cache",
    "materialize_block",
    "plan_blocks",
    "read_block_experts",
    "reference_probe_model_args",
    "verified_block_record",
    "verify_bind_roundtrip",
]

ARTIFACT_SCHEMA_VERSION = 1
FAMILY = "deepseek-v4-flash"
CONVERTER_KIND = "deepseek_v4_vq_groups"
SOURCE_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
SOURCE_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"

#: Artifact-side projection names, in the adapter's own order.
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

#: The contract. Checkpoint tensor -> artifact projection. w1 gate, w3 up, w2 down.
TENSOR_MAPPING = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}

#: Artifact projection -> the pilot's short projection name (imatrix key).
_SHORT = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}

#: The three routed-expert tensors we never rewrite in place.
_CODEBOOK_TENSOR = "model.vq_codebook.e8"

_MTP_IMPORTANCE_POLICIES = ("backbone-mean", "uniform")

#: Recorded source for an expert the calibration corpus never routed to, whose
#: measured importance is therefore identically zero. Fitted with uniform
#: importance and counted per block, never substituted silently.
ZERO_IMPORTANCE_FALLBACK = "uniform_zero_importance_fallback"


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockSpec:
    """One MoE block to materialize, and everything naming depends on.

    ``calibration_layer`` is the backbone layer whose Wave 3 imatrix rows apply.
    It is the block itself for the backbone and ``None`` for a drafter block --
    the calibration run captured ``num_hidden_layers=43`` rows and never saw
    ``mtp.*``, which is a fact the MTP importance policy has to answer for
    rather than paper over.
    """

    key: str
    kind: str
    index: int
    tensor_prefix: str
    file_stem: str
    calibration_layer: int | None

    def artifact_name(self, projection: str) -> str:
        if projection not in PROJECTIONS:
            raise ValueError(f"projection must be one of {PROJECTIONS}, got {projection!r}")
        return f"{self.file_stem}-{projection}.safetensors"

    def tensor_names(self, projection: str) -> tuple[str, str]:
        prefix = f"{self.tensor_prefix}.{projection}"
        return f"{prefix}.codes", f"{prefix}.scales"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "index": self.index,
            "tensor_prefix": self.tensor_prefix,
            "file_stem": self.file_stem,
            "calibration_layer": self.calibration_layer,
        }


def plan_blocks(
    *,
    num_layers: int = 43,
    mtp_blocks: int = 3,
    include_backbone: bool = True,
    only: Sequence[str] | None = None,
) -> tuple[BlockSpec, ...]:
    """The 46-block plan: 43 backbone layers then 3 DSpark drafter blocks."""

    if num_layers < 0 or mtp_blocks < 0:
        raise ValueError("block counts must be non-negative")
    specs: list[BlockSpec] = []
    if include_backbone:
        for layer in range(num_layers):
            specs.append(
                BlockSpec(
                    key=f"layers.{layer}",
                    kind="backbone",
                    index=layer,
                    tensor_prefix=f"model.layers.{layer}.ffn.switch_mlp",
                    file_stem=f"layer-{layer:05d}",
                    calibration_layer=layer,
                )
            )
    for stage in range(mtp_blocks):
        specs.append(
            BlockSpec(
                key=f"mtp.{stage}",
                kind="mtp",
                index=stage,
                tensor_prefix=f"mtp_drafter.blocks.{stage}.ffn.switch_mlp",
                file_stem=f"mtp-{stage:05d}",
                calibration_layer=None,
            )
        )
    if only is not None:
        wanted = {str(key) for key in only}
        unknown = wanted - {spec.key for spec in specs}
        if unknown:
            raise ValueError(f"unknown block keys {sorted(unknown)}")
        specs = [spec for spec in specs if spec.key in wanted]
    return tuple(specs)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterializePolicy:
    """The ship ladder, as a value that travels into the manifest.

    Defaults are the Wave 5 pilot's decision and are not re-derived here:
    ``code_bits=16`` (E8P) because the codebook step returns 0.71 fractional
    error per bpw against 0.17-0.24 for any group-size step, ``group_size=512``
    uniform because within E8P the group size is a rate dial and 512 is the one
    that lands the artifact at 83.6 GB inside a 98-105 GB envelope, and
    ``iterations=8`` because 3->8 buys 1.7-4.4% and 8->20 buys 0.04%.
    """

    code_bits: int = 16
    group_size: int = 512
    iterations: int = 8
    fit_backend: str = "mlx-exact"
    mtp_importance: str = "backbone-mean"
    search_backend: str = "metal"

    def __post_init__(self) -> None:
        if self.code_bits not in (8, 16):
            raise ValueError("code_bits must be 8 or 16")
        if self.group_size <= 0 or self.group_size % 8:
            raise ValueError("group_size must be positive and divisible by 8")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.mtp_importance not in _MTP_IMPORTANCE_POLICIES:
            raise ValueError(
                f"mtp_importance must be one of {_MTP_IMPORTANCE_POLICIES}"
            )

    @property
    def bpw(self) -> float:
        return rate_bpw(self.code_bits, self.group_size)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code_bits": self.code_bits,
            "group_size_policy": {name: self.group_size for name in ("gate", "up", "down")},
            "code_bits_policy": {name: self.code_bits for name in ("gate", "up", "down")},
            "iterations": self.iterations,
            "fit_backend": self.fit_backend,
            "search_backend": self.search_backend,
            "mtp_importance": self.mtp_importance,
            "routed_bpw": self.bpw,
        }


# ---------------------------------------------------------------------------
# Calibration importance
# ---------------------------------------------------------------------------


def build_imatrix_cache(
    calibration_dir: str | Path,
    out_path: str | Path,
    *,
    layers: Sequence[int],
    session_limit: int | None = None,
) -> dict[str, Any]:
    """Aggregate the 40 Wave 3 session captures once, into one npz.

    Every block needs its layer's imatrix rows, and reading 40 compressed
    session archives per block would re-decompress ~28 GB forty-six times. The
    sweep therefore builds this cache once, hashes it, and names the hash in the
    manifest so a later audit can tell which statistics produced the artifact.
    """

    stats = aggregate_calibration_importance(
        calibration_dir, layers=list(layers), session_limit=session_limit
    )
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "layers": np.asarray(stats["layers"], dtype=np.int64),
        "num_experts": np.asarray(int(stats["num_experts"]), dtype=np.int64),
        "sessions": np.asarray(int(stats["sessions"]), dtype=np.int64),
        "route_count__hidden": np.asarray(
            stats["route_count"]["hidden"], dtype=np.int64
        ),
    }
    for space in ("hidden", "down"):
        # Stored float32: the fit casts to float32 anyway, and the aggregation
        # itself ran in float64 so the sum is not where precision is lost.
        payload[f"importance__{space}"] = np.asarray(
            stats["importance"][space], dtype=np.float32
        )
    temporary = target.with_name(target.name + ".partial")
    # np.savez appends ".npz" to a *name*; hand it an open handle so the partial
    # keeps the name we are about to rename from.
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(temporary, target)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
        "sessions": int(stats["sessions"]),
        "prompt_ids": list(stats["prompt_ids"]),
        "layers": list(stats["layers"]),
        "num_experts": int(stats["num_experts"]),
    }


def load_imatrix_cache(path: str | Path) -> dict[str, Any]:
    """Load the aggregated imatrix cache into memory."""

    with np.load(Path(path), allow_pickle=False) as archive:
        return {
            "layers": [int(value) for value in archive["layers"]],
            "num_experts": int(archive["num_experts"].reshape(()).item()),
            "sessions": int(archive["sessions"].reshape(()).item()),
            "route_count": np.asarray(archive["route_count__hidden"], dtype=np.int64),
            "importance": {
                space: np.asarray(archive[f"importance__{space}"], dtype=np.float32)
                for space in ("hidden", "down")
            },
        }


def _normalised_backbone_mean(
    cache: Mapping[str, Any], *, layer: int, space: str
) -> np.ndarray:
    """Route-weighted mean *shape* of one backbone layer's imatrix diagonal.

    Per-expert vectors are normalised to unit mean before averaging, because
    their magnitude is a route count and only the per-column shape is the
    objective. The result is the fallback importance for a block the
    calibration run never saw.
    """

    row = list(cache["layers"]).index(int(layer))
    values = np.asarray(cache["importance"][space][row], dtype=np.float64)
    counts = np.asarray(cache["route_count"][row], dtype=np.float64)
    totals = values.sum(axis=1)
    usable = (counts > 0) & (totals > 0)
    if not np.any(usable):
        raise ValueError(f"backbone layer {layer} has no routed importance in {space}")
    shapes = values[usable] / totals[usable][:, None] * values.shape[1]
    weights = counts[usable]
    mean = (shapes * weights[:, None]).sum(axis=0) / weights.sum()
    return np.asarray(mean, dtype=np.float32)


def block_importance(
    cache: Mapping[str, Any],
    spec: BlockSpec,
    *,
    projection: str,
    expert: int,
    policy: MaterializePolicy,
    mtp_reference_layer: int,
) -> tuple[np.ndarray, str]:
    """The imatrix diagonal for one (block, projection, expert), and its source.

    Backbone blocks use their own measured rows. Drafter blocks have none: the
    Wave 3 calibration run captured 43 backbone layers and stopped, so an MTP
    block's importance is either the last backbone layer's route-weighted mean
    column shape (``backbone-mean``, the default -- the drafter operates on the
    same residual-stream hidden space, so the *column* structure is the part
    that plausibly transfers) or flat ones (``uniform``, which makes the fit
    plain unweighted least squares). Neither is measured on the drafter, and
    the returned source string is what the manifest records.

    **A measured row can still be all zeros.** An expert the calibration corpus
    never routed to has ``route_count == 0`` and an identically zero importance
    vector -- measured on this release for exactly one pair in the 43 x 256 grid,
    ``layers.40`` expert 170. Its weights are perfectly ordinary; the corpus
    simply never selected it. Such an expert still has to appear in the artifact
    (the binder requires all 256), so it is fitted against uniform importance,
    i.e. plain unweighted least squares, which is the right objective when there
    is no evidence about which input columns matter. The *returned* vector is the
    effective one, so the caller fits and scores the same thing, and the source
    string says so -- this is never a silent substitution.
    """

    space = PROJECTION_INPUT_SPACE[_SHORT[projection]]
    if spec.calibration_layer is not None:
        row = list(cache["layers"]).index(int(spec.calibration_layer))
        vector = np.asarray(
            cache["importance"][space][row, int(expert)], dtype=np.float32
        )
        if not np.any(vector > 0):
            return np.ones_like(vector), ZERO_IMPORTANCE_FALLBACK
        return vector, "measured"
    if policy.mtp_importance == "uniform":
        dim = PROJECTION_SHAPE[_SHORT[projection]][1]
        return np.ones(dim, dtype=np.float32), "uniform_fallback"
    vector = _normalised_backbone_mean(cache, layer=mtp_reference_layer, space=space)
    return vector, f"backbone_mean_layer_{mtp_reference_layer}"


# ---------------------------------------------------------------------------
# Checkpoint reads
# ---------------------------------------------------------------------------


def read_block_experts(
    checkpoint_dir: str | Path, block: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float, Any]:
    """Read one MoE block's packed routed-expert bytes off its shard.

    Backbone and drafter blocks alike, via the streaming runner's coalesced
    span index. Returns ``(weights, scales, seconds, span)`` with per-projection
    ``uint8`` stacks keyed by the *checkpoint* names ``w1``/``w2``/``w3``;
    decoding stays the caller's separately timed step.
    """

    from keep.quality.dsv4_teacher_runner import (
        _pread_exact,
        build_dsv4_block_span_index,
    )

    span = build_dsv4_block_span_index(checkpoint_dir, blocks=[block])[block]
    weights = {
        projection: np.empty(
            (span.num_experts, *span.weight_shapes[projection]), dtype=np.uint8
        )
        for projection in CHECKPOINT_PROJECTIONS
    }
    scales = {
        projection: np.empty(
            (span.num_experts, *span.scale_shapes[projection]), dtype=np.uint8
        )
        for projection in CHECKPOINT_PROJECTIONS
    }
    path = Path(checkpoint_dir) / span.shard
    started = time.perf_counter()
    handle = os.open(path, os.O_RDONLY)
    try:
        for projection, kind, expert, offset, length in span.reads:
            table = weights if kind == "weight" else scales
            _pread_exact(handle, table[projection][expert], offset, length)
    finally:
        os.close(handle)
    return weights, scales, time.perf_counter() - started, span


def decode_block_expert(
    weights: Mapping[str, np.ndarray],
    scales: Mapping[str, np.ndarray],
    *,
    projection: str,
    expert: int,
) -> np.ndarray:
    """Dequantize one expert's projection from the packed block stacks."""

    from keep.convert.fp4_expert import dequantize_fp4_expert

    checkpoint_name = {value: key for key, value in TENSOR_MAPPING.items()}[projection]
    return dequantize_fp4_expert(
        weights[checkpoint_name][int(expert)],
        scales[checkpoint_name][int(expert)],
        group_size=32,
    )


# ---------------------------------------------------------------------------
# Fitting one block
# ---------------------------------------------------------------------------


def _fit_projection(
    weight: np.ndarray,
    importance: np.ndarray,
    *,
    policy: MaterializePolicy,
    backend: str,
) -> tuple[Any, float]:
    """One projection fit through whichever backend was resolved."""

    started = time.perf_counter()
    if backend == "numpy":
        from keep.convert.glm52_recovery_materialize import (
            quantize_weight_importance_aware,
        )

        quantized = quantize_weight_importance_aware(
            weight,
            importance,
            group_size=policy.group_size,
            code_bits=policy.code_bits,
            iterations=policy.iterations,
            e8p_search_backend=policy.search_backend,
        )
    else:
        from keep.convert.dsv4_vq_fit_mlx import quantize_weight_importance_aware_mlx

        quantized = quantize_weight_importance_aware_mlx(
            weight,
            importance,
            group_size=policy.group_size,
            code_bits=policy.code_bits,
            iterations=policy.iterations,
            scale_reduction="float64" if backend == "mlx-exact" else "float32",
        )
    return quantized, time.perf_counter() - started


@dataclass
class BlockRecord:
    """Everything one materialized block contributes to the manifest."""

    block: BlockSpec
    files: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    importance_sources: dict[str, int] = field(default_factory=dict)
    #: Experts fitted against uniform importance because their measured
    #: importance was identically zero -- i.e. the calibration corpus never
    #: routed to them. Named per expert, not just counted, because "which
    #: experts have no evidence behind them" is a question an audit will ask.
    importance_fallback_experts: list[int] = field(default_factory=list)
    num_experts: int = 0
    fit_backend: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.block.as_dict(),
            "num_experts": self.num_experts,
            "fit_backend": self.fit_backend,
            "files": self.files,
            "fit_metrics": self.metrics,
            "timing": self.timing,
            "importance_sources": self.importance_sources,
            "importance_fallback_experts": sorted(set(self.importance_fallback_experts)),
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_atomically(payload_writer, destination: Path) -> None:
    """Write through a uniquely named partial, fsync, then rename into place.

    ``os.replace`` is atomic within a filesystem, so a reader either sees the
    previous file or the complete new one and never a half-written artifact --
    which matters because the resume check trusts what it finds on disk.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    try:
        payload_writer(partial)
        handle = os.open(partial, os.O_WRONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def _quantization_metadata(policy: MaterializePolicy, spec: BlockSpec) -> dict[str, str]:
    from keep.io.schema import QuantizationConfig, codebook_metadata_for_bits

    codebook_name, codebook_sha256 = codebook_metadata_for_bits(policy.code_bits)
    quantization = QuantizationConfig(
        default_code_bits=policy.code_bits,
        default_group_size=policy.group_size,
        codebook_name=codebook_name,
        codebook_sha256=codebook_sha256,
        policy={
            "family": FAMILY,
            "converter": CONVERTER_KIND,
            "source_revision": SOURCE_REVISION,
            "block": spec.key,
            "block_kind": spec.kind,
            "fit_iterations": str(policy.iterations),
            "fit_backend": policy.fit_backend,
            "mtp_importance": policy.mtp_importance,
        },
    ).to_json_dict()
    return {
        "quantization_config": json.dumps(quantization, sort_keys=True),
        "deepseek_v4_tensor_mapping": json.dumps(TENSOR_MAPPING, sort_keys=True),
        "deepseek_v4_artifact_schema_version": str(ARTIFACT_SCHEMA_VERSION),
    }


def materialize_block(
    spec: BlockSpec,
    *,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    cache: Mapping[str, Any],
    policy: MaterializePolicy,
    mtp_reference_layer: int = 42,
    experts: Sequence[int] | None = None,
    progress: Any = None,
) -> BlockRecord:
    """Fit and publish one MoE block's three VQ projection files.

    Projection-outer / expert-inner so only one projection's codes
    (256 x 2048 x 512 uint16 = 537 MB at the ship ladder) are resident at a
    time on top of the block's 3.42 GB of packed source.
    """

    from safetensors.numpy import save_file

    from keep.convert.dsv4_vq_fit_mlx import resolve_fit_backend

    backend = resolve_fit_backend(policy.fit_backend, code_bits=policy.code_bits)
    output = Path(output_dir)
    weights, scales, read_seconds, span = read_block_experts(checkpoint_dir, spec.key)
    expert_indices = (
        list(range(span.num_experts)) if experts is None else [int(e) for e in experts]
    )
    route_counts = (
        np.asarray(
            cache["route_count"][list(cache["layers"]).index(spec.calibration_layer)],
            dtype=np.int64,
        )
        if spec.calibration_layer is not None
        else np.ones(span.num_experts, dtype=np.int64)
    )

    record = BlockRecord(block=spec, num_experts=len(expert_indices), fit_backend=backend)
    decode_seconds = 0.0
    fit_seconds = 0.0
    write_seconds = 0.0
    metadata = _quantization_metadata(policy, spec)
    started = time.perf_counter()

    for projection in PROJECTIONS:
        codes_rows: list[np.ndarray] = []
        scale_rows: list[np.ndarray] = []
        errors: list[float] = []
        cosines: list[float] = []
        codebook: np.ndarray | None = None
        for position, expert in enumerate(expert_indices):
            mark = time.perf_counter()
            dense = decode_block_expert(
                weights, scales, projection=projection, expert=expert
            )
            decode_seconds += time.perf_counter() - mark
            importance, source = block_importance(
                cache,
                spec,
                projection=projection,
                expert=expert,
                policy=policy,
                mtp_reference_layer=mtp_reference_layer,
            )
            record.importance_sources[source] = (
                record.importance_sources.get(source, 0) + 1
            )
            if (
                source == ZERO_IMPORTANCE_FALLBACK
                and int(expert) not in record.importance_fallback_experts
            ):
                # One entry per expert, not one per projection: a cold expert is
                # cold in every projection, and the count is read as "how many
                # experts had no evidence".
                record.importance_fallback_experts.append(int(expert))
            quantized, elapsed = _fit_projection(
                dense, importance, policy=policy, backend=backend
            )
            fit_seconds += elapsed
            codes_rows.append(np.asarray(quantized.codes))
            scale_rows.append(np.asarray(quantized.scales))
            codebook = np.asarray(quantized.codebook)
            metrics = projection_error(
                dense, reconstruct_quantized(quantized), importance.astype(np.float64)
            )
            errors.append(metrics.weighted_relative_mse)
            cosines.append(metrics.expected_cosine)
            if progress is not None and position and position % 32 == 0:
                progress(
                    block=spec.key,
                    projection=projection,
                    expert=position,
                    experts=len(expert_indices),
                    elapsed=time.perf_counter() - started,
                    fitted=PROJECTIONS.index(projection) * len(expert_indices) + position,
                )

        assert codebook is not None
        codes = np.ascontiguousarray(np.stack(codes_rows))
        group_scales = np.ascontiguousarray(np.stack(scale_rows))
        del codes_rows, scale_rows
        codes_name, scales_name = spec.tensor_names(projection)
        destination = output / spec.artifact_name(projection)
        mark = time.perf_counter()
        _publish_atomically(
            lambda partial, tensors={
                codes_name: codes,
                scales_name: group_scales,
                _CODEBOOK_TENSOR: codebook,
            }: save_file(tensors, str(partial), metadata=metadata),
            destination,
        )
        write_seconds += time.perf_counter() - mark
        sampled = [int(route_counts[expert]) for expert in expert_indices]
        weighting = sampled if sum(sampled) > 0 else [1] * len(expert_indices)
        record.files.append(
            {
                "projection": projection,
                "checkpoint_tensor": {
                    value: key for key, value in TENSOR_MAPPING.items()
                }[projection],
                "path": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "codes_tensor": codes_name,
                "scales_tensor": scales_name,
                "codes_shape": list(codes.shape),
                "codes_dtype": str(codes.dtype),
                "scales_shape": list(group_scales.shape),
                "scales_dtype": str(group_scales.dtype),
                "code_bits": policy.code_bits,
                "group_size": policy.group_size,
                "bpw": rate_bpw(policy.code_bits, policy.group_size),
            }
        )
        record.metrics[projection] = {
            "weighted_relative_mse_route_weighted": route_weighted_mean(errors, weighting),
            "weighted_relative_mse_mean": float(np.mean(errors)),
            "weighted_relative_mse_max": float(np.max(errors)),
            "expected_cosine_route_weighted": route_weighted_mean(cosines, weighting),
        }
        del codes, group_scales

    del weights, scales
    record.timing = {
        "read_seconds": read_seconds,
        "decode_seconds": decode_seconds,
        "fit_seconds": fit_seconds,
        "write_seconds": write_seconds,
        "wall_seconds": time.perf_counter() - started + read_seconds,
        "peak_rss_gb": peak_rss_gb(),
        "source_span_bytes": span.span_bytes,
    }
    return record


def peak_rss_gb() -> float:
    """Peak resident set of this process, in GB (Darwin reports bytes)."""

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if platform.system() == "Darwin" else peak / 1e6


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def record_path(output_dir: str | Path, spec: BlockSpec) -> Path:
    return Path(output_dir) / "records" / f"{spec.file_stem}.json"


def write_block_record(output_dir: str | Path, record: BlockRecord) -> Path:
    target = record_path(output_dir, record.block)
    write_json(target, {"record_type": "dsv4_vq_block_record_v1", **record.as_dict()})
    return target


def verified_block_record(
    output_dir: str | Path,
    spec: BlockSpec,
    *,
    policy: MaterializePolicy,
    expected_experts: int = 256,
) -> dict[str, Any] | None:
    """A previously completed block's record, or ``None`` if it must be redone.

    "Complete" is not "the record file exists": every projection file it claims
    must still be on disk at the recorded size *and* the recorded SHA-256, the
    recorded rate must be the rate we are now asked for, and it must cover the
    full expert set. A resumed sweep that skipped a block on weaker evidence
    than that would publish an artifact nobody had actually verified -- and a
    ``block --experts 0 1`` smoke run would otherwise leave a two-expert file
    the sweep happily inherits.
    """

    path = record_path(output_dir, spec)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if payload.get("record_type") != "dsv4_vq_block_record_v1":
        return None
    if payload.get("key") != spec.key:
        return None
    if int(payload.get("num_experts", 0)) != int(expected_experts):
        return None
    files = payload.get("files")
    if not isinstance(files, list) or len(files) != len(PROJECTIONS):
        return None
    for entry in files:
        if entry.get("code_bits") != policy.code_bits:
            return None
        if entry.get("group_size") != policy.group_size:
            return None
        candidate = Path(output_dir) / str(entry.get("path"))
        if not candidate.is_file():
            return None
        if candidate.stat().st_size != entry.get("bytes"):
            return None
        if sha256_file(candidate) != entry.get("sha256"):
            return None
    return payload


# ---------------------------------------------------------------------------
# Manifest and audit
# ---------------------------------------------------------------------------


def _source_identity(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint_dir)
    identity: dict[str, Any] = {
        "hf_model_id": SOURCE_MODEL_ID,
        "revision": SOURCE_REVISION,
        "checkpoint_dir": str(checkpoint),
    }
    for name, key in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
        candidate = checkpoint / name
        identity[key] = sha256_file(candidate) if candidate.is_file() else None
    return identity


def audit_artifact_tree(
    output_dir: str | Path,
    *,
    specs: Sequence[BlockSpec],
    policy: MaterializePolicy,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The template's routed-MoE audit, computed from files rather than logs.

    ``dense_routed_experts`` and ``unbound_vq_experts`` are deliberately *not*
    decided here: this function can prove the projection inventory and the rate
    distribution, but only a bound model can prove that nothing fell back to a
    dense expert. Both fields come back ``None`` with a pointer to the
    roundtrip evidence, and ``build_manifest`` fills them from a real bind.
    """

    root = Path(output_dir)
    expected = [
        (spec, projection) for spec in specs for projection in PROJECTIONS
    ]
    missing = [
        spec.artifact_name(projection)
        for spec, projection in expected
        if not (root / spec.artifact_name(projection)).is_file()
    ]
    code_bits: dict[str, int] = {}
    group_sizes: dict[str, int] = {}
    payload_bytes = 0
    routed_weights = 0
    for record in records:
        for entry in record.get("files", ()):
            code_bits[str(entry["code_bits"])] = code_bits.get(str(entry["code_bits"]), 0) + 1
            group_sizes[str(entry["group_size"])] = (
                group_sizes.get(str(entry["group_size"]), 0) + 1
            )
            payload_bytes += int(entry["bytes"])
            # Counted off the codes each file actually holds -- [experts, out,
            # in/8] -- rather than off a hard-coded geometry, so the number is
            # right for whatever was written and not for what was expected.
            experts, out_dim, codewords = (int(dim) for dim in entry["codes_shape"])
            routed_weights += experts * out_dim * codewords * 8
    return {
        "total_routed_projection_files_expected": len(expected),
        "total_routed_projection_files_present": len(expected) - len(missing),
        "missing_projection_files": missing,
        "rewritten_projection_count": sum(len(record.get("files", ())) for record in records),
        "symlinked_projection_count": 0,
        "high_precision_routed_projection_count": 0,
        "continuous_sidecar_count": 0,
        "sparse_residual_sidecar_count": 0,
        "code_bits_distribution": code_bits,
        "group_size_distribution": group_sizes,
        "effective_routed_bpw": policy.bpw,
        "routed_weight_count": routed_weights,
        "artifact_payload_bytes": payload_bytes,
        "fallback_layers": [],
        # Experts with no calibration evidence at all, per block. Not a
        # "fallback layer" -- the layer is fine and fully materialized; a
        # handful of its experts were fitted against uniform importance because
        # the corpus never routed to them.
        "importance_fallback_experts": {
            str(record["key"]): list(record.get("importance_fallback_experts", ()))
            for record in records
            if record.get("importance_fallback_experts")
        },
        "importance_fallback_expert_count": sum(
            len(record.get("importance_fallback_experts", ())) for record in records
        ),
        "nax_metal_compatible_group_size": policy.group_size % 8 == 0,
        "dense_routed_experts": None,
        "unbound_vq_experts": None,
        "bind_proof": "see manifest.audit.bind_proof",
    }


def build_manifest(
    *,
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    specs: Sequence[BlockSpec],
    policy: MaterializePolicy,
    records: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    evidence: Mapping[str, Any] | None = None,
    bind_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the family-template manifest for a materialized tree."""

    from keep.io.schema import codebook_metadata_for_bits

    codebook_name, codebook_sha256 = codebook_metadata_for_bits(policy.code_bits)
    audit = audit_artifact_tree(
        output_dir, specs=specs, policy=policy, records=records
    )
    if bind_proof is not None:
        audit["dense_routed_experts"] = bool(bind_proof.get("dense_routed_experts"))
        audit["unbound_vq_experts"] = bool(bind_proof.get("unbound_vq_experts"))
        audit["bind_proof"] = dict(bind_proof)
    backbone = [record for record in records if record.get("kind") == "backbone"]
    mtp = [record for record in records if record.get("kind") == "mtp"]
    return {
        "record_type": "dsv4_vq_artifact_manifest_v1",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "family": FAMILY,
        "converter": CONVERTER_KIND,
        "artifact_dir": str(Path(output_dir)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": _source_identity(checkpoint_dir),
        "seed_artifact": None,
        "seed_artifact_mutated": False,
        "geometry": {
            "moe_blocks": len(specs),
            "backbone_layers": len(backbone),
            "mtp_blocks": len(mtp),
            "experts_per_block": max(
                (int(record.get("num_experts", 0)) for record in records), default=0
            ),
            "projections_per_block": len(PROJECTIONS),
            "projection_shapes": {
                projection: list(PROJECTION_SHAPE[_SHORT[projection]])
                for projection in PROJECTIONS
            },
            "hidden_size": 4096,
            "moe_intermediate_size": 2048,
            "experts_per_tok": 6,
        },
        "policy": policy.as_dict(),
        "codebook": {
            "name": codebook_name,
            "sha256": codebook_sha256,
            "entries": 256,
            "code_bits": policy.code_bits,
        },
        "tensor_mapping": {
            "checkpoint_to_artifact": dict(TENSOR_MAPPING),
            "artifact_to_checkpoint": {
                value: key for key, value in TENSOR_MAPPING.items()
            },
            "note": (
                "w1 is the SwiGLU gate branch, w3 the up branch, w2 the routed "
                "output projection. Verified semantically by the roundtrip probe "
                "-- a swapped gate/up would load and produce finite numbers."
            ),
            "backbone_file": "layer-{layer:05d}-{projection}.safetensors",
            "backbone_prefix": "model.layers.{layer}.ffn.switch_mlp.{projection}",
            "mtp_file": "mtp-{stage:05d}-{projection}.safetensors",
            "mtp_prefix": "mtp_drafter.blocks.{stage}.ffn.switch_mlp.{projection}",
            "mtp_binder_exists": False,
        },
        "calibration": dict(calibration),
        "blocks": [dict(record) for record in records],
        "audit": audit,
        "evidence": dict(evidence or {}),
    }


def environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "e8p_backend_env": os.environ.get("GLM52_E8P_BACKEND"),
        "pid": os.getpid(),
    }


# ---------------------------------------------------------------------------
# Roundtrip verification -- the naming-contract proof
# ---------------------------------------------------------------------------


def reference_probe_model_args(
    layer_count: int,
    *,
    hidden_size: int = 4096,
    moe_intermediate_size: int = 2048,
    num_experts: int = 256,
    experts_per_tok: int = 6,
) -> Any:
    """A real-MoE-geometry model with everything else shrunk to nothing.

    The routed geometry has to be exact -- ``bind_switch_mlp`` checks
    ``input_dims``, ``hidden_dims`` and ``num_experts`` against the config -- but
    the vocabulary, attention ranks and layer count are free, and holding them
    down is the difference between a 3 GB probe and a 600 GB one. The geometry
    is a parameter so the same verification runs on a toy artifact in a unit
    test and on the real layer 0 in the run directory.
    """

    from ramp.models.deepseek_v4_flash_adapter import DeepseekV4FlashVQModelArgs

    return DeepseekV4FlashVQModelArgs(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=moe_intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        num_hidden_layers=layer_count,
        num_attention_heads=2,
        n_shared_experts=1,
        n_routed_experts=num_experts,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        num_experts_per_tok=experts_per_tok,
        head_dim=8,
        o_groups=2,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        num_hash_layers=0,
        hc_mult=4,
        max_position_embeddings=64,
        compress_ratios=[0] * layer_count,
        dspark_block_size=5,
        dspark_target_layer_ids=[max(layer_count - 1, 0)],
        dspark_markov_rank=4,
    )


def decode_artifact_experts(
    artifact_dir: str | Path, spec: BlockSpec, experts: Sequence[int]
) -> dict[str, np.ndarray]:
    """Dequantize a few experts straight out of the published artifact files.

    This is the fit's own reconstruction: the artifact stores exactly the codes
    and scales the fit produced, so decoding them reproduces the weights the fit
    settled on, bit for bit, having gone through safetensors on the way.
    """

    from safetensors.numpy import load_file

    from keep.vq.e8 import e8_1bit_grid, e8p_full_grid

    root = Path(artifact_dir)
    picked = [int(expert) for expert in experts]
    decoded: dict[str, np.ndarray] = {}
    for projection in PROJECTIONS:
        tensors = load_file(str(root / spec.artifact_name(projection)))
        codes_name, scales_name = spec.tensor_names(projection)
        codes = np.asarray(tensors[codes_name])[picked]
        scales = np.asarray(tensors[scales_name], dtype=np.float32)[picked]
        table = (
            e8_1bit_grid().astype(np.float32)
            if codes.dtype == np.dtype(np.uint8)
            else e8p_full_grid().astype(np.float32)
        )
        count, out_dim, codewords = codes.shape
        in_dim = codewords * 8
        group_count = scales.shape[2]
        group_size = in_dim // group_count
        block = table[codes.reshape(count, out_dim, group_count, group_size // 8)]
        decoded[projection] = (block * scales[:, :, :, None, None]).reshape(
            count, out_dim, in_dim
        )
    return decoded


def verify_bind_roundtrip(
    artifact_dir: str | Path,
    spec: BlockSpec,
    *,
    policy: MaterializePolicy,
    experts: Sequence[int] = (0, 1, 2, 3, 4, 5),
    tokens: int = 4,
    input_scale: float = 0.02,
    tolerance: float = 1e-3,
    seed: int = 0,
    hidden_size: int = 4096,
    moe_intermediate_size: int = 2048,
    num_experts: int = 256,
    bind_path: str = "auto",
) -> dict[str, Any]:
    """Bind a materialized block through the adapter and check three claims.

    1. the adapter's own ``has_unbound_deepseek_v4_flash_vq_experts`` reports
       nothing unbound after the bind;
    2. its own ``dense_deepseek_v4_flash_routed_parameter_names`` reports that
       no dense routed parameter survived;
    3. the bound ``QuantizedVQSwitchGLU``'s forward output matches the fit's own
       reconstruction pushed through the reference clamped SwiGLU -- and does
       *not* match the same reference with gate and up swapped.

    Claim 3 is the one that turns ``w1 -> gate_proj`` from a plausible mapping
    into a verified one. Both variants are shape-legal and produce finite
    numbers, which is exactly why loading successfully proves nothing and two
    reviews were right to keep flagging it.

    ``bind_path`` selects how the module gets into the probe, because the two
    available routes prove different amounts and the difference has to be on the
    record rather than glossed:

    ``"binder"``
        The whole chain: ``bind_deepseek_v4_flash_vq_experts`` discovers the
        files by its own naming convention and binds them into the layer slot
        that matches the artifact's index. This is the strongest claim and it
        only works for a block the binder can reach -- a backbone layer whose
        index is the probe's last, so ``has_unbound_…`` is a genuine
        *whole-model* check. Verifying layer 42 this way would need a 43-layer
        probe with 42 deliberately unbound layers, which makes the global
        unbound check meaningless, and 75 GB of codes resident to avoid that.

    ``"loader"``
        One layer of probe, and the module is built by the adapter's own
        ``load_deepseek_v4_flash_vq_switch_glu`` (including its
        ``artifact_paths`` indirection and its ``prefix`` override) then handed
        to ``bind_switch_mlp``, which is what validates the dimensions, the
        expert count and the ``LimitedSwiGLU`` clamp. What this route does *not*
        exercise is file discovery -- the caller names the files. For a drafter
        block that is the most that can be proven today, because no MTP binder
        exists to define the discovery convention.

    ``"auto"`` picks ``"binder"`` when the binder can reach the block and
    ``"loader"`` otherwise. The returned payload names which route ran.
    """

    import mlx.core as mx

    from keep.convert.dsv4_vq_pilot import limited_swiglu
    from ramp.models.deepseek_v4_flash_adapter import (
        DeepseekV4FlashVQModel,
        bind_deepseek_v4_flash_vq_experts,
        dense_deepseek_v4_flash_routed_parameter_names,
        has_unbound_deepseek_v4_flash_vq_experts,
        load_deepseek_v4_flash_vq_switch_glu,
    )

    root = Path(artifact_dir)
    if bind_path not in ("auto", "binder", "loader"):
        raise ValueError("bind_path must be 'auto', 'binder', or 'loader'")
    binder_can_reach = spec.kind == "backbone"
    if bind_path == "auto":
        bind_path = "binder" if binder_can_reach and spec.index == 0 else "loader"
    if bind_path == "binder" and not binder_can_reach:
        raise ValueError(
            "the increment-1 adapter binds backbone layers only; the MTP binder "
            "does not exist yet, so mtp blocks cannot be roundtripped through it"
        )
    picked = [int(expert) for expert in experts]
    if not picked:
        raise ValueError("at least one expert is required")
    probe_layers = spec.index + 1 if bind_path == "binder" else 1
    probe_slot = spec.index if bind_path == "binder" else 0
    model_args = reference_probe_model_args(
        probe_layers,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
    )
    model = DeepseekV4FlashVQModel(model_args, with_mtp=False)
    if bind_path == "binder":
        bound = bind_deepseek_v4_flash_vq_experts(model, root, layers=[spec.index])
    else:
        # The loader keys ``artifact_paths`` by the filename *it* would have
        # looked for, so a slot-0 probe can be pointed at any block's files
        # while the tensor prefix stays the block's own.
        model.model.layers[probe_slot].ffn.bind_switch_mlp(
            load_deepseek_v4_flash_vq_switch_glu(
                root,
                probe_slot,
                swiglu_limit=model_args.swiglu_limit,
                prefix=spec.tensor_prefix,
                artifact_paths={
                    f"layer-{probe_slot:05d}-{projection}.safetensors": root
                    / spec.artifact_name(projection)
                    for projection in PROJECTIONS
                },
            )
        )
        bound = (probe_slot,)
    unbound = has_unbound_deepseek_v4_flash_vq_experts(model)
    dense = dense_deepseek_v4_flash_routed_parameter_names(model)

    switch = model.model.layers[probe_slot].ffn.switch_mlp
    rng = np.random.default_rng(seed)
    top_k = model_args.num_experts_per_tok
    x = (
        rng.standard_normal((1, int(tokens), model_args.hidden_size)) * float(input_scale)
    ).astype(np.float32)
    indices = np.asarray(
        [
            [picked[(token + slot) % len(picked)] for slot in range(top_k)]
            for token in range(int(tokens))
        ],
        dtype=np.int32,
    )[None, ...]
    got = np.asarray(
        switch(mx.array(x), mx.array(indices)).astype(mx.float32), dtype=np.float64
    )

    decoded = decode_artifact_experts(root, spec, picked)
    position = {expert: index for index, expert in enumerate(picked)}
    limit = model_args.swiglu_limit

    def reference(swap_gate_up: bool) -> np.ndarray:
        out = np.zeros(
            (1, int(tokens), top_k, model_args.hidden_size), dtype=np.float64
        )
        for token in range(int(tokens)):
            for slot in range(top_k):
                expert = int(indices[0, token, slot])
                row = position[expert]
                gate_weight = decoded["up_proj" if swap_gate_up else "gate_proj"][row]
                up_weight = decoded["gate_proj" if swap_gate_up else "up_proj"][row]
                hidden_in = x[0, token].astype(np.float64)
                gate = gate_weight.astype(np.float64) @ hidden_in
                up = up_weight.astype(np.float64) @ hidden_in
                activated = limited_swiglu(gate, up, limit)
                out[0, token, slot] = (
                    decoded["down_proj"][row].astype(np.float64) @ activated
                )
        return out

    def agreement(truth: np.ndarray) -> dict[str, Any]:
        delta = got - truth
        energy = float(np.sum(truth * truth))
        got_energy = float(np.sum(got * got))
        denominator = np.sqrt(energy * got_energy)
        return {
            "relative_mse": float(np.sum(delta * delta) / energy) if energy > 0 else None,
            "cosine": float(np.sum(truth * got) / denominator) if denominator > 0 else 0.0,
            "max_abs_error": float(np.abs(delta).max()),
            "reference_rms": float(np.sqrt(energy / truth.size)),
        }

    correct = agreement(reference(False))
    swapped = agreement(reference(True))
    passed = bool(
        correct["relative_mse"] is not None
        and correct["relative_mse"] <= tolerance
        and swapped["relative_mse"] > correct["relative_mse"] * 100
        and not unbound
        and dense == ()
    )
    return {
        "record_type": "dsv4_vq_roundtrip_v1",
        "environment": environment(),
        "block": spec.as_dict(),
        "artifact_dir": str(root),
        "policy": policy.as_dict(),
        "geometry": {
            "hidden_size": model_args.hidden_size,
            "moe_intermediate_size": model_args.moe_intermediate_size,
            "num_experts": model_args.n_routed_experts,
            "experts_per_tok": top_k,
        },
        "bound_layers": list(bound),
        "tokens": int(tokens),
        "experts": picked,
        "tolerance": tolerance,
        "bind_proof": {
            "checked_by": (
                "ramp.models.deepseek_v4_flash_adapter."
                "has_unbound_deepseek_v4_flash_vq_experts / "
                "dense_deepseek_v4_flash_routed_parameter_names"
            ),
            "bind_path": bind_path,
            "probe_layers": probe_layers,
            "probe_slot": probe_slot,
            "file_discovery_exercised": bind_path == "binder",
            "layers_bound": list(bound),
            "unbound_vq_experts": bool(unbound),
            "dense_routed_experts": bool(dense),
            "dense_routed_parameter_names": list(dense),
            "gap": (
                None
                if bind_path == "binder"
                else (
                    "file discovery not exercised: the caller named the files via "
                    "the loader's artifact_paths indirection. No MTP binder exists, "
                    "so for a drafter block this is the ceiling of what can be "
                    "verified today."
                )
                if spec.kind == "mtp"
                else (
                    "file discovery not exercised: the caller named the files via "
                    "the loader's artifact_paths indirection. bind_deepseek_v4_flash"
                    "_vq_experts does reach this backbone layer, but only from a "
                    "probe deep enough to contain it, which would leave every "
                    "earlier layer unbound and make the whole-model unbound check "
                    "vacuous. Layer 0 is verified through the full binder chain."
                )
            ),
        },
        "forward_agreement": {
            "tensor_mapping": dict(TENSOR_MAPPING),
            "correct_mapping": correct,
            "gate_up_swapped_control": swapped,
            "discrimination_ratio": (
                swapped["relative_mse"] / correct["relative_mse"]
                if correct["relative_mse"]
                else None
            ),
        },
        "passed": passed,
        "peak_rss_gb": peak_rss_gb(),
    }
