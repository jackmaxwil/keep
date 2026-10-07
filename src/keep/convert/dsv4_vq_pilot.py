"""Wave 5 pilot: group-size policy and VQ-fit schedule for DeepSeek-V4-Flash.

The campaign profile (``models/deepseek-v4-flash-0731.yaml``) still carries
GLM-5.2's placeholder ``group_size_policy`` of 512/512/512. That policy was
derived for a different expert geometry (GLM-5.2's wide experts) and this
module exists to replace it with numbers measured on V4-Flash's own geometry:
256 routed experts per layer, ``moe_intermediate_size`` 2048, ``hidden_size``
4096, so ``w1``/``w3`` are ``[2048, 4096]`` and ``w2`` is ``[4096, 2048]``.

Everything here composes existing KEEP machinery rather than restating it:

* the fit is :func:`keep.convert.glm52_recovery_materialize.quantize_weight_importance_aware`
  -- the same alternating weighted-assignment / exact weighted-least-squares
  scale update the GLM lineage ships;
* the objective is the llama.cpp-style imatrix diagonal, ``sum(activation^2)``
  per input column, as accumulated by ``keep.quality.imatrix`` and captured
  for V4-Flash by the Wave 3 runner;
* the source decode is :func:`keep.convert.fp4_expert.dequantize_fp4_expert`
  (routed experts are FP4-in-I8 with E8M0 group-32 scales -- ``fp8_block``
  cannot read them);
* the checkpoint reads are the Wave 3 runner's own coalesced span index.

Two things this module is careful to keep honest.

**The reference is the FP4 source, not bf16.** DeepSeek released V4-Flash with
its routed experts already at 4 bits. Every e2m1 value times a power-of-two
E8M0 scale is *exactly* representable in bf16 (1 mantissa bit into 8), so
"dequantise to bf16" and "dequantise to fp32" are the same numbers and there is
no second reference hiding on this machine. What cannot be measured locally is
DeepSeek's own bf16-to-FP4 step; :func:`estimate_source_fp4_step` gives a
same-grid estimate of that term so a reader can see how much of the total
distance to the true weights the VQ fit is actually responsible for.

**The quality signal is a proxy and is named as one.** The full model cannot
run here. But the imatrix diagonal *is* the measured per-column second moment
of the real routed activations, so under a diagonal input-covariance model the
imatrix-weighted relative error is exactly the expected relative output MSE of
that projection, and :func:`projection_error` also reports the matching
expected cosine. The single approximation is dropping the off-diagonal input
correlation; the SwiGLU nonlinearity and the routing softmax sit downstream of
it. That is a local proxy, not teacher agreement.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CHECKPOINT_PROJECTIONS",
    "PROJECTION_INPUT_SPACE",
    "RATE_LADDER",
    "GroupSizePolicy",
    "LayerFitTiming",
    "ProjectionErrorMetrics",
    "ScheduleEstimate",
    "aggregate_calibration_importance",
    "artifact_bytes",
    "estimate_source_fp4_step",
    "expert_block_proxy",
    "expert_weight_counts",
    "extrapolate_schedule",
    "fit_projection",
    "limited_swiglu",
    "policy_bpw",
    "projection_error",
    "rate_bpw",
    "reconstruct_quantized",
    "route_weighted_mean",
]

# The checkpoint's own names. w1 = gate, w3 = up, w2 = down.
CHECKPOINT_PROJECTIONS = ("w1", "w2", "w3")

#: Checkpoint tensor -> the profile's projection name -> the imatrix input
#: space the Wave 3 calibration capture accumulated it in.
PROJECTION_ALIAS = {"w1": "gate", "w3": "up", "w2": "down"}
PROJECTION_INPUT_SPACE = {"gate": "hidden", "up": "hidden", "down": "down"}

#: Logical weight shapes for one expert at V4-Flash's geometry, [out, in].
PROJECTION_SHAPE = {
    "gate": (2048, 4096),
    "up": (2048, 4096),
    "down": (4096, 2048),
}

#: fp16 group scales, one per (output row, input group).
SCALE_BITS = 16

#: E8 codeword dimension -- 8 weights per code, for both codebooks.
CODEWORD_DIM = 8


def rate_bpw(code_bits: int, group_size: int, *, scale_bits: int = SCALE_BITS) -> float:
    """Stored bits per weight for one (codebook, group size) point.

    ``code_bits`` is the width of one *codeword* index covering
    ``CODEWORD_DIM`` weights: 8 selects the 256-entry E8-1bit grid (1.0 bpw of
    codes), 16 selects the 2^16-entry E8P grid (2.0 bpw of codes). Group scales
    add ``scale_bits / group_size``. The codebook itself is shared across the
    whole artifact and is accounted separately in :func:`artifact_bytes`.
    """

    if code_bits not in (8, 16):
        raise ValueError(f"code_bits must be 8 or 16, got {code_bits}")
    if group_size <= 0 or group_size % CODEWORD_DIM:
        raise ValueError(
            f"group_size must be positive and divisible by {CODEWORD_DIM}, "
            f"got {group_size}"
        )
    if scale_bits <= 0:
        raise ValueError(f"scale_bits must be positive, got {scale_bits}")
    return code_bits / CODEWORD_DIM + scale_bits / group_size


#: Every rate this pilot can reach with the codebooks and packing KEEP already
#: ships -- no new kernel, no new codebook. Sorted by bpw.
RATE_LADDER: tuple[tuple[int, int], ...] = (
    (8, 512),
    (8, 256),
    (8, 128),
    (8, 64),
    (8, 32),
    (16, 512),
    (16, 256),
    (16, 128),
    (16, 64),
    (16, 32),
    (16, 16),
    (16, 8),
)


@dataclass(frozen=True)
class GroupSizePolicy:
    """A per-projection (code_bits, group_size) assignment.

    The profile's ``group_size_policy`` block is exactly the ``group_size``
    half of this; the pilot carries ``code_bits`` alongside because on this
    geometry the two knobs trade against each other inside one bit budget and
    reporting a group size without its codebook is meaningless.
    """

    gate: tuple[int, int]
    up: tuple[int, int]
    down: tuple[int, int]

    def for_projection(self, projection: str) -> tuple[int, int]:
        try:
            return getattr(self, projection)
        except AttributeError as error:
            raise ValueError(f"unknown projection {projection!r}") from error

    def as_profile_block(self) -> dict[str, Any]:
        """The YAML-shaped block a profile would carry."""

        return {
            "group_size_policy": {
                name: self.for_projection(name)[1] for name in ("gate", "up", "down")
            },
            "code_bits_policy": {
                name: self.for_projection(name)[0] for name in ("gate", "up", "down")
            },
        }


def expert_weight_counts() -> dict[str, int]:
    """Logical weight count of one expert's each projection."""

    return {name: rows * cols for name, (rows, cols) in PROJECTION_SHAPE.items()}


def policy_bpw(policy: GroupSizePolicy, *, scale_bits: int = SCALE_BITS) -> float:
    """Weight-count-weighted mean bpw of a per-projection policy.

    All three projections hold the same 2048*4096 weights on this geometry, so
    the mean is unweighted in practice -- the weighting is written out anyway
    so the function stays correct if a future profile changes a shape.
    """

    counts = expert_weight_counts()
    total = sum(counts.values())
    return sum(
        counts[name] * rate_bpw(*policy.for_projection(name), scale_bits=scale_bits)
        for name in counts
    ) / total


# ---------------------------------------------------------------------------
# Calibration statistics
# ---------------------------------------------------------------------------


def aggregate_calibration_importance(
    calibration_dir: str | Path,
    *,
    layers: Sequence[int],
    session_limit: int | None = None,
) -> dict[str, Any]:
    """Sum the Wave 3 per-session activation statistics for chosen layers.

    The 40 session captures are ``dsv4_teacher_calibration_v1`` records holding
    ``importance_sum__{space}`` of shape ``[layers, experts, input_dim]`` --
    llama.cpp semantics, ``sum(activation^2)`` over every position routed to
    that expert. Summing across sessions is the same merge
    :func:`keep.quality.dsv4_teacher_runner.finalize_dsv4_calibration_imatrix`
    performs; this reads only the requested layer rows so a two-layer pilot
    does not have to write 33,024 sidecar files first.

    Returns a dict with ``importance``/``affinity``/``route_count`` keyed by
    input space, plus the session provenance.
    """

    root = Path(calibration_dir)
    session_files = sorted((root / "sessions").glob("*.npz"))
    if not session_files:
        raise ValueError(f"no session outputs under {root / 'sessions'}")
    if session_limit is not None:
        if session_limit <= 0:
            raise ValueError("session_limit must be positive when given")
        session_files = session_files[:session_limit]

    wanted = [int(layer) for layer in layers]
    if not wanted:
        raise ValueError("at least one layer is required")

    spaces = ("hidden", "down")
    importance: dict[str, np.ndarray] | None = None
    affinity: dict[str, np.ndarray] | None = None
    route_count: dict[str, np.ndarray] | None = None
    rows: dict[int, int] | None = None
    prompt_ids: list[str] = []
    num_experts = 0

    for path in session_files:
        with np.load(path, allow_pickle=False) as archive:
            record = str(np.asarray(archive["record_type"]).reshape(()).item())
            if record != "dsv4_teacher_calibration_v1":
                raise ValueError(f"{path} is not a calibration capture ({record})")
            stored = np.asarray(archive["layers"]).astype(np.int64).tolist()
            if rows is None:
                rows = {layer: index for index, layer in enumerate(stored)}
                missing = [layer for layer in wanted if layer not in rows]
                if missing:
                    raise ValueError(f"calibration capture has no layers {missing}")
                num_experts = int(np.asarray(archive["num_experts"]).reshape(()).item())
            elif rows != {layer: index for index, layer in enumerate(stored)}:
                raise ValueError(f"{path} layer set differs from the first session")
            picked = [rows[layer] for layer in wanted]
            prompt_ids.append(str(np.asarray(archive["prompt_id"]).reshape(()).item()))

            for space in spaces:
                chunk = np.asarray(
                    archive[f"importance_sum__{space}"][picked], dtype=np.float64
                )
                aff = np.asarray(
                    archive[f"affinity_weighted_importance__{space}"][picked],
                    dtype=np.float64,
                )
                counts = np.asarray(
                    archive[f"route_count__{space}"][picked], dtype=np.int64
                )
                if importance is None or affinity is None or route_count is None:
                    importance, affinity, route_count = {}, {}, {}
                if space not in importance:
                    importance[space] = chunk
                    affinity[space] = aff
                    route_count[space] = counts
                else:
                    importance[space] += chunk
                    affinity[space] += aff
                    route_count[space] += counts

    assert importance is not None and affinity is not None and route_count is not None
    return {
        "layers": wanted,
        "num_experts": num_experts,
        "sessions": len(session_files),
        "prompt_ids": tuple(sorted(set(prompt_ids))),
        "importance": importance,
        "affinity_weighted_importance": affinity,
        "route_count": route_count,
    }


def projection_importance(
    stats: Mapping[str, Any],
    *,
    layer: int,
    projection: str,
    expert: int,
    affinity_weighted: bool = False,
) -> np.ndarray:
    """The imatrix diagonal for one (layer, projection, expert), length ``in_dim``."""

    space = PROJECTION_INPUT_SPACE[projection]
    row = list(stats["layers"]).index(int(layer))
    key = "affinity_weighted_importance" if affinity_weighted else "importance"
    vector = np.asarray(stats[key][space][row, int(expert)], dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("importance must be a 1-D per-input-column vector")
    return vector


def route_weighted_mean(values: Sequence[float], route_counts: Sequence[int]) -> float:
    """Average a per-expert quantity by how often the router actually used it.

    An unweighted mean over 256 experts lets the tail experts -- some with
    single-digit route counts across the whole calibration split -- dominate a
    layer-level number they contribute almost nothing to at inference.
    """

    weights = np.asarray(route_counts, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if data.shape != weights.shape:
        raise ValueError("values and route_counts must have the same length")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("route counts sum to zero; nothing was routed")
    return float((data * weights).sum() / total)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectionErrorMetrics:
    """Reconstruction error for one projection, weighted and unweighted.

    ``weighted_relative_mse`` is the headline: with ``d_k`` the measured
    second moment of input column ``k``, it equals the expected relative output
    MSE ``E||(W - W_hat)x||^2 / E||W x||^2`` under a diagonal input covariance.
    ``expected_cosine`` is the matching expected cosine of the output vectors
    under the same model. Both are proxies exactly to the extent that the input
    covariance is not diagonal.
    """

    weighted_relative_mse: float
    weighted_rms: float
    expected_cosine: float
    unweighted_relative_mse: float
    max_abs_error: float

    def as_dict(self) -> dict[str, float]:
        return {
            "weighted_relative_mse": self.weighted_relative_mse,
            "weighted_rms": self.weighted_rms,
            "expected_cosine": self.expected_cosine,
            "unweighted_relative_mse": self.unweighted_relative_mse,
            "max_abs_error": self.max_abs_error,
        }


def projection_error(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    importance: np.ndarray,
) -> ProjectionErrorMetrics:
    """Imatrix-weighted reconstruction error of one ``[out, in]`` projection."""

    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(reconstruction, dtype=np.float64)
    if ref.shape != got.shape:
        raise ValueError(
            f"reference {ref.shape} and reconstruction {got.shape} must match"
        )
    if ref.ndim != 2:
        raise ValueError(f"projections must be 2-D [out, in], got {ref.shape}")
    diag = np.asarray(importance, dtype=np.float64)
    if diag.shape != (ref.shape[1],):
        raise ValueError(
            f"importance must have shape {(ref.shape[1],)}, got {diag.shape}"
        )
    if not np.isfinite(diag).all() or np.any(diag < 0):
        raise ValueError("importance must be finite and non-negative")

    delta = ref - got
    # Column sums first: the [out, in] products never leave one row block.
    err_cols = np.einsum("ok,ok->k", delta, delta)
    ref_cols = np.einsum("ok,ok->k", ref, ref)
    got_cols = np.einsum("ok,ok->k", got, got)
    cross_cols = np.einsum("ok,ok->k", ref, got)

    weighted_err = float(diag @ err_cols)
    weighted_ref = float(diag @ ref_cols)
    weighted_got = float(diag @ got_cols)
    weighted_cross = float(diag @ cross_cols)
    if weighted_ref <= 0:
        raise ValueError("weighted reference energy is zero; importance or W is empty")

    denominator = np.sqrt(weighted_ref * weighted_got)
    cosine = float(weighted_cross / denominator) if denominator > 0 else 0.0
    unweighted_ref = float(ref_cols.sum())
    return ProjectionErrorMetrics(
        weighted_relative_mse=weighted_err / weighted_ref,
        weighted_rms=float(np.sqrt(weighted_err / (diag.sum() * ref.shape[0]))),
        expected_cosine=cosine,
        unweighted_relative_mse=float(err_cols.sum() / unweighted_ref)
        if unweighted_ref > 0
        else 0.0,
        max_abs_error=float(np.abs(delta).max()),
    )


def reconstruct_quantized(quantized: Any) -> np.ndarray:
    """Decode an ``ImportanceAwareQuantizedWeight`` back to float32 ``[out, in]``."""

    from keep.vq.e8 import e8_1bit_grid, e8p_full_grid

    table = (
        e8_1bit_grid().astype(np.float32)
        if quantized.code_bits == 8
        else e8p_full_grid().astype(np.float32)
    )
    codes = np.asarray(quantized.codes)
    out_dim = codes.shape[0]
    in_dim = codes.shape[1] * CODEWORD_DIM
    group_size = int(quantized.group_size)
    group_count = in_dim // group_size
    decoded = table[codes.reshape(out_dim, group_count, group_size // CODEWORD_DIM)]
    scales = np.asarray(quantized.scales, dtype=np.float32)
    return (decoded * scales[:, :, None, None]).reshape(out_dim, in_dim)


def estimate_source_fp4_step(
    decoded_source: np.ndarray, *, group_size: int = 32, rng_seed: int = 0
) -> dict[str, float]:
    """Estimate the error DeepSeek's own bf16 -> FP4 step already spent.

    The released weight is already on the e2m1 x 2^E8M0 grid, so re-encoding it
    measures nothing (it round-trips exactly, which this function asserts by
    reporting ``roundtrip_relative_mse``). What *can* be estimated is the error
    that grid costs a weight which was **not** already on it: draw a Gaussian
    matrix whose per-group RMS matches the real weight's, quantise it through
    the same FP4/group-32 encoder, and report the relative error. It is a
    same-statistics synthetic, not a measurement of the actual pre-quantisation
    weights -- those were never released -- and the report says so.
    """

    values = np.asarray(decoded_source, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("decoded_source must be 2-D [out, in]")
    if values.shape[1] % group_size:
        raise ValueError("input dim must be a multiple of group_size")

    roundtrip = _fp4_group_quantize(values, group_size=group_size)
    roundtrip_rel = float(
        np.sum((values - roundtrip) ** 2) / max(np.sum(values.astype(np.float64) ** 2), 1e-30)
    )

    grouped = values.reshape(values.shape[0], -1, group_size)
    rms = np.sqrt(np.mean(grouped.astype(np.float64) ** 2, axis=2, keepdims=True))
    rng = np.random.default_rng(rng_seed)
    synthetic = (rng.standard_normal(grouped.shape) * rms).astype(np.float32)
    synthetic = synthetic.reshape(values.shape)
    quantized = _fp4_group_quantize(synthetic, group_size=group_size)
    synthetic_rel = float(
        np.sum((synthetic - quantized).astype(np.float64) ** 2)
        / max(np.sum(synthetic.astype(np.float64) ** 2), 1e-30)
    )
    return {
        "roundtrip_relative_mse": roundtrip_rel,
        "synthetic_gaussian_relative_mse": synthetic_rel,
        "group_size": float(group_size),
    }


_E2M1_MAGNITUDES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32
)


def _fp4_group_quantize(values: np.ndarray, *, group_size: int) -> np.ndarray:
    """Round-to-nearest e2m1 under a per-group power-of-two (E8M0) scale."""

    data = np.asarray(values, dtype=np.float32)
    grouped = data.reshape(data.shape[0], -1, group_size)
    amax = np.max(np.abs(grouped), axis=2, keepdims=True)
    # E8M0: the scale is a power of two, chosen so the group max lands at or
    # below the codebook's 6.0 ceiling -- the release's own rule.
    with np.errstate(divide="ignore"):
        exponent = np.where(amax > 0, np.ceil(np.log2(amax / 6.0)), 0.0)
    scale = np.exp2(exponent).astype(np.float32)
    normalized = grouped / scale
    magnitude = np.abs(normalized)
    index = np.abs(magnitude[..., None] - _E2M1_MAGNITUDES).argmin(axis=-1)
    rounded = _E2M1_MAGNITUDES[index] * np.sign(normalized)
    return (rounded * scale).reshape(data.shape)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def limited_swiglu(gate: np.ndarray, up: np.ndarray, limit: float) -> np.ndarray:
    """DeepSeek-V4's clamped SwiGLU, in numpy.

    Mirrors ``ramp.models.deepseek_v4_flash_adapter._limited_swiglu``: the gate
    is clamped from above only, the up branch on both sides, and the release
    sets ``limit`` to 10.0.
    """

    gate_values = np.asarray(gate, dtype=np.float64)
    up_values = np.asarray(up, dtype=np.float64)
    if limit and limit > 0:
        gate_values = np.minimum(gate_values, limit)
        up_values = np.clip(up_values, -limit, limit)
    silu = gate_values / (1.0 + np.exp(-gate_values))
    return silu * up_values


def expert_block_proxy(
    reference: Mapping[str, np.ndarray],
    reconstruction: Mapping[str, np.ndarray],
    *,
    column_sigma: np.ndarray,
    tokens: int = 512,
    swiglu_limit: float = 10.0,
    seed: int = 0,
) -> dict[str, float]:
    """Whole-expert output error through the real SwiGLU, on synthetic inputs.

    The per-projection metric is linear and therefore cannot see what the
    nonlinearity does to the error. This pushes ``tokens`` synthetic hidden
    states through both the reference expert and its VQ reconstruction --
    ``down(silu(min(gate(x), L)) * clip(up(x), -L, L))`` -- and reports the
    relative output MSE and cosine of the expert's actual contribution.

    ``column_sigma`` is the per-hidden-column RMS activation measured by the
    Wave 3 calibration run (``sqrt(importance_sum / route_count)``), so the
    input has the right *scale* per channel. It does **not** have the right
    *shape*: real hidden states are heavy-tailed with correlated outlier
    channels, and a diagonal Gaussian has neither property. This is a stronger
    proxy than the linear one and still a proxy.
    """

    sigma = np.asarray(column_sigma, dtype=np.float64)
    if sigma.ndim != 1:
        raise ValueError("column_sigma must be 1-D over hidden columns")
    if not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError("column_sigma must be finite and non-negative")
    missing = {"gate", "up", "down"} - set(reference) | {"gate", "up", "down"} - set(
        reconstruction
    )
    if missing:
        raise ValueError(f"both weight sets need gate/up/down, missing {sorted(missing)}")
    if tokens <= 0:
        raise ValueError("tokens must be positive")

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((tokens, sigma.shape[0])) * sigma

    def forward(weights: Mapping[str, np.ndarray]) -> np.ndarray:
        gate = x @ np.asarray(weights["gate"], dtype=np.float64).T
        up = x @ np.asarray(weights["up"], dtype=np.float64).T
        hidden = limited_swiglu(gate, up, swiglu_limit)
        return hidden @ np.asarray(weights["down"], dtype=np.float64).T

    truth = forward(reference)
    got = forward(reconstruction)
    delta = truth - got
    truth_energy = float(np.sum(truth * truth))
    if truth_energy <= 0:
        raise ValueError("reference expert produced zero output")
    got_energy = float(np.sum(got * got))
    denominator = np.sqrt(truth_energy * got_energy)
    return {
        "block_relative_mse": float(np.sum(delta * delta) / truth_energy),
        "block_cosine": float(np.sum(truth * got) / denominator)
        if denominator > 0
        else 0.0,
        "tokens": float(tokens),
    }


def fit_projection(
    weight: np.ndarray,
    importance: np.ndarray,
    *,
    group_size: int,
    code_bits: int,
    iterations: int = 3,
    backend: str = "metal",
) -> tuple[Any, ProjectionErrorMetrics, float]:
    """Fit one projection and measure it. Returns (quantized, metrics, seconds)."""

    from keep.convert.glm52_recovery_materialize import (
        quantize_weight_importance_aware,
    )

    values = np.ascontiguousarray(np.asarray(weight, dtype=np.float32))
    diag = np.ascontiguousarray(np.asarray(importance, dtype=np.float32))
    started = time.perf_counter()
    quantized = quantize_weight_importance_aware(
        values,
        diag,
        group_size=group_size,
        code_bits=code_bits,
        iterations=iterations,
        e8p_search_backend=backend,
    )
    elapsed = time.perf_counter() - started
    metrics = projection_error(values, reconstruct_quantized(quantized), diag)
    return quantized, metrics, elapsed


# ---------------------------------------------------------------------------
# Schedule extrapolation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerFitTiming:
    """Measured per-layer cost of one materialization pass."""

    layer: int
    projections_fitted: int
    read_seconds: float
    decode_seconds: float
    fit_seconds: float
    peak_rss_gb: float

    @property
    def total_seconds(self) -> float:
        return self.read_seconds + self.decode_seconds + self.fit_seconds

    def scaled_to_full_layer(self, projections_per_layer: int) -> LayerFitTiming:
        """Scale a sampled-expert measurement up to a whole layer."""

        if self.projections_fitted <= 0:
            raise ValueError("projections_fitted must be positive")
        factor = projections_per_layer / self.projections_fitted
        return LayerFitTiming(
            layer=self.layer,
            projections_fitted=projections_per_layer,
            # The read is one coalesced whole-layer span whatever the sample,
            # so it does not scale with the number of experts fitted.
            read_seconds=self.read_seconds,
            decode_seconds=self.decode_seconds * factor,
            fit_seconds=self.fit_seconds * factor,
            peak_rss_gb=self.peak_rss_gb,
        )


@dataclass(frozen=True)
class ScheduleEstimate:
    """A full-materialization projection with every assumed term named."""

    label: str
    bpw: float
    layers: int
    projections_per_layer: int
    seconds_per_layer: float
    total_seconds: float
    routed_bytes: float
    resident_bytes: float
    artifact_bytes: float
    peak_rss_gb: float
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_hours(self) -> float:
        return self.total_seconds / 3600.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bpw": round(self.bpw, 4),
            "layers": self.layers,
            "projections_per_layer": self.projections_per_layer,
            "seconds_per_layer": round(self.seconds_per_layer, 2),
            "total_hours": round(self.total_hours, 2),
            "routed_gb": round(self.routed_bytes / 1e9, 2),
            "resident_gb": round(self.resident_bytes / 1e9, 2),
            "artifact_gb": round(self.artifact_bytes / 1e9, 2),
            "peak_rss_gb": round(self.peak_rss_gb, 2),
            "assumptions": list(self.assumptions),
        }


def artifact_bytes(
    *,
    bpw: float,
    routed_weight_count: int,
    resident_weight_count: int,
    resident_bpw: float,
    codebook_bytes: int = 0,
) -> dict[str, float]:
    """Split an artifact's size into routed payload, residents, and codebook."""

    if bpw <= 0 or resident_bpw <= 0:
        raise ValueError("bit rates must be positive")
    if routed_weight_count <= 0 or resident_weight_count < 0:
        raise ValueError("weight counts must be non-negative and routed must be positive")
    routed = routed_weight_count * bpw / 8.0
    resident = resident_weight_count * resident_bpw / 8.0
    return {
        "routed_bytes": routed,
        "resident_bytes": resident,
        "codebook_bytes": float(codebook_bytes),
        "total_bytes": routed + resident + codebook_bytes,
    }


def extrapolate_schedule(
    timings: Sequence[LayerFitTiming],
    *,
    label: str,
    bpw: float,
    layers: int,
    projections_per_layer: int,
    routed_weight_count: int,
    resident_weight_count: int,
    resident_bpw: float,
    codebook_bytes: int = 0,
    assumptions: Sequence[str] = (),
) -> ScheduleEstimate:
    """Project a full materialization from one or more measured layers.

    The per-layer cost is the mean of the supplied measurements. Nothing here
    models a speed-up from batching layers or a slow-down from thermal drift;
    both are named in ``assumptions`` by the caller rather than folded in
    silently.
    """

    if not timings:
        raise ValueError("at least one measured layer is required")
    if layers <= 0:
        raise ValueError("layers must be positive")
    scaled = [timing.scaled_to_full_layer(projections_per_layer) for timing in timings]
    seconds_per_layer = float(np.mean([timing.total_seconds for timing in scaled]))
    sizes = artifact_bytes(
        bpw=bpw,
        routed_weight_count=routed_weight_count,
        resident_weight_count=resident_weight_count,
        resident_bpw=resident_bpw,
        codebook_bytes=codebook_bytes,
    )
    return ScheduleEstimate(
        label=label,
        bpw=bpw,
        layers=layers,
        projections_per_layer=projections_per_layer,
        seconds_per_layer=seconds_per_layer,
        total_seconds=seconds_per_layer * layers,
        routed_bytes=sizes["routed_bytes"],
        resident_bytes=sizes["resident_bytes"],
        artifact_bytes=sizes["total_bytes"],
        peak_rss_gb=float(max(timing.peak_rss_gb for timing in scaled)),
        assumptions=tuple(assumptions),
    )


# ---------------------------------------------------------------------------
# Checkpoint access
# ---------------------------------------------------------------------------


def read_layer_experts(
    checkpoint_dir: str | Path,
    layer: int,
    *,
    experts: Sequence[int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Read one layer's packed routed-expert bytes off the shards.

    Reuses the Wave 3 runner's coalesced span index so the pilot and the
    teacher run agree on where every expert lives. Returns
    ``(weights, scales, seconds)`` with per-projection stacks
    ``[num_experts, ...]`` of raw ``uint8`` -- decoding is the caller's step so
    it can be timed separately.
    """

    from keep.quality.dsv4_teacher_runner import (
        _pread_exact,
        build_dsv4_expert_span_index,
    )

    span = build_dsv4_expert_span_index(checkpoint_dir, layers=[int(layer)])[int(layer)]
    wanted = None if experts is None else {int(index) for index in experts}
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
    fd = os.open(path, os.O_RDONLY)
    try:
        for projection, kind, expert, offset, length in span.reads:
            if wanted is not None and expert not in wanted:
                continue
            table = weights if kind == "weight" else scales
            _pread_exact(fd, table[projection][expert], offset, length)
    finally:
        os.close(fd)
    return weights, scales, time.perf_counter() - started


def decode_expert(
    weights: Mapping[str, np.ndarray],
    scales: Mapping[str, np.ndarray],
    *,
    projection: str,
    expert: int,
) -> np.ndarray:
    """Dequantize one expert's projection from the packed layer stacks."""

    from keep.convert.fp4_expert import dequantize_fp4_expert

    checkpoint_name = {"gate": "w1", "up": "w3", "down": "w2"}[projection]
    return dequantize_fp4_expert(
        weights[checkpoint_name][int(expert)],
        scales[checkpoint_name][int(expert)],
        group_size=32,
    )


def stratified_experts(route_counts: Sequence[int], count: int) -> list[int]:
    """Pick ``count`` experts spanning the route-count distribution.

    Route counts on this model span three orders of magnitude within one layer
    (measured: 9 to 21,731 on layer 5 of one session), so a contiguous or
    random slice of experts is not representative of the layer. This sorts by
    route count and takes evenly spaced ranks, always including the busiest and
    the quietest.
    """

    counts = np.asarray(route_counts, dtype=np.int64)
    if counts.ndim != 1:
        raise ValueError("route_counts must be 1-D")
    total = counts.shape[0]
    if count <= 0 or count > total:
        raise ValueError(f"count must be in [1, {total}], got {count}")
    order = np.argsort(counts, kind="stable")
    ranks = np.linspace(0, total - 1, count).round().astype(int)
    return sorted(int(order[rank]) for rank in dict.fromkeys(ranks.tolist()))


def load_profile_geometry(profile_path: str | Path) -> dict[str, Any]:
    """Read the campaign profile's structural facts without a YAML dependency.

    The profile is hand-written flat YAML; the pilot needs six scalars from it
    and pulling in a parser for that is more surface than it is worth. Anything
    unparsable raises rather than defaulting.
    """

    text = Path(profile_path).read_text()
    wanted = {
        "num_layers": int,
        "num_sparse_layers": int,
        "hidden_size": int,
        "moe_intermediate_size": int,
        "num_experts": int,
        "experts_per_tok": int,
    }
    found: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line or line.startswith((" ", "#", "-")):
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key in wanted:
            found[key] = wanted[key](value.split("#")[0].strip())
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise ValueError(f"profile {profile_path} is missing {missing}")
    return found


def write_json(path: str | Path, payload: Any) -> None:
    """Atomically write a pilot artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
