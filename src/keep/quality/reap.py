"""REAP expert saliency from the real routed forward, and the prune it drives.

REAP (Router-weighted Expert Activation Pruning) scores expert ``i`` by the
mean, over the tokens actually routed to it, of ``g * ||f_i(x)||_2``: the gate
weight that combined its output times the L2 norm of that output before
gating. Experts with the lowest score are dropped and the router re-selects
top-k among the survivors.

This module computes that score from what KEEP's teacher runner already has
in hand -- the model's own gate (bias-corrected selection, the family's own
weight function and renormalisation) and the per-route expert outputs -- so it
costs a norm and a few scatter-adds per layer instead of REAP's dense pass,
which runs every expert on every token. Two differences from the upstream
implementation are deliberate:

- **The real gate.** Upstream computes ``softmax`` of raw router logits and
  picks top-k without the correction bias. For a sqrtsoftplus/noaux router
  (DeepSeek-V4, V4.1) that is a different routing distribution; the score
  here uses the weights the model actually used.
- **Routed tokens only.** Saliency never needs an unrouted expert's output.

The statistics also support the baselines a kill test compares against
(route frequency, gate-weighted frequency, mean output norm) and the
"super-expert" outlier screen, which here uses ``max |f|`` so negative
outliers are not invisible.

Pruning is uniform: one surviving expert count for every layer, because RAMP
binds a single ``n_routed_experts``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

__all__ = [
    "REAP_ARRAY_PREFIX",
    "ReapAccumulator",
    "ReapStats",
    "SALIENCY_CRITERIA",
    "expert_saliency",
    "prune_checkpoint_plan",
    "pruned_set_jaccard",
    "select_survivors",
    "super_expert_candidates",
    "write_survivors",
]

REAP_ARRAY_PREFIX = "reap__"
SALIENCY_CRITERIA = ("reap", "frequency", "weighted_frequency", "ean_mean")
Criterion = Literal["reap", "frequency", "weighted_frequency", "ean_mean"]

_STAT_FIELDS = (
    "route_count",  # int64 [L, E]: routed slots per expert
    "gate_sum",  # float64 [L, E]: sum of gate weights
    "gate_norm_sum",  # float64 [L, E]: sum of g * ||f||  (the REAP numerator)
    "norm_sum",  # float64 [L, E]: sum of ||f||
    "norm_max",  # float64 [L, E]: max ||f|| over routed slots
    "abs_max",  # float64 [L, E]: max over routed slots of max_h |f_h|
)


@dataclass
class ReapStats:
    """Summed REAP statistics for one or more sessions."""

    layers: np.ndarray  # int64 [L]
    num_experts: int
    route_count: np.ndarray
    gate_sum: np.ndarray
    gate_norm_sum: np.ndarray
    norm_sum: np.ndarray
    norm_max: np.ndarray
    abs_max: np.ndarray
    routed_slots: np.ndarray  # int64 [L]: total routed slots per layer
    sessions: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.layers = np.asarray(self.layers, dtype=np.int64)
        n = self.layers.size
        shape = (n, int(self.num_experts))
        self.route_count = np.asarray(self.route_count, dtype=np.int64)
        for name in _STAT_FIELDS[1:]:
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        self.routed_slots = np.asarray(self.routed_slots, dtype=np.int64)
        for name in _STAT_FIELDS:
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"REAP {name} has shape {value.shape}, expected {shape}")
            if not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"REAP {name} must be finite and non-negative")
        if self.routed_slots.shape != (n,):
            raise ValueError("REAP routed_slots must be one count per layer")
        if np.any(self.route_count.sum(axis=1) != self.routed_slots):
            raise ValueError("REAP route counts do not sum to the routed slots")
        if np.any((self.route_count == 0) & (self.gate_norm_sum != 0)):
            raise ValueError("REAP has gate mass on an expert with no routed slots")

    # -- persistence -------------------------------------------------------

    def as_arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            f"{REAP_ARRAY_PREFIX}layers": self.layers,
            f"{REAP_ARRAY_PREFIX}num_experts": np.asarray(self.num_experts, dtype=np.int64),
            f"{REAP_ARRAY_PREFIX}routed_slots": self.routed_slots,
        }
        for name in _STAT_FIELDS:
            arrays[f"{REAP_ARRAY_PREFIX}{name}"] = getattr(self, name)
        return arrays

    @classmethod
    def from_arrays(
        cls, arrays: Mapping[str, Any], *, sessions: Sequence[str] = ()
    ) -> "ReapStats":
        key = REAP_ARRAY_PREFIX.__add__
        missing = [
            name
            for name in ("layers", "num_experts", "routed_slots", *_STAT_FIELDS)
            if key(name) not in arrays
        ]
        if missing:
            raise KeyError("REAP arrays missing: " + ", ".join(missing))
        return cls(
            layers=np.asarray(arrays[key("layers")]),
            num_experts=int(np.asarray(arrays[key("num_experts")]).reshape(())),
            routed_slots=np.asarray(arrays[key("routed_slots")]),
            sessions=tuple(sessions),
            **{name: np.asarray(arrays[key(name)]) for name in _STAT_FIELDS},
        )

    def merged(self, other: "ReapStats") -> "ReapStats":
        """Sum two disjoint sets of sessions over the same layers and experts."""

        if not np.array_equal(self.layers, other.layers) or self.num_experts != other.num_experts:
            raise ValueError("REAP stats cover different layers or expert counts")
        overlap = set(self.sessions) & set(other.sessions)
        if overlap:
            raise ValueError(f"REAP stats would double-count sessions: {sorted(overlap)[:5]}")
        return ReapStats(
            layers=self.layers,
            num_experts=self.num_experts,
            route_count=self.route_count + other.route_count,
            gate_sum=self.gate_sum + other.gate_sum,
            gate_norm_sum=self.gate_norm_sum + other.gate_norm_sum,
            norm_sum=self.norm_sum + other.norm_sum,
            norm_max=np.maximum(self.norm_max, other.norm_max),
            abs_max=np.maximum(self.abs_max, other.abs_max),
            routed_slots=self.routed_slots + other.routed_slots,
            sessions=self.sessions + other.sessions,
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp.npz")
        np.savez(tmp, **self.as_arrays(), **{f"{REAP_ARRAY_PREFIX}sessions": np.asarray(self.sessions, dtype=str)})
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "ReapStats":
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        sessions = tuple(str(s) for s in arrays.pop(f"{REAP_ARRAY_PREFIX}sessions", np.asarray([], dtype=str)))
        return cls.from_arrays(arrays, sessions=sessions)


class ReapAccumulator:
    """Device scatter-add of REAP statistics; folds into float64 once per layer.

    Same design as the teacher runner's imatrix accumulator: float32 sums on
    device for one layer's chunks, then a float64 host fold, so cross-chunk and
    cross-session accumulation carries no float32 drift beyond one layer.
    """

    def __init__(self, *, layers: Sequence[int], num_experts: int) -> None:
        self.layers = tuple(int(layer) for layer in layers)
        self.num_experts = int(num_experts)
        self._row = {layer: row for row, layer in enumerate(self.layers)}
        shape = (len(self.layers), self.num_experts)
        self._host = {name: np.zeros(shape, dtype=np.float64) for name in _STAT_FIELDS}
        self._routed_slots = np.zeros(len(self.layers), dtype=np.int64)
        self._device: dict[int, dict[str, Any]] = {}

    def add_chunk(
        self, *, layer: int, expert_ids: Any, gate_weights: Any, expert_outputs: Any
    ) -> None:
        """Add one chunk's routed slots.

        ``expert_ids`` and ``gate_weights`` are ``[rows, top_k]``;
        ``expert_outputs`` is ``[rows, top_k, hidden]``, each expert's output
        for that slot *before* the gate weight multiplies it.
        """

        import mlx.core as mx

        if layer not in self._row:
            raise KeyError(f"layer {layer} is not tracked by this accumulator")
        if expert_outputs.ndim != 3 or tuple(expert_outputs.shape[:2]) != tuple(expert_ids.shape):
            raise ValueError(
                f"expert outputs {tuple(expert_outputs.shape)} do not match routes "
                f"{tuple(expert_ids.shape)}"
            )
        outputs = expert_outputs.astype(mx.float32)
        norms = mx.sqrt(mx.sum(outputs * outputs, axis=-1)).reshape(-1)
        abs_max = mx.max(mx.abs(outputs), axis=-1).reshape(-1)
        ids = expert_ids.reshape(-1).astype(mx.int32)
        gates = gate_weights.reshape(-1).astype(mx.float32)
        bucket = self._device.get(layer)
        if bucket is None:
            zeros = lambda: mx.zeros((self.num_experts,), dtype=mx.float32)  # noqa: E731
            bucket = {name: zeros() for name in _STAT_FIELDS}
            bucket["slots"] = 0
            self._device[layer] = bucket
        bucket["route_count"] = bucket["route_count"].at[ids].add(mx.ones_like(norms))
        bucket["gate_sum"] = bucket["gate_sum"].at[ids].add(gates)
        bucket["gate_norm_sum"] = bucket["gate_norm_sum"].at[ids].add(gates * norms)
        bucket["norm_sum"] = bucket["norm_sum"].at[ids].add(norms)
        bucket["norm_max"] = bucket["norm_max"].at[ids].maximum(norms)
        bucket["abs_max"] = bucket["abs_max"].at[ids].maximum(abs_max)
        bucket["slots"] += int(ids.size)

    def eval_pending(self) -> None:
        import mlx.core as mx

        for bucket in self._device.values():
            mx.eval([bucket[name] for name in _STAT_FIELDS])

    def flush_device(self) -> None:
        import mlx.core as mx

        for layer, bucket in self._device.items():
            mx.eval([bucket[name] for name in _STAT_FIELDS])
            row = self._row[layer]
            for name in ("gate_sum", "gate_norm_sum", "norm_sum"):
                self._host[name][row] += np.asarray(bucket[name], dtype=np.float64)
            # float32 scatter-added ones are exact small integers; rint makes a
            # future non-integral weighting fail loudly at the int cast below.
            self._host["route_count"][row] += np.rint(np.asarray(bucket["route_count"], dtype=np.float64))
            for name in ("norm_max", "abs_max"):
                self._host[name][row] = np.maximum(
                    self._host[name][row], np.asarray(bucket[name], dtype=np.float64)
                )
            self._routed_slots[row] += int(bucket["slots"])
        self._device.clear()

    def stats(self, *, session: str | None = None) -> ReapStats:
        self.flush_device()
        return ReapStats(
            layers=np.asarray(self.layers, dtype=np.int64),
            num_experts=self.num_experts,
            route_count=self._host["route_count"].astype(np.int64),
            gate_sum=self._host["gate_sum"],
            gate_norm_sum=self._host["gate_norm_sum"],
            norm_sum=self._host["norm_sum"],
            norm_max=self._host["norm_max"],
            abs_max=self._host["abs_max"],
            routed_slots=self._routed_slots.copy(),
            sessions=() if session is None else (session,),
        )

    def as_arrays(self) -> dict[str, np.ndarray]:
        return self.stats().as_arrays()


def expert_saliency(
    stats: ReapStats, criterion: Criterion = "reap", *, prior_slots: float = 0.0
) -> np.ndarray:
    """Per-(layer, expert) saliency under ``criterion``; higher means keep.

    - ``reap``: mean of ``g * ||f||`` over routed slots (REAP's score).
    - ``frequency``: routed slot count.
    - ``weighted_frequency``: sum of gate weights.
    - ``ean_mean``: mean output norm over routed slots, ignoring the gate.

    An expert with no routed slots scores 0 under every criterion, so it is
    pruned first, as upstream does. ``prior_slots > 0`` shrinks the mean-type
    scores toward the layer mean as if each expert had seen that many extra
    average slots, which damps the noise of rarely routed experts; 0 (the
    default) is REAP exactly.
    """

    if criterion not in SALIENCY_CRITERIA:
        raise ValueError(f"criterion must be one of {SALIENCY_CRITERIA}, got {criterion!r}")
    if prior_slots < 0:
        raise ValueError("prior_slots must be non-negative")
    count = stats.route_count.astype(np.float64)
    if criterion == "frequency":
        return count
    if criterion == "weighted_frequency":
        return stats.gate_sum.copy()
    numerator = stats.gate_norm_sum if criterion == "reap" else stats.norm_sum
    with np.errstate(invalid="ignore", divide="ignore"):
        if prior_slots == 0:
            return np.where(count > 0, numerator / np.maximum(count, 1.0), 0.0)
        layer_mean = numerator.sum(axis=1, keepdims=True) / np.maximum(
            count.sum(axis=1, keepdims=True), 1.0
        )
        shrunk = (numerator + prior_slots * layer_mean) / (count + prior_slots)
        return np.where(count > 0, shrunk, 0.0)


def select_survivors(saliency: np.ndarray, keep: int) -> np.ndarray:
    """Keep the ``keep`` most salient experts in every layer.

    Returns ``int64 [L, keep]`` expert ids, ascending within each layer, which
    is the order the sliced router rows and renumbered experts follow. Ties
    break toward the lower expert id, so the result is deterministic.
    """

    saliency = np.asarray(saliency, dtype=np.float64)
    if saliency.ndim != 2:
        raise ValueError("saliency must be [layers, experts]")
    layers, experts = saliency.shape
    if not 0 < int(keep) <= experts:
        raise ValueError(f"keep must be in 1..{experts}, got {keep}")
    if not np.all(np.isfinite(saliency)):
        raise ValueError("saliency must be finite")
    survivors = np.empty((layers, int(keep)), dtype=np.int64)
    ids = np.arange(experts)
    for row in range(layers):
        order = np.lexsort((ids, -saliency[row]))  # by saliency desc, then id asc
        survivors[row] = np.sort(order[: int(keep)])
    return survivors


def pruned_set_jaccard(a: np.ndarray, b: np.ndarray, *, num_experts: int) -> np.ndarray:
    """Per-layer Jaccard index of the *pruned* sets implied by two survivor lists."""

    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("survivor arrays must share a [layers, keep] shape")
    out = np.empty(a.shape[0], dtype=np.float64)
    universe = set(range(int(num_experts)))
    for row in range(a.shape[0]):
        pruned_a = universe - set(a[row].tolist())
        pruned_b = universe - set(b[row].tolist())
        union = pruned_a | pruned_b
        out[row] = 1.0 if not union else len(pruned_a & pruned_b) / len(union)
    return out


def super_expert_candidates(
    stats: ReapStats,
    *,
    percentile: float = 99.5,
    fraction_of_max: float = 0.1,
    layer_fraction: float = 0.75,
) -> list[tuple[int, int]]:
    """``(layer, expert)`` pairs whose largest routed output is an outlier.

    REAP's rule on ``max |f|`` rather than the signed maximum: an expert is a
    candidate if its peak exceeds both the ``percentile`` of all peaks and
    ``fraction_of_max`` of the global peak, within the first ``layer_fraction``
    of layers. Upstream starts its signed maximum at 0, so negative outliers
    never register; this screen sees them.
    """

    peaks = stats.abs_max
    if peaks.size == 0:
        return []
    threshold = max(float(np.percentile(peaks, percentile)), float(peaks.max()) * fraction_of_max)
    limit = int(np.ceil(len(stats.layers) * layer_fraction))
    hits = np.argwhere(peaks[:limit] > threshold)
    return [(int(stats.layers[row]), int(expert)) for row, expert in hits]


_ROUTER_RE = re.compile(r"^(?P<prefix>(?:layers|mtp)\.(?P<idx>\d+))\.ffn\.gate\.(?P<leaf>weight|bias|bias_vl|tid2eid)$")
_EXPERT_RE = re.compile(
    r"^(?P<prefix>(?:layers|mtp)\.(?P<idx>\d+))\.ffn\.experts\.(?P<expert>\d+)\.(?P<rest>.+)$"
)


def prune_checkpoint_plan(
    names: Iterable[str],
    survivors: Mapping[int, Sequence[int]],
    *,
    prefix: str = "layers",
) -> dict[str, Any]:
    """Plan a uniform prune over DeepSeek-V4-style checkpoint tensor names.

    ``survivors`` maps a block index under ``prefix`` (``layers`` for the
    backbone, ``mtp`` for drafter stages) to the expert ids it keeps. Returns:

    - ``rename``: ``{source_name: target_name}`` for every kept routed-expert
      tensor, renumbered ``0..keep-1`` in ascending source order;
    - ``slice_rows``: ``{router_tensor_name: [kept ids]}`` for the gate's
      ``weight``, ``bias`` and ``bias_vl`` (the image-token bias V4.1 adds,
      which must be sliced with the rest or vision tokens route to the wrong
      experts);
    - ``drop``: expert tensors of pruned experts;
    - ``untouched``: every other name, including blocks not in ``survivors``.

    Hash-routed blocks (``gate.tid2eid``) are refused: their token-to-expert
    table names fixed experts and needs a remapping policy this plan does not
    invent. DeepSeek-V4.1 has none; V4-Flash has three.
    """

    keep_sets: dict[int, list[int]] = {}
    width = None
    for block, kept in survivors.items():
        kept_sorted = sorted(int(e) for e in kept)
        if len(set(kept_sorted)) != len(kept_sorted):
            raise ValueError(f"block {block} keeps a duplicate expert")
        if width is None:
            width = len(kept_sorted)
        elif len(kept_sorted) != width:
            raise ValueError("every pruned block must keep the same expert count")
        keep_sets[int(block)] = kept_sorted

    rename: dict[str, str] = {}
    slice_rows: dict[str, list[int]] = {}
    drop: list[str] = []
    untouched: list[str] = []
    seen_experts: dict[int, set[int]] = {block: set() for block in keep_sets}
    for name in sorted(names):
        router = _ROUTER_RE.match(name)
        expert = _EXPERT_RE.match(name)
        match = router or expert
        if match is None or not match.group("prefix").startswith(prefix + "."):
            untouched.append(name)
            continue
        block = int(match.group("idx"))
        if block not in keep_sets:
            untouched.append(name)
            continue
        kept = keep_sets[block]
        if router is not None:
            if router.group("leaf") == "tid2eid":
                raise NotImplementedError(
                    f"{name}: hash-routed block {block} needs a token-to-expert "
                    "remapping policy before it can be pruned"
                )
            slice_rows[name] = list(kept)
            continue
        old = int(expert.group("expert"))
        seen_experts[block].add(old)
        if old in kept:
            new = kept.index(old)
            rename[name] = f"{expert.group('prefix')}.ffn.experts.{new}.{expert.group('rest')}"
        else:
            drop.append(name)
    for block, kept in keep_sets.items():
        absent = sorted(set(kept) - seen_experts[block])
        if absent:
            raise ValueError(f"block {block} keeps experts with no tensors: {absent[:10]}")
    return {
        "record_type": "keep_reap_prune_plan_v1",
        "prefix": prefix,
        "keep": width,
        "survivors": {str(block): kept for block, kept in sorted(keep_sets.items())},
        "rename": rename,
        "slice_rows": slice_rows,
        "drop": drop,
        "untouched": untouched,
    }


def write_survivors(path: str | Path, survivors: np.ndarray, *, layers: Sequence[int], meta: Mapping[str, Any]) -> None:
    """Persist a survivor table with the settings that produced it."""

    payload = {
        "record_type": "keep_reap_survivors_v1",
        "layers": [int(layer) for layer in layers],
        "keep": int(survivors.shape[1]),
        "survivors": {str(int(layer)): survivors[row].tolist() for row, layer in enumerate(layers)},
        **dict(meta),
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
