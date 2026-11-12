"""Would a hot set of experts in RAM, with the rest on SSD, keep decode fast?

The kill test for keeping every routed expert of a model too large for RAM:
replay real routing through a fixed number of resident experts per layer and
count how often a selected expert is not resident. Each miss is an SSD read of
one expert (about 9 MB at 2.031 bpw on DeepSeek-V4.1) on the decode critical
path, so the miss rate converts directly into bytes and milliseconds.

Routing is a property of the token, not of how it was computed: in a causal
model the experts token ``t`` selects in a prefill pass are the ones it selects
when decoded. So routes recorded during the teacher's calibration prefill
(:class:`RouteRecorder`) are the decode routing sequence for those tokens.

Policies replayed by :func:`simulate_residency`:

- ``static``: the resident set never changes (the REAP-ordered hot set a
  pruned-plus-SSD artifact would ship). Misses are streamed, not cached.
- ``lru``: evict the least recently used resident.
- ``decayed``: evict the resident with the lowest exponentially decayed use
  count (half-life in tokens), the policy oMLX's DeepSeek-V4.1 offload uses.

``fetch_threshold`` models the hybrid "soft prune": a miss whose gate weight
is below the threshold is dropped (the expert is skipped and the remaining
weights renormalised, exactly as a pruned model would), and only heavier
misses are fetched. The result reports how much gate mass that drops.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

__all__ = [
    "ROUTE_ARRAY_PREFIX",
    "ROUTE_RECORD_TOKENS",
    "ResidencyResult",
    "RouteRecorder",
    "decode_tokens_per_second",
    "residency_report",
    "simulate_residency",
    "simulate_sessions",
]

ROUTE_ARRAY_PREFIX = "routes__"
#: Tokens of routing recorded per calibration session: enough for a decode-
#: length replay, about 1 MB per session at 43 layers x top-6.
ROUTE_RECORD_TOKENS = 2048
Policy = Literal["static", "lru", "decayed"]


class RouteRecorder:
    """Keep the first ``max_tokens`` routes of every layer, in token order.

    Implements the teacher runner's route-observer interface. The layer loop
    is layer-outer and chunk-inner, so each layer's rows arrive in token order.
    """

    def __init__(self, *, layers: Sequence[int], max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.layers = tuple(int(layer) for layer in layers)
        self.max_tokens = int(max_tokens)
        self._ids: dict[int, list[np.ndarray]] = {layer: [] for layer in self.layers}
        self._gates: dict[int, list[np.ndarray]] = {layer: [] for layer in self.layers}
        self._count = {layer: 0 for layer in self.layers}

    def add_chunk(
        self, *, layer: int, expert_ids: Any, gate_weights: Any, expert_outputs: Any = None
    ) -> None:
        del expert_outputs  # routing only
        if layer not in self._count:
            raise KeyError(f"layer {layer} is not tracked by this recorder")
        room = self.max_tokens - self._count[layer]
        if room <= 0:
            return
        ids = np.asarray(expert_ids)[:room]
        gates = np.asarray(gate_weights, dtype=np.float32)[:room]
        if ids.shape != gates.shape or ids.ndim != 2:
            raise ValueError("expert ids and gate weights must share a [rows, top_k] shape")
        self._ids[layer].append(ids.astype(np.int16))
        self._gates[layer].append(gates.astype(np.float16))
        self._count[layer] += int(ids.shape[0])

    def eval_pending(self) -> None:
        return None

    def flush_device(self) -> None:
        return None

    def as_arrays(self) -> dict[str, np.ndarray]:
        counts = {self._count[layer] for layer in self.layers}
        if len(counts) != 1:
            raise ValueError(f"layers recorded different token counts: {sorted(counts)}")
        ids = np.stack([np.concatenate(self._ids[layer]) for layer in self.layers], axis=1)
        gates = np.stack([np.concatenate(self._gates[layer]) for layer in self.layers], axis=1)
        return {
            f"{ROUTE_ARRAY_PREFIX}layers": np.asarray(self.layers, dtype=np.int64),
            f"{ROUTE_ARRAY_PREFIX}expert_ids": ids,  # int16 [tokens, layers, top_k]
            f"{ROUTE_ARRAY_PREFIX}gate_weights": gates,  # float16, same shape
        }


@dataclass(frozen=True)
class ResidencyResult:
    policy: str
    hot_per_layer: int
    num_experts: int
    tokens: int
    routed_slots: int
    misses: int  # fetched from SSD
    dropped: int  # skipped under fetch_threshold
    miss_rate: float
    misses_per_token: float
    dropped_gate_mass_fraction: float
    ssd_bytes_per_token: float | None
    per_layer_miss_rate: tuple[float, ...]
    dropped_gate_mass: float = 0.0
    total_gate_mass: float = 0.0
    per_layer_misses: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["per_layer_miss_rate"] = list(self.per_layer_miss_rate)
        return out


def simulate_residency(
    expert_ids: np.ndarray,
    *,
    num_experts: int,
    hot_per_layer: int,
    policy: Policy = "decayed",
    half_life: float = 64.0,
    initial_order: np.ndarray | None = None,
    gate_weights: np.ndarray | None = None,
    fetch_threshold: float | None = None,
    expert_bytes: float | None = None,
    warmup_tokens: int = 0,
) -> ResidencyResult:
    """Replay ``expert_ids [tokens, layers, top_k]`` through a per-layer hot set.

    ``initial_order`` is ``[layers, experts]`` saliency (e.g. REAP scores); the
    hot set starts as its top ``hot_per_layer`` experts. Without it the set
    starts empty and fills on demand. ``warmup_tokens`` are replayed but not
    counted.
    """

    ids = np.asarray(expert_ids)
    if ids.ndim != 3:
        raise ValueError("expert_ids must be [tokens, layers, top_k]")
    tokens, layers, top_k = ids.shape
    if not 0 < hot_per_layer <= num_experts:
        raise ValueError(f"hot_per_layer must be in 1..{num_experts}")
    if hot_per_layer < top_k:
        raise ValueError("the hot set must hold at least one token's top-k")
    if np.any(ids < 0) or np.any(ids >= num_experts):
        raise ValueError("expert ids outside the expert range")
    if policy not in ("static", "lru", "decayed"):
        raise ValueError(f"unknown policy {policy!r}")
    if policy == "static" and initial_order is None:
        raise ValueError("a static hot set needs initial_order to choose it")
    if fetch_threshold is not None and gate_weights is None:
        raise ValueError("fetch_threshold needs gate_weights")
    gates = None if gate_weights is None else np.asarray(gate_weights, dtype=np.float64)
    if gates is not None and gates.shape != ids.shape:
        raise ValueError("gate_weights must match expert_ids")
    if not 0 <= warmup_tokens < tokens:
        raise ValueError("warmup_tokens must leave at least one counted token")
    decay = np.log(2.0) / float(half_life)

    misses = np.zeros(layers, dtype=np.int64)
    dropped = 0
    dropped_mass = 0.0
    total_mass = 0.0
    for layer in range(layers):
        resident = np.zeros(num_experts, dtype=bool)
        if initial_order is not None:
            order = np.lexsort((np.arange(num_experts), -np.asarray(initial_order)[layer]))
            resident[order[:hot_per_layer]] = True
        score = np.zeros(num_experts, dtype=np.float64)  # decayed count at `stamp`
        stamp = np.zeros(num_experts, dtype=np.float64)
        last_use = np.full(num_experts, -1.0)
        for t in range(tokens):
            counted = t >= warmup_tokens
            needed = ids[t, layer]
            needed_set = set(int(e) for e in needed)
            for slot, expert in enumerate(needed):
                expert = int(expert)
                weight = None if gates is None else float(gates[t, layer, slot])
                if counted and weight is not None:
                    total_mass += weight
                if not resident[expert]:
                    if fetch_threshold is not None and weight < fetch_threshold:
                        if counted:
                            dropped += 1
                            dropped_mass += weight
                        continue
                    if counted:
                        misses[layer] += 1
                    if policy != "static":
                        if resident.sum() >= hot_per_layer:
                            candidates = np.flatnonzero(resident)
                            candidates = candidates[
                                [c not in needed_set for c in candidates.tolist()]
                            ]
                            if policy == "lru":
                                victim = candidates[np.argmin(last_use[candidates])]
                            else:
                                current = score[candidates] * np.exp(-decay * (t - stamp[candidates]))
                                victim = candidates[np.argmin(current)]
                            resident[victim] = False
                        resident[expert] = True
                # Usage bookkeeping for every selected expert, resident or not.
                score[expert] = score[expert] * np.exp(-decay * (t - stamp[expert])) + 1.0
                stamp[expert] = t
                last_use[expert] = t
    counted_tokens = tokens - warmup_tokens
    slots = counted_tokens * layers * top_k
    total_misses = int(misses.sum())
    per_layer = tuple(float(m) / (counted_tokens * top_k) for m in misses)
    return ResidencyResult(
        policy=policy,
        hot_per_layer=int(hot_per_layer),
        num_experts=int(num_experts),
        tokens=int(counted_tokens),
        routed_slots=int(slots),
        misses=total_misses,
        dropped=int(dropped),
        miss_rate=total_misses / slots,
        misses_per_token=total_misses / counted_tokens,
        dropped_gate_mass_fraction=(dropped_mass / total_mass) if total_mass else 0.0,
        ssd_bytes_per_token=None if expert_bytes is None else total_misses / counted_tokens * float(expert_bytes),
        per_layer_miss_rate=per_layer,
        dropped_gate_mass=float(dropped_mass),
        total_gate_mass=float(total_mass),
        per_layer_misses=tuple(int(m) for m in misses),
    )


def simulate_sessions(
    sessions: Sequence[tuple[np.ndarray, np.ndarray | None]],
    *,
    expert_bytes: float | None = None,
    **kwargs: Any,
) -> ResidencyResult:
    """Replay each session from the initial hot set and pool the counts.

    Residency is reset at every session boundary: unrelated sessions do not
    share a warm cache in a real deployment's worst case, and pooling raw
    counts weights each session by its tokens.
    """

    if not sessions:
        raise ValueError("no sessions to replay")
    parts = [
        simulate_residency(ids, gate_weights=gates, expert_bytes=expert_bytes, **kwargs)
        for ids, gates in sessions
    ]
    tokens = sum(r.tokens for r in parts)
    slots = sum(r.routed_slots for r in parts)
    misses = sum(r.misses for r in parts)
    layer_misses = np.sum([r.per_layer_misses for r in parts], axis=0)
    top_k = slots // (tokens * len(layer_misses))
    dropped_mass = sum(r.dropped_gate_mass for r in parts)
    total_mass = sum(r.total_gate_mass for r in parts)
    first = parts[0]
    return ResidencyResult(
        policy=first.policy,
        hot_per_layer=first.hot_per_layer,
        num_experts=first.num_experts,
        tokens=tokens,
        routed_slots=slots,
        misses=misses,
        dropped=sum(r.dropped for r in parts),
        miss_rate=misses / slots,
        misses_per_token=misses / tokens,
        dropped_gate_mass_fraction=(dropped_mass / total_mass) if total_mass else 0.0,
        ssd_bytes_per_token=None if expert_bytes is None else misses / tokens * float(expert_bytes),
        per_layer_miss_rate=tuple(float(m) / (tokens * top_k) for m in layer_misses),
        dropped_gate_mass=dropped_mass,
        total_gate_mass=total_mass,
        per_layer_misses=tuple(int(m) for m in layer_misses),
    )


def decode_tokens_per_second(
    *,
    resident_ms_per_token: float,
    misses_per_token: float,
    expert_bytes: float,
    ssd_bytes_per_second: float,
    host_sync_ms_per_token: float = 0.0,
) -> float:
    """Decode rate once misses are read synchronously on the critical path.

    ``host_sync_ms_per_token`` is the cost of learning which experts a layer
    picked before issuing the reads (oMLX measured it as a large share of its
    decode cycle); it is paid only by policies that fetch.
    """

    miss_ms = misses_per_token * expert_bytes / ssd_bytes_per_second * 1e3
    return 1e3 / (resident_ms_per_token + miss_ms + host_sync_ms_per_token)


def residency_report(
    results: Sequence[ResidencyResult], *, kill_miss_rate: float = 0.05
) -> Mapping[str, Any]:
    """Summarise replays and apply the plan's kill rule per hot-set size.

    For each hot-set size, the best policy that *fetches* every miss (no
    dropped gate mass) decides: above ``kill_miss_rate`` misses the SSD tier
    costs too much at that size.
    """

    by_hot: dict[int, list[ResidencyResult]] = {}
    for result in results:
        by_hot.setdefault(result.hot_per_layer, []).append(result)
    verdicts: dict[str, Any] = {}
    for hot, group in sorted(by_hot.items()):
        exact = [r for r in group if r.dropped == 0]
        if not exact:
            continue
        best = min(exact, key=lambda r: r.miss_rate)
        verdicts[str(hot)] = {
            "best_policy": best.policy,
            "miss_rate": best.miss_rate,
            "misses_per_token": best.misses_per_token,
            "ssd_bytes_per_token": best.ssd_bytes_per_token,
            "verdict": "KILL" if best.miss_rate > kill_miss_rate else "PASS",
        }
    return {
        "record_type": "keep_expert_residency_replay_v1",
        "kill_miss_rate": kill_miss_rate,
        "results": [result.to_dict() for result in results],
        "by_hot_per_layer": verdicts,
    }
