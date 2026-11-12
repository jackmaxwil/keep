"""An expert cache that never waits, under stock oMLX: a miss re-routes, the fetch lands later.

Starts from ``omlx_compact.py``'s model (C resident experts per layer as plain stacked arrays, the
router cut to their rows, so oMLX's fused decode kernels apply) and adds three things:

- Demand. Each layer also scores the whole pool's router rows on the same input; the top experts
  and their gate weights ride along in the token's graph (``mx.depends``), so reading them later
  costs no wait.
- A cache policy, run once per forward call before its first layer: an expert demanded with gate
  weight >= tau that is not resident is admitted into a free slot; the least recently demanded
  residents are evicted to keep ``spare`` slots free.
- No-wait installs. A slot's row in the stacked weight arrays is overwritten in place by reader
  threads (the stored bytes are the array's bytes). The router is the switch: an evicted slot's
  bias goes to -1e30 first, its bytes are overwritten only once every forward built before that
  has finished, and its router row is set to the new expert only after the bytes are in. A token
  never waits for a read and never runs a half-written expert.

    OMLX_COMPACT=DIR/compact-map.json OMLX_ROUTED_LOG=log.jsonl python omlx_routed.py serve ...

Environment: ``OMLX_ROUTED_TAU`` (default 0.2; ``inf`` keeps the starting set and must reproduce
the fixed set exactly), ``OMLX_ROUTED_SPARE`` (free slots kept per layer, default 4),
``OMLX_ROUTED_LAG`` (forward calls between a demand and its read-out, default 2: the call in
flight is never waited for), ``OMLX_ROUTED_VERIFY`` (each log line also counts routed slots whose
bytes differ from their expert's stored bytes; slow, for testing).
"""

import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from omlx_compact import patch as compact_patch

PROJECTIONS = ("w1", "w2", "w3")
OFF = -1e30  # a router bias no score can overcome


class Store:
    """Positional reads of one expert's stored tensors from the source checkpoint."""

    def __init__(self, source: Path):
        self.source = source
        self.index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
        self.files: dict[str, tuple[int, dict, int]] = {}

    def slab(self, key: str) -> tuple[int, int, int]:
        name = self.index[key]
        if name not in self.files:
            fd = os.open(self.source / name, os.O_RDONLY)
            size = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            self.files[name] = fd, json.loads(os.pread(fd, size, 8)), 8 + size
        fd, header, start = self.files[name]
        begin, end = header[key]["data_offsets"]
        return fd, start + begin, end - begin

    @staticmethod
    def pread_into(fd: int, offset: int, nbytes: int, view: np.ndarray) -> int:
        """Fill ``view`` from the file. Positional, so reader threads may call it."""
        target = memoryview(view.reshape(-1).view(np.uint8))
        assert target.nbytes == nbytes, (target.nbytes, nbytes)
        done = 0
        while done < nbytes:
            count = os.preadv(fd, [target[done:]], offset + done)
            assert count > 0, "truncated tensor"
            done += count
        return nbytes


class Layer:
    def __init__(self, index, moe, full_weight, full_bias, hot, rank, mx):
        self.index, self.gate, self.experts = index, moe.gate, moe.experts
        self.full_weight = full_weight.astype(mx.float32)  # [pool, dim]
        self.full_bias = full_bias.astype(mx.float32)
        self.typed_weight, self.typed_bias = full_weight, full_bias  # the rows a slot's router entry is copied from
        mx.eval(self.full_weight, self.full_bias)
        self.slot_expert = np.array(hot, dtype=np.int64)  # -1: unrouted
        self.slot_of = {e: s for s, e in enumerate(hot)}
        self.last_use = -1.0 - np.array(rank, dtype=np.float64)  # colder starting experts go first
        self.cooling: list[tuple[int, int]] = []  # (slot, forward call that unrouted it)
        self.free: list[int] = []
        self.reading: dict[int, tuple[int, list]] = {}  # expert -> (slot, futures)
        self.views = {}
        for proj in PROJECTIONS:
            lin = getattr(self.experts, proj)
            assert lin["weight"].dtype == mx.uint32 and lin["scales"].dtype == mx.uint8, proj
            self.views[proj] = (lin, lin["weight"], lin["scales"],
                                np.asarray(memoryview(lin["weight"])), np.asarray(memoryview(lin["scales"])))

    def views_current(self) -> bool:
        """The host views still alias the arrays the kernels read."""
        for lin, weight, scales, weight_view, scales_view in self.views.values():
            for owner, name, view in ((weight, "weight", weight_view), (scales, "scales", scales_view)):
                if lin[name] is not owner or np.frombuffer(memoryview(owner), np.uint8).ctypes.data != view.ctypes.data:
                    return False
        return True


class Cache:
    """The policy. A subclass sets ``layers`` and says where a slot's bytes live (``slabs``) and
    how the router switches an expert on and off (``route``, ``unroute``)."""

    def __init__(self, config, mx):
        self.mx, self.config = mx, config
        self.tau = float(os.environ.get("OMLX_ROUTED_TAU", "0.2"))
        self.spare = int(os.environ.get("OMLX_ROUTED_SPARE", "4"))
        self.lag = int(os.environ.get("OMLX_ROUTED_LAG", "2"))
        self.log_path = os.environ.get("OMLX_ROUTED_LOG")
        assert config.score_func == "sqrtsoftplus" and self.lag >= 1
        self.pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="routed-io")
        self.step = 0
        self.pending: dict[int, list] = {}  # forward call -> per-layer (index, weight) demand arrays
        self.stats = dict.fromkeys(("calls", "tokens", "demand", "resident", "wanted", "admitted", "installed",
                                    "evicted", "bytes", "no_slot", "policy_s"), 0)
        self.started = time.time()

    def ready(self, layers):
        self.layers = layers
        self.by_gate = {id(layer.gate): layer for layer in layers}
        # A resident expert's rows must equal its stored bytes: installs rely on it.
        for layer in layers[:: max(1, len(layers) // 4)]:
            assert not self.differing(layer, 0), f"layer {layer.index}: slot 0 differs from its stored bytes"

    def slabs(self, layer, expert, slot):
        """(fd, offset, nbytes, the slot's host view) for each stored tensor of ``expert``."""
        raise NotImplementedError

    def route(self, layer, expert, slot):
        raise NotImplementedError

    def unroute(self, layer, expert, slot):
        raise NotImplementedError

    def differing(self, layer, slot) -> int:
        """Tensors of a routed slot whose bytes differ from its expert's stored bytes."""
        bad = 0
        for fd, offset, nbytes, view in self.slabs(layer, int(layer.slot_expert[slot]), slot):
            stored = np.empty(nbytes, np.uint8)
            Store.pread_into(fd, offset, nbytes, stored)
            bad += not np.array_equal(stored, view.reshape(-1).view(np.uint8))
        return bad

    # --- demand, recorded while a forward call is built -------------------------------------

    def demand(self, layer, x):
        mx, c = self.mx, self.config
        raw = (x.reshape(-1, x.shape[-1]).astype(mx.float32) @ layer.full_weight.T) / c.gate_temp
        scores = mx.sqrt(mx.logaddexp(raw, 0))
        index = mx.argpartition(-(scores + layer.full_bias), kth=c.n_activated_experts - 1, axis=-1)
        index = index[..., : c.n_activated_experts]
        return index, mx.take_along_axis(scores, index, -1)

    def observe(self, gate, x, tied):
        """Record this layer's demand and tie it to ``tied`` so one eval covers both."""
        layer = self.by_gate[id(gate)]
        index, weight = self.demand(layer, x)
        self.pending.setdefault(self.step, []).append((index, weight))
        tied = self.mx.depends(tied, [index, weight])
        return tied[0] if isinstance(tied, (list, tuple)) else tied

    # --- policy, run before the first layer of each forward call ----------------------------

    def begin(self):
        self.step += 1
        ready = self.step - self.lag
        if self.tau != float("inf") and ready in self.pending:
            started = time.perf_counter()
            self.harvest(ready, self.pending.pop(ready))
            self.stats["policy_s"] += time.perf_counter() - started
        for old in [s for s in self.pending if s < ready]:
            del self.pending[old]
        if self.log_path and self.step % 64 == 0:
            self.write_log()

    def harvest(self, call, demand):
        c = self.config
        self.stats["calls"] += 1
        # Reading this call's demand means every earlier call has left the GPU.
        for layer, (index, weight) in zip(self.layers, demand):
            index = np.asarray(index).reshape(-1, c.n_activated_experts)
            weight = np.asarray(weight, dtype=np.float64).reshape(index.shape)
            weight = weight / (weight.sum(-1, keepdims=True) + 1e-20) * c.route_scale
            if layer.index == 0:
                self.stats["tokens"] += index.shape[0]
            self.stats["demand"] += index.size
            strong = weight >= self.tau
            wanted: dict[int, float] = {}
            for expert, w in zip(index[strong].tolist(), weight[strong].tolist()):
                wanted[expert] = max(w, wanted.get(expert, 0.0))
            self.stats["resident"] += sum(e in layer.slot_of for e in index.reshape(-1).tolist())
            self.install(layer)
            missing = []
            for expert, w in wanted.items():
                slot = layer.slot_of.get(expert)
                if slot is not None:
                    layer.last_use[slot] = call
                elif expert not in layer.reading:
                    missing.append((w, expert))
            self.stats["wanted"] += len(missing)
            layer.free += [slot for slot, since in layer.cooling if since <= call]
            layer.cooling = [(slot, since) for slot, since in layer.cooling if since > call]
            if missing and layer.free and not layer.views_current():
                raise RuntimeError(f"layer {layer.index}: expert arrays moved; cannot install in place")
            for _, expert in sorted(missing, reverse=True):
                if not layer.free:
                    self.stats["no_slot"] += 1
                    continue
                slot = layer.free.pop()
                futures = [self.pool.submit(Store.pread_into, *slab) for slab in self.slabs(layer, expert, slot)]
                layer.reading[expert] = (slot, futures)
                self.stats["admitted"] += 1
            self.evict(layer, set(wanted))

    def install(self, layer) -> None:
        """Route the experts whose bytes have landed. The router arrays are replaced, never
        written in place and never evaluated here: calls already built keep the old ones, and
        the next call's own eval computes the new ones without a wait."""
        done = [e for e, (_, futures) in layer.reading.items() if all(f.done() for f in futures)]
        for expert in done:
            slot, futures = layer.reading.pop(expert)
            self.stats["bytes"] += sum(f.result() for f in futures)
            self.route(layer, expert, slot)
            layer.slot_expert[slot], layer.slot_of[expert] = expert, slot
            layer.last_use[slot] = self.step
            self.stats["installed"] += 1

    def evict(self, layer, protected) -> None:
        """Unroute the least recently demanded residents until ``spare`` slots are free or coming."""
        short = self.spare - len(layer.free) - len(layer.cooling)
        if short <= 0:
            return
        order = np.argsort(layer.last_use, kind="stable")
        victims = [int(s) for s in order if layer.slot_expert[s] >= 0 and int(layer.slot_expert[s]) not in protected]
        for slot in victims[:short]:
            expert = int(layer.slot_expert[slot])
            del layer.slot_of[expert]
            layer.slot_expert[slot] = -1
            layer.last_use[slot] = np.inf
            self.unroute(layer, expert, slot)
            # Calls built before this one may still route here; its bytes may change only once
            # the call being built now is known to have finished.
            layer.cooling.append((slot, self.step))
            self.stats["evicted"] += 1

    def mismatched(self) -> int:
        """Routed slots (first and last layer) whose rows differ from their expert's stored bytes."""
        return sum(self.differing(layer, slot) for layer in (self.layers[0], self.layers[-1])
                   for slot, expert in enumerate(layer.slot_expert.tolist()) if expert >= 0)

    def write_log(self):
        row = {"t": round(time.time() - self.started, 2), "step": self.step, **self.stats,
               "reading": sum(len(layer.reading) for layer in self.layers)}
        if os.environ.get("OMLX_ROUTED_VERIFY"):
            row["mismatched"] = self.mismatched()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")


class FP4Cache(Cache):
    """Slots are rows of oMLX's stacked MXFP4 arrays; the router holds one row per slot."""

    def __init__(self, model, config, spec, full, mx):
        super().__init__(config, mx)
        self.store = Store(Path(spec["source"]))
        for key in {name: key for key, name in self.store.index.items() if ".ffn.experts." in key}.values():
            self.store.slab(key)  # open every expert shard now, not during a token
        self.ready([Layer(i, blk.ffn, full[i, "weight"], full[i, "bias"], spec["hot"][i], spec["rank"][i], mx)
                    for i, blk in enumerate(model.language_model.layers)])

    def slabs(self, layer, expert, slot):
        for proj in PROJECTIONS:
            base = f"layers.{layer.index}.ffn.experts.{expert}.{proj}."
            for field, view in (("weight", layer.views[proj][3]), ("scale", layer.views[proj][4])):
                yield (*self.store.slab(base + field), view[slot])

    def route(self, layer, expert, slot):
        layer.gate.weight[slot] = layer.typed_weight[expert]
        layer.gate.bias[slot] = layer.typed_bias[expert]

    def unroute(self, layer, expert, slot):
        layer.gate.bias[slot] = OFF


def patch(spec: dict) -> None:
    import mlx.core as mx
    from omlx.patches.deepseek_v41 import decode_moe, language, loading

    full: dict = {}
    compact_patch(spec, full)
    state: dict = {}
    load, fused_gate, gate_call = loading.load, decode_moe._gate, language.Gate.__call__

    def load_routed(*args, **kwargs):
        model, processor = load(*args, **kwargs)
        state["cache"] = FP4Cache(model, model.language_model.layers[0].ffn.gate._config, spec, full, mx)
        full.clear()
        return model, processor

    def gate_fused(moe, x, config):  # one-token and batched fused decode
        cache = state.get("cache")
        if cache is None or id(moe.gate) not in cache.by_gate:
            return fused_gate(moe, x, config)
        if cache.by_gate[id(moe.gate)].index == 0:
            cache.begin()
        scores, quantized = fused_gate(moe, x, config)
        return cache.observe(moe.gate, x, scores), quantized

    def gate_plain(self, x, image_mask):  # prefill and every unfused path
        cache = state.get("cache")
        if cache is None or id(self) not in cache.by_gate:
            return gate_call(self, x, image_mask)
        if cache.by_gate[id(self)].index == 0:
            cache.begin()
        index, weights = gate_call(self, x, image_mask)
        return cache.observe(self, x, index), weights

    loading.load = load_routed
    decode_moe._gate = gate_fused
    language.Gate.__call__ = gate_plain


if __name__ == "__main__":
    patch(json.loads(Path(os.environ["OMLX_COMPACT"]).read_text()))
    from omlx.cli import main

    sys.exit(main())
