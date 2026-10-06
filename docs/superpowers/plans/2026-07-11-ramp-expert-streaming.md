# RAMP Compressed-Expert Disk Streaming (F1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let RAMP run a KEEP artifact whose routed experts do not all fit in
RAM by streaming per-layer compressed expert shards from disk on demand through
a byte-budgeted LRU residency cache — removing the resident-memory ceiling while
keeping every routed matmul on the existing 8-bit NAX fast path.

**Architecture:** Replace eager expert binding with a shared `ExpertStore`. Each
sparse MoE layer is bound to a `StreamingSwitchGLUProxy` instead of a resident
`QuantizedVQSwitchGLU`. On the forward pass the proxy asks the store for its
layer's `QuantizedVQSwitchGLU`; the store loads the three shards on a cache miss,
serves the cached module on a hit, and evicts least-recently-used non-pinned
layers to stay under a byte budget. Because the proxy hands the kernel the exact
same `QuantizedVQSwitchGLU` eager binding would, the `gather_vqmm → nax_e8`
dispatch is unchanged — streaming buys RAM headroom, not a new kernel path.

**Tech Stack:** Python 3.11+, MLX, safetensors, pytest.

## Global Constraints

Inherited from [`2026-07-11-ramp-beyond-ram-roadmap.md`](2026-07-11-ramp-beyond-ram-roadmap.md).
Load-bearing here:

- **The NAX speed wall.** Streaming must not change kernel dispatch. The proxy
  delegates to a real `QuantizedVQSwitchGLU`, so `code_bits=8` experts stay on
  `nax_e8`. A task explicitly asserts the proxy is transparent.
- **Off by default, never regress the resident path.** Streaming is opt-in
  (`StreamingConfig.enabled`); when disabled, binding is byte-identical to today.
- **Streaming benchmarks are labeled, never compared to Lane S.** They page from
  disk by design; report tok/s, cache hit-rate, and `pageouts`/`swapouts` from
  `collect_metric_snapshot`, tagged as streaming runs.
- Tests run with **no model artifact and no Metal** — fabricate tiny safetensors
  and inject fake loaders/size functions.
- Do not set wired limit here (that is Plan B, opt-in only).
- `from __future__ import annotations`; pytest; `tmp_path`. New code under
  `src/mlx_vq/runtime/`; `ramp` re-export as the final task. One commit/task.

## File Structure

- Create `src/mlx_vq/runtime/expert_store.py` — `StreamingConfig`,
  `ExpertResidencyCache`, `ExpertStore`, `StreamingSwitchGLUProxy`,
  `shard_group_nbytes`, `bind_streaming_experts`.
- Modify `src/mlx_vq/models/glm52_vq_adapter.py` — add
  `bind_glm52_vq_experts_streaming` beside the eager `bind_glm52_vq_experts`.
- Modify `src/mlx_vq/models/glm52_composite_loader.py` — thread an optional
  `streaming` config into `load_authenticated_glm52_composite`.
- Create `tests/test_expert_store.py`, `tests/test_expert_streaming_bind.py`.
- Create `src/ramp/runtime/expert_store.py` — re-export (final task).

## Verified facts (2026-07-11)

- Eager load: `load_glm52_vq_switch_glu(artifact_dir, layer)` (`glm52_vq_adapter.py:581`) reads `layer-{layer:05d}-{gate_proj,up_proj,down_proj}.safetensors` via `load_quantized_vq_switch_linear` (`io/load.py:235`, eager `mx.load`), returning a `QuantizedVQSwitchGLU` (`glm4_moe_adapter.py:109`).
- Bind loop: `bind_glm52_vq_experts` (`glm52_vq_adapter.py:698`) iterates sparse layers, calling `layer.mlp.bind_switch_mlp(switch_mlp)` on each `Glm52VQMoE` (`:427`). Forward: `Glm52VQMoE.__call__` (`:451`) does `y = self.switch_mlp(x, inds)`.
- `QuantizedVQSwitchGLU.__call__(x, indices)` (`glm4_moe_adapter.py:311`) is the delegate the proxy must mirror.
- No cache/mmap/eviction exists today; every sparse layer's experts are resident from construction.

## Design decisions (recorded)

- **Granularity = per layer.** A layer's three projection shards load/evict as a
  unit. This matches the on-disk shard layout and keeps kernels untouched.
  Per-expert streaming (partial-tensor reads + kernel index remap) is a larger,
  separate follow-up and is explicitly out of scope for v1.
- **Memory release.** Eviction drops the Python reference to the module (and its
  `codes`/`scales` `mx.array`s) and asks MLX to return pooled memory to the OS.
- **Budget source.** Per-layer byte size is read from the safetensors headers
  (payload offsets) without materializing arrays.

---

### Task 1: `shard_group_nbytes` — size a layer from headers without loading

**Files:**
- Create: `src/mlx_vq/runtime/expert_store.py`
- Test: `tests/test_expert_store.py`

**Interfaces:**
- Produces:
  `safetensors_payload_nbytes(path: Path) -> int`;
  `shard_group_nbytes(artifact_dir: Path, layer: int) -> int` (sum of the three
  `layer-{layer:05d}-{proj}.safetensors` payloads).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_expert_store.py
from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from mlx_vq.runtime.expert_store import (
    safetensors_payload_nbytes,
    shard_group_nbytes,
)


def _write_shard(path: Path, n_bytes_codes: int) -> None:
    # codes: uint8 vector; scales: float32 vector. Sizes are exact & known.
    arrays = {
        "codes": mx.zeros((n_bytes_codes,), dtype=mx.uint8),
        "scales": mx.zeros((4,), dtype=mx.float32),
    }
    mx.save_safetensors(str(path), arrays)


def test_payload_nbytes_counts_only_tensor_payload(tmp_path: Path) -> None:
    shard = tmp_path / "one.safetensors"
    _write_shard(shard, n_bytes_codes=32)
    # 32 bytes of uint8 codes + 4*4 bytes of float32 scales = 48 payload bytes.
    assert safetensors_payload_nbytes(shard) == 48


def test_shard_group_nbytes_sums_three_projections(tmp_path: Path) -> None:
    layer = 3
    for proj in ("gate_proj", "up_proj", "down_proj"):
        _write_shard(tmp_path / f"layer-{layer:05d}-{proj}.safetensors", n_bytes_codes=16)
    # each shard: 16 + 16 = 32 payload bytes; three shards → 96.
    assert shard_group_nbytes(tmp_path, layer) == 96
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_expert_store.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_vq.runtime.expert_store'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mlx_vq/runtime/expert_store.py
from __future__ import annotations

import json
from pathlib import Path

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def safetensors_payload_nbytes(path: Path) -> int:
    """Sum tensor payload bytes from a safetensors header without loading data."""
    with open(path, "rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len).decode("utf-8"))
    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        total += int(end) - int(begin)
    return total


def shard_group_nbytes(artifact_dir: Path, layer: int) -> int:
    artifact_dir = Path(artifact_dir)
    total = 0
    for proj in _PROJECTIONS:
        total += safetensors_payload_nbytes(
            artifact_dir / f"layer-{layer:05d}-{proj}.safetensors"
        )
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_expert_store.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/expert_store.py tests/test_expert_store.py
git commit -m "feat: size expert shard groups from safetensors headers"
```

---

### Task 2: `ExpertResidencyCache` — byte-budgeted LRU with pinning and hooks

**Files:**
- Modify: `src/mlx_vq/runtime/expert_store.py`
- Test: `tests/test_expert_store.py`

**Interfaces:**
- Produces:
  `ExpertResidencyCache(budget_bytes: int, *, pinned_layers: Iterable[int] = (), on_event: Callable[[str, int], None] | None = None, release: Callable[[], None] | None = None)`;
  `.get(layer: int, load: Callable[[], object], nbytes: int) -> object`;
  `.resident_layers() -> tuple[int, ...]`; `.bytes_resident() -> int`.
  `on_event(kind, layer)` fires with `kind in {"hit","load","evict"}`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_expert_store.py
from mlx_vq.runtime.expert_store import ExpertResidencyCache


def test_cache_hit_does_not_reload() -> None:
    calls = {"n": 0}

    def load() -> str:
        calls["n"] += 1
        return "module-7"

    cache = ExpertResidencyCache(budget_bytes=1000)
    assert cache.get(7, load, nbytes=100) == "module-7"
    assert cache.get(7, load, nbytes=100) == "module-7"
    assert calls["n"] == 1  # second call served from cache
    assert cache.resident_layers() == (7,)


def test_cache_evicts_lru_to_stay_under_budget() -> None:
    events: list[tuple[str, int]] = []
    cache = ExpertResidencyCache(budget_bytes=250, on_event=events.append if False else (lambda k, l: events.append((k, l))))
    for layer in (1, 2, 3):  # each 100 bytes; budget 250 holds two
        cache.get(layer, lambda l=layer: f"m{l}", nbytes=100)
    assert cache.bytes_resident() <= 250
    assert 1 not in cache.resident_layers()  # oldest evicted
    assert ("evict", 1) in events


def test_pinned_layer_is_never_evicted() -> None:
    cache = ExpertResidencyCache(budget_bytes=150, pinned_layers=(1,))
    cache.get(1, lambda: "m1", nbytes=100)
    cache.get(2, lambda: "m2", nbytes=100)
    cache.get(3, lambda: "m3", nbytes=100)
    assert 1 in cache.resident_layers()  # pinned survives
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_expert_store.py -k cache -q`
Expected: FAIL with `ImportError: cannot import name 'ExpertResidencyCache'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/runtime/expert_store.py
from collections import OrderedDict
from typing import Callable, Iterable

import mlx.core as mx


def _release_pooled_memory() -> None:
    clear = getattr(mx, "clear_cache", None)
    if clear is not None:
        clear()


class ExpertResidencyCache:
    """LRU cache of per-layer expert modules bounded by a byte budget."""

    def __init__(
        self,
        budget_bytes: int,
        *,
        pinned_layers: Iterable[int] = (),
        on_event: Callable[[str, int], None] | None = None,
        release: Callable[[], None] | None = None,
    ) -> None:
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        self._budget = budget_bytes
        self._pinned = frozenset(pinned_layers)
        self._on_event = on_event
        self._release = release if release is not None else _release_pooled_memory
        self._entries: "OrderedDict[int, tuple[object, int]]" = OrderedDict()

    def _emit(self, kind: str, layer: int) -> None:
        if self._on_event is not None:
            self._on_event(kind, layer)

    def bytes_resident(self) -> int:
        return sum(nbytes for _module, nbytes in self._entries.values())

    def resident_layers(self) -> tuple[int, ...]:
        return tuple(self._entries.keys())

    def get(self, layer: int, load: Callable[[], object], nbytes: int) -> object:
        entry = self._entries.get(layer)
        if entry is not None:
            self._entries.move_to_end(layer)
            self._emit("hit", layer)
            return entry[0]
        module = load()
        self._entries[layer] = (module, nbytes)
        self._entries.move_to_end(layer)
        self._emit("load", layer)
        self._evict_to_budget(protect=layer)
        return module

    def _evict_to_budget(self, *, protect: int) -> None:
        released = False
        while self.bytes_resident() > self._budget:
            victim = self._first_evictable(protect=protect)
            if victim is None:
                break  # cannot shrink further (all pinned/protected)
            self._entries.pop(victim)
            self._emit("evict", victim)
            released = True
        if released:
            self._release()

    def _first_evictable(self, *, protect: int) -> int | None:
        for layer in self._entries:  # OrderedDict iterates LRU-first
            if layer == protect or layer in self._pinned:
                continue
            return layer
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_expert_store.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/expert_store.py tests/test_expert_store.py
git commit -m "feat: add byte-budgeted LRU expert residency cache"
```

---

### Task 3: `StreamingConfig` + `ExpertStore`

**Files:**
- Modify: `src/mlx_vq/runtime/expert_store.py`
- Test: `tests/test_expert_store.py`

**Interfaces:**
- Consumes: `ExpertResidencyCache`, `shard_group_nbytes`.
- Produces:
  `StreamingConfig(budget_bytes: int, pinned_layers: tuple[int, ...] = (), enabled: bool = True)`;
  `ExpertStore(artifact_dir, sparse_layers, *, config, loader=load_glm52_vq_switch_glu, size_fn=None, on_event=None)`
  with `.switch_glu(layer) -> QuantizedVQSwitchGLU` and `.cache` (the
  `ExpertResidencyCache`). `loader(artifact_dir, layer)` and
  `size_fn(layer) -> int` are injectable for tests.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_expert_store.py
from mlx_vq.runtime.expert_store import ExpertStore, StreamingConfig


def test_expert_store_caches_and_reloads_after_eviction(tmp_path: Path) -> None:
    loads: list[int] = []

    def fake_loader(_artifact_dir, layer: int) -> str:
        loads.append(layer)
        return f"glu-{layer}"

    store = ExpertStore(
        tmp_path,
        sparse_layers=(1, 2, 3),
        config=StreamingConfig(budget_bytes=250),
        loader=fake_loader,
        size_fn=lambda _layer: 100,  # 250 budget holds two layers
    )
    assert store.switch_glu(1) == "glu-1"
    assert store.switch_glu(1) == "glu-1"  # cached
    assert loads == [1]

    store.switch_glu(2)
    store.switch_glu(3)  # evicts layer 1
    assert 1 not in store.cache.resident_layers()
    store.switch_glu(1)  # reload after eviction
    assert loads == [1, 2, 3, 1]


def test_expert_store_rejects_unknown_layer(tmp_path: Path) -> None:
    store = ExpertStore(
        tmp_path,
        sparse_layers=(1,),
        config=StreamingConfig(budget_bytes=1000),
        loader=lambda _d, _l: "x",
        size_fn=lambda _l: 1,
    )
    import pytest

    with pytest.raises(KeyError, match="not a sparse layer"):
        store.switch_glu(99)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_expert_store.py -k expert_store -q`
Expected: FAIL with `ImportError: cannot import name 'ExpertStore'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/runtime/expert_store.py
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StreamingConfig:
    budget_bytes: int
    pinned_layers: tuple[int, ...] = ()
    enabled: bool = True


class ExpertStore:
    def __init__(
        self,
        artifact_dir,
        sparse_layers: Iterable[int],
        *,
        config: StreamingConfig,
        loader=None,
        size_fn: Callable[[int], int] | None = None,
        on_event: Callable[[str, int], None] | None = None,
    ) -> None:
        self._artifact_dir = Path(artifact_dir)
        self._layers = frozenset(sparse_layers)
        self._config = config
        if loader is None:
            # Imported lazily to avoid a heavy import at module load.
            from mlx_vq.models.glm52_vq_adapter import load_glm52_vq_switch_glu

            loader = load_glm52_vq_switch_glu
        self._loader = loader
        self._size_fn = size_fn or (lambda layer: shard_group_nbytes(self._artifact_dir, layer))
        self.cache = ExpertResidencyCache(
            budget_bytes=config.budget_bytes,
            pinned_layers=config.pinned_layers,
            on_event=on_event,
        )

    @property
    def config(self) -> StreamingConfig:
        return self._config

    def switch_glu(self, layer: int):
        if layer not in self._layers:
            raise KeyError(f"layer {layer} is not a sparse layer of this store")
        return self.cache.get(
            layer,
            load=lambda: self._loader(self._artifact_dir, layer),
            nbytes=self._size_fn(layer),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_expert_store.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/expert_store.py tests/test_expert_store.py
git commit -m "feat: add ExpertStore streaming loader with residency cache"
```

---

### Task 4: `StreamingSwitchGLUProxy` — transparent per-layer delegate

**Files:**
- Modify: `src/mlx_vq/runtime/expert_store.py`
- Test: `tests/test_expert_store.py`

**Interfaces:**
- Consumes: `ExpertStore`.
- Produces: `StreamingSwitchGLUProxy(store: ExpertStore, layer: int)` — an
  `nn.Module` whose `__call__(x, indices, **kwargs)` resolves
  `store.switch_glu(layer)` and delegates unchanged, and which forwards
  `num_experts`, `input_dims`, `hidden_dims` from the resolved module so profile
  checks keep working.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_expert_store.py
from mlx_vq.runtime.expert_store import StreamingSwitchGLUProxy


class _FakeGLU:
    num_experts = 128
    input_dims = 512
    hidden_dims = 1408

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, x, indices, **kwargs):
        self.calls.append((x, indices, tuple(sorted(kwargs))))
        return f"y({x},{indices})"


def test_proxy_delegates_to_resolved_glu(tmp_path: Path) -> None:
    glu = _FakeGLU()
    store = ExpertStore(
        tmp_path,
        sparse_layers=(5,),
        config=StreamingConfig(budget_bytes=1000),
        loader=lambda _d, _l: glu,
        size_fn=lambda _l: 10,
    )
    proxy = StreamingSwitchGLUProxy(store, layer=5)
    assert proxy("X", "IDS") == "y(X,IDS)"
    assert proxy.num_experts == 128
    assert glu.calls == [("X", "IDS", ())]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_expert_store.py -k proxy -q`
Expected: FAIL with `ImportError: cannot import name 'StreamingSwitchGLUProxy'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/runtime/expert_store.py
import mlx.nn as nn


class StreamingSwitchGLUProxy(nn.Module):
    """Stands in for a resident QuantizedVQSwitchGLU; resolves it per call."""

    def __init__(self, store: ExpertStore, layer: int) -> None:
        super().__init__()
        # Store references are attributes, not module parameters (leading '_'
        # keeps them out of nn.Module parameter traversal).
        self._store = store
        self._layer = layer

    def __call__(self, x, indices, **kwargs):
        return self._store.switch_glu(self._layer)(x, indices, **kwargs)

    def _resolved(self):
        return self._store.switch_glu(self._layer)

    @property
    def num_experts(self) -> int:
        return self._resolved().num_experts

    @property
    def input_dims(self) -> int:
        return self._resolved().input_dims

    @property
    def hidden_dims(self) -> int:
        return self._resolved().hidden_dims
```

> Transparency guarantee: the object handed to the kernel is the resolved
> `QuantizedVQSwitchGLU` itself, so `gather_vqmm → nax_e8` dispatch is identical
> to the resident path. The proxy adds a dict lookup, not a kernel change.

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_expert_store.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/expert_store.py tests/test_expert_store.py
git commit -m "feat: add transparent streaming switch-GLU proxy"
```

---

### Task 5: `bind_streaming_experts` + model wiring

**Files:**
- Modify: `src/mlx_vq/runtime/expert_store.py` (add `bind_streaming_experts`)
- Modify: `src/mlx_vq/models/glm52_vq_adapter.py` (add
  `bind_glm52_vq_experts_streaming`)
- Test: `tests/test_expert_streaming_bind.py`

**Interfaces:**
- Consumes: `ExpertStore`, `StreamingSwitchGLUProxy`.
- Produces:
  `bind_streaming_experts(model, store, *, sparse_layers) -> tuple[int, ...]`
  (binds a proxy to each sparse layer's `mlp` via `bind_switch_mlp`);
  `bind_glm52_vq_experts_streaming(model, artifact_dir, *, profile, config, layers=None) -> ExpertStore`
  (mirrors `bind_glm52_vq_experts` validation, then binds proxies through a
  shared store).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_expert_streaming_bind.py
from __future__ import annotations

from pathlib import Path

from mlx_vq.runtime.expert_store import (
    ExpertStore,
    StreamingConfig,
    StreamingSwitchGLUProxy,
    bind_streaming_experts,
)


class _FakeMoE:
    def __init__(self) -> None:
        self.switch_mlp = None

    def bind_switch_mlp(self, module) -> None:
        self.switch_mlp = module


class _FakeLayer:
    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.mlp = _FakeMoE()


class _FakeModel:
    def __init__(self, layer_indices) -> None:
        self.layers = [_FakeLayer(i) for i in layer_indices]


def test_bind_streaming_experts_binds_proxies(tmp_path: Path) -> None:
    model = _FakeModel([1, 2, 3])
    store = ExpertStore(
        tmp_path,
        sparse_layers=(1, 2, 3),
        config=StreamingConfig(budget_bytes=1000),
        loader=lambda _d, layer: f"glu-{layer}",
        size_fn=lambda _l: 10,
    )
    bound = bind_streaming_experts(model, store, sparse_layers=(1, 2, 3))
    assert bound == (1, 2, 3)
    for layer in model.layers:
        assert isinstance(layer.mlp.switch_mlp, StreamingSwitchGLUProxy)
    # No load happened just from binding — streaming is lazy.
    assert model.layers[0].mlp.switch_mlp._layer == 1
    assert store.cache.resident_layers() == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_expert_streaming_bind.py -q`
Expected: FAIL with `ImportError: cannot import name 'bind_streaming_experts'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/runtime/expert_store.py
def bind_streaming_experts(model, store: ExpertStore, *, sparse_layers) -> tuple[int, ...]:
    by_index = {getattr(layer, "layer_idx", idx): layer for idx, layer in enumerate(model.layers)}
    bound: list[int] = []
    for layer_idx in sparse_layers:
        layer = by_index[layer_idx]
        proxy = StreamingSwitchGLUProxy(store, layer=layer_idx)
        layer.mlp.bind_switch_mlp(proxy)
        bound.append(layer_idx)
    return tuple(bound)
```

Add the model-aware entry to `src/mlx_vq/models/glm52_vq_adapter.py`, reusing the
same sparse-layer discovery as the eager binder (`is_glm52_sparse_layer`,
`glm52_vq_adapter.py:714`):

```python
# src/mlx_vq/models/glm52_vq_adapter.py (new function beside bind_glm52_vq_experts)
def bind_glm52_vq_experts_streaming(
    model: "GLM52VQModel",
    artifact_dir,
    *,
    profile: "ModelProfile",
    config: "StreamingConfig",
    layers: tuple[int, ...] | None = None,
) -> "ExpertStore":
    from mlx_vq.runtime.expert_store import ExpertStore, bind_streaming_experts

    expected_sparse_layers = tuple(
        layer_idx
        for layer_idx in range(model.args.num_hidden_layers)
        if is_glm52_sparse_layer(model.args, layer_idx)
    )
    requested = expected_sparse_layers if layers is None else tuple(layers)
    store = ExpertStore(artifact_dir, sparse_layers=requested, config=config)
    bind_streaming_experts(model, store, sparse_layers=requested)
    return store
```

Add the import of `StreamingConfig` for typing at the top of the adapter (inside
`TYPE_CHECKING` to avoid a runtime cycle):

```python
# near the top of src/mlx_vq/models/glm52_vq_adapter.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_vq.runtime.expert_store import ExpertStore, StreamingConfig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_expert_streaming_bind.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/expert_store.py src/mlx_vq/models/glm52_vq_adapter.py tests/test_expert_streaming_bind.py
git commit -m "feat: bind streaming expert proxies for GLM52"
```

---

### Task 6: Thread streaming into the composite loader + ramp re-export + docs

**Files:**
- Modify: `src/mlx_vq/models/glm52_composite_loader.py`
  (`load_authenticated_glm52_composite`, `:1466`)
- Modify: `README.md` (Architecture: note the streaming path)
- Create: `src/ramp/runtime/expert_store.py`
- Test: `tests/test_expert_streaming_bind.py` (loader routing with a fake binder)

**Interfaces:**
- Consumes: `StreamingConfig`, `bind_glm52_vq_experts_streaming`.
- Produces: `load_authenticated_glm52_composite(..., streaming: StreamingConfig | None = None)`.
  When `streaming is None` or `streaming.enabled is False`, the eager
  `bind_glm52_vq_experts` path is used unchanged. Otherwise experts bind through
  the store and the returned report carries the `ExpertStore`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_expert_streaming_bind.py
def test_choose_expert_binder_selects_streaming_when_enabled() -> None:
    from mlx_vq.models.glm52_composite_loader import _choose_expert_binder
    from mlx_vq.runtime.expert_store import StreamingConfig

    eager_called: list[str] = []
    streaming_called: list[str] = []

    def eager(**_kwargs):
        eager_called.append("eager")
        return ("eager", None)

    def streaming(**_kwargs):
        streaming_called.append("streaming")
        return ("streaming", object())

    binder = _choose_expert_binder(
        StreamingConfig(budget_bytes=1, enabled=True), eager=eager, streaming=streaming
    )
    binder()
    assert streaming_called == ["streaming"] and eager_called == []

    binder_off = _choose_expert_binder(None, eager=eager, streaming=streaming)
    binder_off()
    assert eager_called == ["eager"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_expert_streaming_bind.py::test_choose_expert_binder_selects_streaming_when_enabled -q`
Expected: FAIL with `ImportError: cannot import name '_choose_expert_binder'`.

- [ ] **Step 3: Write minimal implementation**

Add the small selector to `src/mlx_vq/models/glm52_composite_loader.py` and use it
at the expert-binding site (`:1509-1514`):

```python
# src/mlx_vq/models/glm52_composite_loader.py
def _choose_expert_binder(streaming, *, eager, streaming as _streaming=None):  # see note
    ...
```

> Python forbids `as` in a parameter list — use this exact signature instead:

```python
# src/mlx_vq/models/glm52_composite_loader.py
def _choose_expert_binder(streaming_config, *, eager, streaming):
    """Return the binder callable to run based on the streaming config."""
    if streaming_config is not None and getattr(streaming_config, "enabled", False):
        return streaming
    return eager
```

At the binding site inside `load_authenticated_glm52_composite`, wrap the
existing eager call and the new streaming call as zero-arg thunks and dispatch:

```python
    def _bind_eager():
        return bind_glm52_vq_experts(
            model, validated.routed_artifact_dir, profile=profile, strict=True
        ), None

    def _bind_streaming():
        from mlx_vq.models.glm52_vq_adapter import bind_glm52_vq_experts_streaming

        store = bind_glm52_vq_experts_streaming(
            model, validated.routed_artifact_dir, profile=profile, config=streaming
        )
        return store.cache.resident_layers(), store

    bound, expert_store = _choose_expert_binder(
        streaming, eager=_bind_eager, streaming=_bind_streaming
    )()
```

Add `streaming: "StreamingConfig | None" = None` to the
`load_authenticated_glm52_composite` signature and attach `expert_store` to the
returned `GLM52CompositeLoadReport` (add an `expert_store: object | None = None`
field defaulting to `None`).

Re-export:

```python
# src/ramp/runtime/expert_store.py
from __future__ import annotations

from mlx_vq.runtime.expert_store import (  # noqa: F401
    ExpertResidencyCache,
    ExpertStore,
    StreamingConfig,
    StreamingSwitchGLUProxy,
    bind_streaming_experts,
    shard_group_nbytes,
)
```

Add to `README.md` Architecture section:

```markdown
- **Expert streaming (optional):** when a KEEP artifact's routed experts exceed
  RAM, `load_authenticated_glm52_composite(..., streaming=StreamingConfig(...))`
  binds each sparse layer to a streaming proxy backed by a byte-budgeted LRU
  `ExpertStore` (`src/mlx_vq/runtime/expert_store.py`). Experts load from disk on
  demand and stay on the 8-bit NAX path. Streaming runs page from disk by design
  and are benchmarked separately from resident Lane S.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_expert_store.py tests/test_expert_streaming_bind.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/models/glm52_composite_loader.py src/ramp/runtime/expert_store.py README.md tests/test_expert_streaming_bind.py
git commit -m "feat: route composite loader through optional expert streaming"
```

---

### Task 7: Streaming smoke benchmark (labeled, manual-heavy)

**Files:**
- Create: `benchmarks/bench_glm52_expert_streaming.py`
- Test: `tests/test_expert_streaming_bind.py` (arg-parsing + report-shape only)

**Interfaces:**
- Produces: a script that, given an artifact dir + `--expert-cache-gb`, loads the
  model with streaming, generates N tokens, and prints a JSON row with
  `tokens_per_second`, `cache_hits`, `cache_loads`, `cache_evicts`,
  `pageouts_delta`, `swapouts_delta`, tagged `"mode": "streaming"`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_expert_streaming_bind.py
def test_streaming_bench_report_shape() -> None:
    from benchmarks.bench_glm52_expert_streaming import summarize_streaming_run

    row = summarize_streaming_run(
        tokens=32,
        elapsed_seconds=8.0,
        events=[("load", 1), ("hit", 1), ("evict", 1), ("hit", 2)],
        pageouts_delta=0,
        swapouts_delta=0,
    )
    assert row["mode"] == "streaming"
    assert row["tokens_per_second"] == 4.0
    assert row["cache_hits"] == 2
    assert row["cache_loads"] == 1
    assert row["cache_evicts"] == 1
    assert row["memory_clean"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_expert_streaming_bind.py::test_streaming_bench_report_shape -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks.bench_glm52_expert_streaming'`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/bench_glm52_expert_streaming.py
from __future__ import annotations

import argparse
import json


def summarize_streaming_run(
    *,
    tokens: int,
    elapsed_seconds: float,
    events: list[tuple[str, int]],
    pageouts_delta: int,
    swapouts_delta: int,
) -> dict:
    kinds = [kind for kind, _layer in events]
    return {
        "mode": "streaming",
        "tokens": tokens,
        "tokens_per_second": tokens / elapsed_seconds if elapsed_seconds else 0.0,
        "cache_hits": kinds.count("hit"),
        "cache_loads": kinds.count("load"),
        "cache_evicts": kinds.count("evict"),
        "pageouts_delta": pageouts_delta,
        "swapouts_delta": swapouts_delta,
        # Streaming pages *its own file reads*; "memory_clean" tracks OS swap of
        # the process, which must stay zero even while streaming.
        "memory_clean": pageouts_delta == 0 and swapouts_delta == 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench_glm52_expert_streaming")
    parser.add_argument("artifact_dir")
    parser.add_argument("--expert-cache-gb", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="Explain mixture-of-experts briefly.")
    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - heavy path
    args = build_parser().parse_args(argv)
    # Heavy: acquire .keep-heavy-job.lock, load the model with
    #   StreamingConfig(budget_bytes=int(args.expert_cache_gb * 1024**3)),
    #   collect_metric_snapshot() before/after, generate args.max_new_tokens,
    #   subscribe to store.cache.on_event to gather events, then:
    print(json.dumps({"note": "run on a Metal host with the artifact present"}))
    return 0
```

> The pure `summarize_streaming_run` is unit-tested here; `main()` is the heavy,
> host-only path (guarded `# pragma: no cover`). It must acquire
> `.keep-heavy-job.lock` and use `collect_metric_snapshot`
> (`src/mlx_vq/benchmark/metrics.py:178`) for pageouts/swapouts. Label every row
> `"mode": "streaming"` and never compare it to a Lane S row.

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_expert_streaming_bind.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/bench_glm52_expert_streaming.py tests/test_expert_streaming_bind.py
git commit -m "feat: add labeled streaming smoke benchmark scaffold"
```

---

## Self-Review

- **Spec coverage (F1):** disk-backed per-layer loading (Tasks 1, 3), byte budget
  + LRU + pinning (Task 2), transparent NAX-preserving delegation (Task 4),
  model binding (Task 5), opt-in loader routing that never regresses the resident
  path (Task 6), labeled streaming benchmark (Task 7).
- **NAX speed-wall guard:** Task 4 hands the kernel the resolved
  `QuantizedVQSwitchGLU` unchanged; no dispatch change anywhere.
- **Placeholder scan:** none. The one Python-syntax trap (`as` in a parameter
  list) is called out and the correct signature given.
- **Type consistency:** `StreamingConfig`, `ExpertStore`,
  `ExpertResidencyCache`, `StreamingSwitchGLUProxy`, `bind_streaming_experts`,
  `bind_glm52_vq_experts_streaming` are used identically across tasks and
  re-exports. `on_event(kind, layer)` signature is stable for Plan B to consume.
- **Deferred (noted):** per-expert (sub-layer) streaming; real end-to-end tok/s
  numbers (Task 7 `main` is host-only). Neither is silently dropped.
