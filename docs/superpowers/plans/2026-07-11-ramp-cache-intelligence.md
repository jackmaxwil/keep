# RAMP Cache Intelligence (F6 + F2 + F4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Depends on Plan A** ([`2026-07-11-ramp-expert-streaming.md`](2026-07-11-ramp-expert-streaming.md)):
> requires `ExpertStore`, `ExpertResidencyCache`, `StreamingConfig`, and the
> `on_event(kind, layer)` hook. Do not start until Plan A's tasks are merged.

**Goal:** Make the streamed expert cache smart — size its budget to the host's
RAM automatically (F6), learn which experts/layers are hot and pin them across
sessions via a durable usage file (F2), and prefetch the next layer's shards
while the current layer computes (F4).

**Architecture:** Three cooperating modules over Plan A's `ExpertStore`.
`memory_budget` detects RAM and derives a software cache budget (never touching
wired limit by default). `usage_stats` subscribes to the store's `on_event`
stream, persists per-layer access counts to `.keep_usage.json`, and computes a
pin set for the next session. `prefetch` predicts the next sparse layer (the
per-layer visit order is deterministic) and warms it on a background thread. A
small fan-out lets the store feed both the recorder and the prefetcher.

**Tech Stack:** Python 3.11+, MLX (`mx.device_info`, `mx.set_wired_limit`),
`threading`/`queue`, pytest.

## Global Constraints

Inherited from [`2026-07-11-ramp-beyond-ram-roadmap.md`](2026-07-11-ramp-beyond-ram-roadmap.md).
Load-bearing here:

- **Never set wired limit by default.** Leave `GLM_MLX_WIRED_LIMIT_GB` unset.
  F6 sizes the *software* `ExpertResidencyCache` budget only. `mx.set_wired_limit`
  is called **only** through an explicit, off-by-default opt-in, with a test
  proving the default path does not call it.
- **Preserve the NAX path.** These modules choose *what* to cache/prefetch, never
  *how* experts are computed. No kernel dispatch changes.
- Tests run with **no Metal and no model** — inject RAM detectors, size
  functions, and fake stores.
- `from __future__ import annotations`; pytest; `tmp_path`. New code under
  `src/mlx_vq/runtime/`; `ramp` re-exports as the final task. One commit/task.

## File Structure

- Create `src/mlx_vq/runtime/memory_budget.py` — F6.
- Create `src/mlx_vq/runtime/usage_stats.py` — F2.
- Create `src/mlx_vq/runtime/prefetch.py` — F4.
- Modify `src/mlx_vq/runtime/expert_store.py` — add `EventFanout` + a
  `streaming_config_for_host(...)` convenience that wires all three.
- Modify `src/mlx_vq/build/cli.py` — `keep runtime-budget` subcommand.
- Create `tests/test_memory_budget.py`, `tests/test_usage_stats.py`,
  `tests/test_prefetch.py`, `tests/test_runtime_budget_cli.py`.
- Create `src/ramp/runtime/{memory_budget,usage_stats,prefetch}.py`.

## Verified facts (2026-07-11)

- `mx.device_info()` exists (`.venv/.../mlx/core/__init__.pyi:75`) and returns a
  dict of device properties including memory sizes. `mx.set_wired_limit`,
  `mx.set_memory_limit`, `mx.set_cache_limit` all exist. `metrics.py` already
  uses `subprocess` (for `vm_stat`), so `sysctl -n hw.memsize` is in-style for
  RAM detection.
- Plan A's `ExpertStore(..., on_event=callback)` fires `callback(kind, layer)`
  with `kind in {"hit","load","evict"}`; `ExpertResidencyCache` accepts the same.

---

### Task 1: RAM detection + cache-budget math (F6 core)

**Files:**
- Create: `src/mlx_vq/runtime/memory_budget.py`
- Test: `tests/test_memory_budget.py`

**Interfaces:**
- Produces:
  `detect_total_ram_bytes(detector: Callable[[], int] | None = None) -> int`;
  `resolve_expert_cache_budget(*, total_ram_bytes: int, reserve_bytes: int, fraction: float, min_bytes: int) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory_budget.py
from __future__ import annotations

import pytest

from mlx_vq.runtime.memory_budget import (
    detect_total_ram_bytes,
    resolve_expert_cache_budget,
)


def test_detect_total_ram_uses_injected_detector() -> None:
    assert detect_total_ram_bytes(detector=lambda: 64 * 1024**3) == 64 * 1024**3


def test_resolve_budget_takes_fraction_of_available() -> None:
    budget = resolve_expert_cache_budget(
        total_ram_bytes=64 * 1024**3,
        reserve_bytes=16 * 1024**3,   # dense layer + OS + activations
        fraction=0.5,
        min_bytes=2 * 1024**3,
    )
    # (64 - 16) * 0.5 = 24 GiB
    assert budget == 24 * 1024**3


def test_resolve_budget_clamps_to_min_when_ram_tiny() -> None:
    budget = resolve_expert_cache_budget(
        total_ram_bytes=18 * 1024**3,
        reserve_bytes=16 * 1024**3,
        fraction=0.5,
        min_bytes=2 * 1024**3,
    )
    assert budget == 2 * 1024**3  # (18-16)*0.5 = 1 GiB → clamped up to min


def test_resolve_budget_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="fraction"):
        resolve_expert_cache_budget(
            total_ram_bytes=1, reserve_bytes=0, fraction=1.5, min_bytes=1
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_memory_budget.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_vq.runtime.memory_budget'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mlx_vq/runtime/memory_budget.py
from __future__ import annotations

import subprocess
from typing import Callable


def _detect_via_sysctl() -> int:
    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return int(result.stdout.strip())


def _detect_via_mlx() -> int:
    import mlx.core as mx

    info = mx.device_info()
    for key in ("memory_size", "max_recommended_working_set_size"):
        value = info.get(key)
        if isinstance(value, int) and value > 0:
            return value
    raise RuntimeError("mx.device_info() exposed no usable memory size")


def detect_total_ram_bytes(detector: Callable[[], int] | None = None) -> int:
    if detector is not None:
        return detector()
    try:
        return _detect_via_sysctl()
    except (OSError, subprocess.SubprocessError, ValueError):
        return _detect_via_mlx()


def resolve_expert_cache_budget(
    *,
    total_ram_bytes: int,
    reserve_bytes: int,
    fraction: float,
    min_bytes: int,
) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    available = max(0, total_ram_bytes - reserve_bytes)
    budget = int(available * fraction)
    return max(min_bytes, budget)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_memory_budget.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/memory_budget.py tests/test_memory_budget.py
git commit -m "feat: detect host RAM and derive expert cache budget"
```

---

### Task 2: Opt-in wired-limit guard (F6 safety)

**Files:**
- Modify: `src/mlx_vq/runtime/memory_budget.py`
- Test: `tests/test_memory_budget.py`

**Interfaces:**
- Produces: `apply_wired_limit(limit_bytes: int, *, enabled: bool, setter: Callable[[int], int] | None = None) -> bool`
  — returns `True` iff it actually set the limit. Default `enabled=False` is a
  no-op that never calls the setter.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_memory_budget.py
from mlx_vq.runtime.memory_budget import apply_wired_limit


def test_apply_wired_limit_is_noop_by_default() -> None:
    calls: list[int] = []
    did = apply_wired_limit(123, enabled=False, setter=calls.append)
    assert did is False
    assert calls == []  # default path never touches wired limit


def test_apply_wired_limit_sets_only_when_enabled() -> None:
    calls: list[int] = []
    did = apply_wired_limit(123, enabled=True, setter=calls.append)
    assert did is True
    assert calls == [123]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_memory_budget.py -k wired -q`
Expected: FAIL with `ImportError: cannot import name 'apply_wired_limit'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/runtime/memory_budget.py
def apply_wired_limit(
    limit_bytes: int,
    *,
    enabled: bool,
    setter: Callable[[int], int] | None = None,
) -> bool:
    if not enabled:
        return False
    if setter is None:
        import mlx.core as mx

        setter = mx.set_wired_limit
    setter(limit_bytes)
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_memory_budget.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/memory_budget.py tests/test_memory_budget.py
git commit -m "feat: gate wired-limit behind explicit opt-in"
```

---

### Task 3: `UsageRecorder` — durable per-layer access counts (F2 core)

**Files:**
- Create: `src/mlx_vq/runtime/usage_stats.py`
- Test: `tests/test_usage_stats.py`

**Interfaces:**
- Produces:
  `UsageRecorder(path: Path)` with `.record(kind: str, layer: int) -> None`
  (counts `hit`+`load` as accesses; `evict` ignored), `.counts() -> dict[int,int]`,
  `.flush() -> None` (persist to `.keep_usage.json`), and classmethod
  `.load(path) -> UsageRecorder` (resume prior counts).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_usage_stats.py
from __future__ import annotations

from pathlib import Path

from mlx_vq.runtime.usage_stats import UsageRecorder


def test_recorder_counts_accesses_and_ignores_evict(tmp_path: Path) -> None:
    rec = UsageRecorder(tmp_path / ".keep_usage.json")
    for kind, layer in [("load", 3), ("hit", 3), ("hit", 3), ("evict", 3), ("load", 5)]:
        rec.record(kind, layer)
    assert rec.counts() == {3: 3, 5: 1}


def test_recorder_persists_and_resumes(tmp_path: Path) -> None:
    path = tmp_path / ".keep_usage.json"
    rec = UsageRecorder(path)
    rec.record("load", 7)
    rec.record("hit", 7)
    rec.flush()

    resumed = UsageRecorder.load(path)
    resumed.record("hit", 7)
    assert resumed.counts() == {7: 3}  # prior 2 + new 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_usage_stats.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_vq.runtime.usage_stats'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mlx_vq/runtime/usage_stats.py
from __future__ import annotations

import json
from pathlib import Path

_ACCESS_KINDS = frozenset({"hit", "load"})


class UsageRecorder:
    def __init__(self, path: Path, counts: dict[int, int] | None = None) -> None:
        self._path = Path(path)
        self._counts: dict[int, int] = dict(counts or {})

    def record(self, kind: str, layer: int) -> None:
        if kind in _ACCESS_KINDS:
            self._counts[layer] = self._counts.get(layer, 0) + 1

    def counts(self) -> dict[int, int]:
        return dict(self._counts)

    def flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"counts": {str(k): v for k, v in self._counts.items()}}
        self._path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "UsageRecorder":
        path = Path(path)
        if not path.exists():
            return cls(path)
        raw = json.loads(path.read_text())
        counts = {int(k): int(v) for k, v in raw.get("counts", {}).items()}
        return cls(path, counts=counts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_usage_stats.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/usage_stats.py tests/test_usage_stats.py
git commit -m "feat: add durable per-layer expert usage recorder"
```

---

### Task 4: `pinned_layers_from_usage` — greedy pin set under a byte budget (F2 payoff)

**Files:**
- Modify: `src/mlx_vq/runtime/usage_stats.py`
- Test: `tests/test_usage_stats.py`

**Interfaces:**
- Consumes: `UsageRecorder`.
- Produces:
  `pinned_layers_from_usage(counts: dict[int,int], *, size_fn: Callable[[int], int], pin_budget_bytes: int) -> tuple[int, ...]`
  — most-used layers first, greedily admitted until the next one would exceed
  `pin_budget_bytes`; ties broken by lower layer index for determinism.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_usage_stats.py
from mlx_vq.runtime.usage_stats import pinned_layers_from_usage


def test_pins_hottest_layers_within_budget() -> None:
    counts = {1: 100, 2: 50, 3: 10, 4: 5}
    pins = pinned_layers_from_usage(
        counts,
        size_fn=lambda _layer: 100,  # each layer 100 bytes
        pin_budget_bytes=250,        # room for two
    )
    assert pins == (1, 2)  # hottest two, budget-bounded, sorted for determinism


def test_pins_break_ties_by_layer_index() -> None:
    counts = {5: 10, 2: 10, 8: 10}
    pins = pinned_layers_from_usage(
        counts, size_fn=lambda _l: 100, pin_budget_bytes=200
    )
    assert pins == (2, 5)  # equal counts → lowest indices win, then sorted


def test_empty_counts_pins_nothing() -> None:
    assert pinned_layers_from_usage({}, size_fn=lambda _l: 1, pin_budget_bytes=10) == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_usage_stats.py -k pin -q`
Expected: FAIL with `ImportError: cannot import name 'pinned_layers_from_usage'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/runtime/usage_stats.py
from typing import Callable


def pinned_layers_from_usage(
    counts: dict[int, int],
    *,
    size_fn: Callable[[int], int],
    pin_budget_bytes: int,
) -> tuple[int, ...]:
    # Sort by descending count, then ascending layer index for stable ties.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    pinned: list[int] = []
    spent = 0
    for layer, _count in ranked:
        cost = size_fn(layer)
        if spent + cost > pin_budget_bytes:
            continue
        pinned.append(layer)
        spent += cost
    return tuple(sorted(pinned))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_usage_stats.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/usage_stats.py tests/test_usage_stats.py
git commit -m "feat: derive next-session pin set from usage stats"
```

---

### Task 5: Predictive prefetch (F4)

**Files:**
- Create: `src/mlx_vq/runtime/prefetch.py`
- Test: `tests/test_prefetch.py`

**Interfaces:**
- Produces:
  `next_sparse_layer(current: int, sparse_layers: Sequence[int]) -> int | None`
  (the layer visited after `current` in the fixed sparse order; wraps to the
  first for the last layer; `None` if `current` unknown);
  `BackgroundPrefetcher(warm: Callable[[int], object], sparse_layers)` with
  `.on_event(kind, layer) -> None` (enqueues the successor on access),
  `.drain() -> None` (process queued warms — used synchronously in tests),
  `.start()/.stop()` (background thread for production).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prefetch.py
from __future__ import annotations

from mlx_vq.runtime.prefetch import BackgroundPrefetcher, next_sparse_layer


def test_next_sparse_layer_follows_fixed_order() -> None:
    layers = [3, 6, 9, 12]
    assert next_sparse_layer(3, layers) == 6
    assert next_sparse_layer(9, layers) == 12
    assert next_sparse_layer(12, layers) == 3  # wraps for the next token
    assert next_sparse_layer(99, layers) is None


def test_prefetcher_warms_successor_once_on_access() -> None:
    warmed: list[int] = []
    pf = BackgroundPrefetcher(warm=warmed.append, sparse_layers=[3, 6, 9])
    pf.on_event("load", 3)   # visiting 3 → predict 6
    pf.on_event("hit", 6)    # visiting 6 → predict 9
    pf.on_event("evict", 3)  # evict must NOT trigger a prefetch
    pf.drain()
    assert warmed == [6, 9]


def test_prefetcher_dedups_pending_target() -> None:
    warmed: list[int] = []
    pf = BackgroundPrefetcher(warm=warmed.append, sparse_layers=[3, 6, 9])
    pf.on_event("hit", 3)  # predict 6
    pf.on_event("load", 3)  # predict 6 again before drain → dedup
    pf.drain()
    assert warmed == [6]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_prefetch.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_vq.runtime.prefetch'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mlx_vq/runtime/prefetch.py
from __future__ import annotations

import queue
import threading
from typing import Callable, Sequence


def next_sparse_layer(current: int, sparse_layers: Sequence[int]) -> int | None:
    layers = list(sparse_layers)
    try:
        idx = layers.index(current)
    except ValueError:
        return None
    return layers[(idx + 1) % len(layers)]


class BackgroundPrefetcher:
    """Warms the predicted next sparse layer while the current one computes."""

    def __init__(self, warm: Callable[[int], object], sparse_layers: Sequence[int]) -> None:
        self._warm = warm
        self._layers = list(sparse_layers)
        self._queue: "queue.Queue[int]" = queue.Queue()
        self._pending: set[int] = set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def on_event(self, kind: str, layer: int) -> None:
        if kind not in ("hit", "load"):
            return
        target = next_sparse_layer(layer, self._layers)
        if target is None:
            return
        with self._lock:
            if target in self._pending:
                return
            self._pending.add(target)
        self._queue.put(target)

    def _warm_target(self, target: int) -> None:
        try:
            self._warm(target)
        finally:
            with self._lock:
                self._pending.discard(target)

    def drain(self) -> None:
        """Synchronously process all queued warms (test + single-thread use)."""
        while True:
            try:
                target = self._queue.get_nowait()
            except queue.Empty:
                return
            self._warm_target(target)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                target = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._warm_target(target)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ramp-prefetch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
```

> Why per-layer prediction is ~perfect: sparse layers are visited in a fixed
> order every forward pass, so the successor is deterministic — no 71.6% guess
> needed at this granularity. Colibri's router-probability prefetch applies at
> the finer *per-expert* granularity, which is future work tied to Plan A's
> deferred per-expert streaming. `warm` is `store.switch_glu`, which loads the
> shard into the residency cache; the LRU keeps prefetched layers only if the
> budget allows.

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_prefetch.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/prefetch.py tests/test_prefetch.py
git commit -m "feat: add deterministic next-layer background prefetcher"
```

---

### Task 6: Fan-out wiring + `streaming_config_for_host` + `keep runtime-budget` + re-exports

**Files:**
- Modify: `src/mlx_vq/runtime/expert_store.py` (add `EventFanout`,
  `streaming_config_for_host`)
- Modify: `src/mlx_vq/build/cli.py` (add `runtime-budget` subcommand)
- Create: `src/ramp/runtime/{memory_budget,usage_stats,prefetch}.py`
- Test: `tests/test_prefetch.py` (fan-out), `tests/test_runtime_budget_cli.py`

**Interfaces:**
- Produces:
  `EventFanout(*handlers: Callable[[str, int], None])` callable as
  `fanout(kind, layer)` dispatching to every handler;
  `streaming_config_for_host(sparse_layers, size_fn, *, usage_path=None, fraction=0.5, reserve_bytes, min_bytes, detector=None) -> StreamingConfig`
  (budget from F6, pins from F2 usage if present);
  CLI `keep runtime-budget [--usage PATH] [--fraction F] [--reserve-gb G]`
  printing detected RAM + recommended budget + pin set size.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_prefetch.py
from mlx_vq.runtime.expert_store import EventFanout


def test_event_fanout_dispatches_to_all_handlers() -> None:
    a: list[tuple[str, int]] = []
    b: list[tuple[str, int]] = []
    fanout = EventFanout(lambda k, l: a.append((k, l)), lambda k, l: b.append((k, l)))
    fanout("load", 3)
    fanout("hit", 6)
    assert a == b == [("load", 3), ("hit", 6)]
```

```python
# tests/test_runtime_budget_cli.py
from __future__ import annotations

from mlx_vq.build.cli import main as cli_main


def test_runtime_budget_prints_recommendation(monkeypatch, capsys) -> None:
    import mlx_vq.runtime.memory_budget as mb

    monkeypatch.setattr(mb, "_detect_via_sysctl", lambda: 64 * 1024**3)
    rc = cli_main(["runtime-budget", "--fraction", "0.5", "--reserve-gb", "16"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "total RAM" in out
    assert "24" in out  # (64-16)*0.5 GiB recommended
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_prefetch.py::test_event_fanout_dispatches_to_all_handlers tests/test_runtime_budget_cli.py -q`
Expected: FAIL — `EventFanout` import error and unknown `runtime-budget` command.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/runtime/expert_store.py
class EventFanout:
    def __init__(self, *handlers: Callable[[str, int], None]) -> None:
        self._handlers = handlers

    def __call__(self, kind: str, layer: int) -> None:
        for handler in self._handlers:
            handler(kind, layer)


def streaming_config_for_host(
    sparse_layers,
    size_fn: Callable[[int], int],
    *,
    usage_path=None,
    fraction: float = 0.5,
    reserve_bytes: int,
    min_bytes: int,
    detector: Callable[[], int] | None = None,
) -> "StreamingConfig":
    from mlx_vq.runtime.memory_budget import (
        detect_total_ram_bytes,
        resolve_expert_cache_budget,
    )

    total = detect_total_ram_bytes(detector=detector)
    budget = resolve_expert_cache_budget(
        total_ram_bytes=total,
        reserve_bytes=reserve_bytes,
        fraction=fraction,
        min_bytes=min_bytes,
    )
    pinned: tuple[int, ...] = ()
    if usage_path is not None:
        from pathlib import Path

        from mlx_vq.runtime.usage_stats import (
            UsageRecorder,
            pinned_layers_from_usage,
        )

        if Path(usage_path).exists():
            counts = UsageRecorder.load(usage_path).counts()
            pinned = pinned_layers_from_usage(
                counts, size_fn=size_fn, pin_budget_bytes=budget // 2
            )
    return StreamingConfig(budget_bytes=budget, pinned_layers=pinned, enabled=True)
```

Register the CLI subcommand in `build_parser()` (`src/mlx_vq/build/cli.py:418`):

```python
    runtime_budget = subparsers.add_parser(
        "runtime-budget", help="report detected RAM and a recommended expert cache budget"
    )
    runtime_budget.add_argument("--usage", default=None)
    runtime_budget.add_argument("--fraction", type=float, default=0.5)
    runtime_budget.add_argument("--reserve-gb", type=float, default=16.0)
    runtime_budget.set_defaults(func=_cmd_runtime_budget)
```

```python
# handler with the other _cmd_* functions
def _cmd_runtime_budget(args: argparse.Namespace) -> int:
    from mlx_vq.runtime.memory_budget import (
        detect_total_ram_bytes,
        resolve_expert_cache_budget,
    )

    total = detect_total_ram_bytes()
    reserve = int(args.reserve_gb * 1024**3)
    budget = resolve_expert_cache_budget(
        total_ram_bytes=total,
        reserve_bytes=reserve,
        fraction=args.fraction,
        min_bytes=2 * 1024**3,
    )
    gib = 1024**3
    print(f"total RAM: {total / gib:.1f} GiB")
    print(f"reserve:   {reserve / gib:.1f} GiB")
    print(f"recommended expert cache budget: {budget / gib:.1f} GiB")
    return 0
```

Add the three ramp re-exports:

```python
# src/ramp/runtime/memory_budget.py
from __future__ import annotations
from mlx_vq.runtime.memory_budget import (  # noqa: F401
    apply_wired_limit, detect_total_ram_bytes, resolve_expert_cache_budget,
)
```

```python
# src/ramp/runtime/usage_stats.py
from __future__ import annotations
from mlx_vq.runtime.usage_stats import (  # noqa: F401
    UsageRecorder, pinned_layers_from_usage,
)
```

```python
# src/ramp/runtime/prefetch.py
from __future__ import annotations
from mlx_vq.runtime.prefetch import BackgroundPrefetcher, next_sparse_layer  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_memory_budget.py tests/test_usage_stats.py tests/test_prefetch.py tests/test_runtime_budget_cli.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/expert_store.py src/mlx_vq/build/cli.py src/ramp/runtime tests/test_prefetch.py tests/test_runtime_budget_cli.py
git commit -m "feat: wire cache intelligence (fanout, host config, runtime-budget CLI)"
```

---

### Task 7: End-to-end wiring into `ExpertStore` construction (integration seam)

**Files:**
- Modify: `src/mlx_vq/models/glm52_vq_adapter.py`
  (`bind_glm52_vq_experts_streaming` from Plan A)
- Test: `tests/test_usage_stats.py` (store + recorder + prefetcher fan-out)

**Interfaces:**
- Consumes: `ExpertStore`, `EventFanout`, `UsageRecorder`, `BackgroundPrefetcher`.
- Produces: `bind_glm52_vq_experts_streaming(..., usage_path=None, prefetch=False)`
  — when provided, attaches a `UsageRecorder` and/or `BackgroundPrefetcher` to the
  store via `EventFanout` and returns them on the store for the caller to flush.

- [ ] **Step 1: Write the failing test**

Test the composed event path with a fake store built directly (no model):

```python
# append to tests/test_usage_stats.py
from mlx_vq.runtime.expert_store import (
    EventFanout,
    ExpertStore,
    StreamingConfig,
)
from mlx_vq.runtime.prefetch import BackgroundPrefetcher


def test_store_events_feed_recorder_and_prefetcher(tmp_path) -> None:
    usage_path = tmp_path / ".keep_usage.json"
    recorder = UsageRecorder(usage_path)
    warmed: list[int] = []
    prefetcher = BackgroundPrefetcher(warm=warmed.append, sparse_layers=[1, 2, 3])
    fanout = EventFanout(recorder.record, prefetcher.on_event)

    store = ExpertStore(
        tmp_path,
        sparse_layers=(1, 2, 3),
        config=StreamingConfig(budget_bytes=1000),
        loader=lambda _d, layer: f"glu-{layer}",
        size_fn=lambda _l: 10,
        on_event=fanout,
    )
    store.switch_glu(1)   # load 1 → record + predict 2
    store.switch_glu(1)   # hit 1 → record + predict 2 (deduped)
    prefetcher.drain()

    assert recorder.counts() == {1: 2}
    assert warmed == [2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_usage_stats.py::test_store_events_feed_recorder_and_prefetcher -q`
Expected: PASS or FAIL depending on Plan A's `on_event` support — if it fails,
it is because the composed handlers are not yet exercised end-to-end; the fix is
purely in the binder wiring below (no new store behavior needed).

- [ ] **Step 3: Write minimal implementation**

Extend `bind_glm52_vq_experts_streaming` (added in Plan A Task 5) to optionally
compose the recorder and prefetcher:

```python
# src/mlx_vq/models/glm52_vq_adapter.py  (extend the streaming binder)
def bind_glm52_vq_experts_streaming(
    model,
    artifact_dir,
    *,
    profile,
    config,
    layers=None,
    usage_path=None,
    prefetch: bool = False,
):
    from mlx_vq.runtime.expert_store import (
        EventFanout,
        ExpertStore,
        bind_streaming_experts,
        shard_group_nbytes,
    )
    from mlx_vq.runtime.prefetch import BackgroundPrefetcher
    from mlx_vq.runtime.usage_stats import UsageRecorder

    expected_sparse_layers = tuple(
        layer_idx
        for layer_idx in range(model.args.num_hidden_layers)
        if is_glm52_sparse_layer(model.args, layer_idx)
    )
    requested = expected_sparse_layers if layers is None else tuple(layers)

    handlers = []
    recorder = None
    if usage_path is not None:
        recorder = UsageRecorder.load(usage_path)
        handlers.append(recorder.record)

    prefetcher = None
    # The store is created after handlers so the prefetcher can warm through it.
    def _warm(layer: int):
        return store.switch_glu(layer)

    if prefetch:
        prefetcher = BackgroundPrefetcher(warm=_warm, sparse_layers=requested)
        handlers.append(prefetcher.on_event)

    on_event = EventFanout(*handlers) if handlers else None
    store = ExpertStore(
        artifact_dir,
        sparse_layers=requested,
        config=config,
        size_fn=lambda layer: shard_group_nbytes(artifact_dir, layer),
        on_event=on_event,
    )
    bind_streaming_experts(model, store, sparse_layers=requested)
    # Expose companions for the caller to start()/flush().
    store.usage_recorder = recorder
    store.prefetcher = prefetcher
    return store
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_usage_stats.py tests/test_prefetch.py tests/test_memory_budget.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/models/glm52_vq_adapter.py tests/test_usage_stats.py
git commit -m "feat: compose usage recorder and prefetcher onto expert store"
```

---

## Self-Review

- **Spec coverage:** F6 RAM auto-size (Tasks 1–2, 6), F2 durable usage +
  cross-session pinning (Tasks 3–4, 7), F4 predictive prefetch (Task 5, 7).
- **Wired-limit safety:** Task 2 proves the default path never calls
  `set_wired_limit`; `GLM_MLX_WIRED_LIMIT_GB` stays unset.
- **NAX path:** untouched — these modules only choose what to cache/prefetch.
- **Placeholder scan:** none; all steps runnable without a model or Metal.
- **Type consistency:** `on_event(kind, layer)`, `EventFanout`, `UsageRecorder`,
  `BackgroundPrefetcher`, `StreamingConfig`, `ExpertStore` names match Plan A and
  are used consistently through Task 7 and the re-exports.
- **Deferred (noted):** router-probability *per-expert* prefetch (colibri's
  71.6%) waits on Plan A's per-expert streaming; per-layer prediction here is
  deterministic and needs no probability model.
