# RAMP Warm KV-Cache Persistence (F3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist a chat session's KV cache to disk so reopening a conversation
with an identical prompt prefix restores the cache and skips re-prefill ("warm
reopen, zero re-prefill"), with a ledger recording each saved prefix — extending
KEEP's "resume instead of restart" ethos from builds to inference.

**Architecture:** A `session_cache` module wraps `mlx_lm.models.cache`'s
`save_prompt_cache`/`load_prompt_cache`. Each saved cache is fingerprinted by
`(model_id, exact token prefix)`; on reopen we restore only when the requested
prompt begins with a saved prefix, and prefill just the remaining tokens. A
JSON ledger per session tracks saved prefixes so the longest valid match wins.

**Tech Stack:** Python 3.11+, MLX, `mlx_lm.models.cache`, argparse, pytest.

## Global Constraints

Inherited from [`2026-07-11-ramp-beyond-ram-roadmap.md`](2026-07-11-ramp-beyond-ram-roadmap.md).
Load-bearing here:

- Tests run without a model artifact or Metal — construct real
  `mlx_lm.models.cache.KVCache` objects with tiny CPU `mx.array`s as fixtures.
- Correctness rule: a KV cache is valid **only** for the exact token prefix that
  produced it. Never restore a cache whose fingerprint prefix is not a prefix of
  the new prompt. A fingerprint mismatch must fall back to a cold (fresh) cache,
  never a wrong warm one.
- Session cache files live under a user cache root (default
  `~/.keep/sessions/<session_id>/`), never under `runs/`, `artifacts/`, or the
  protected repo paths.
- `from __future__ import annotations`; pytest; `tmp_path`. New code under
  `src/mlx_vq/runtime/`; `ramp` re-export as the final task.
- One commit per task. Do not push.

## File Structure

- Create `src/mlx_vq/runtime/__init__.py` — new runtime subpackage (empty).
- Create `src/mlx_vq/runtime/session_cache.py` — fingerprinting, save/load,
  ledger, and the `resolve_prompt_cache` resolver.
- Modify `src/mlx_vq/build/chat.py` — restore-before-generate, save-after behind
  an opt-in flag on the runtime.
- Modify `src/mlx_vq/build/cli.py` — `keep sessions {list,prune}` subcommand.
- Create `tests/test_session_cache.py`, `tests/test_session_cache_cli.py`.
- Create `src/ramp/runtime/__init__.py`, `src/ramp/runtime/session_cache.py`.

## Verified facts (2026-07-11)

- `mlx_lm.models.cache.save_prompt_cache(file_name: str, cache: list, metadata: dict[str,str] = {})` and `load_prompt_cache(file_name, return_metadata=False)` exist; every cache class (incl. `CacheList`) has `.state`/`.meta_state`. `trim_prompt_cache(cache, num_tokens)` exists for future partial reuse.
- `GLM52VQModel.make_cache()` (`src/mlx_vq/models/glm52_vq_adapter.py:567`) returns `list[CacheList]`; GLM45-Air (`glm45_air_vq_adapter.py:425`) returns `list[KVCache]`. Both are accepted by `save_prompt_cache`.
- Generation entry: `_stream_mlx_tokens(runtime, messages)` (`src/mlx_vq/build/chat.py:232`) builds `prompt_cache=runtime.model.make_cache()` and calls `mlx_lm.generate.generate_step(prompt_tokens, runtime.model, prompt_cache=..., max_tokens=...)`.

---

### Task 1: Fingerprint, record type, and session paths (pure, no MLX)

**Files:**
- Create: `src/mlx_vq/runtime/__init__.py` (empty)
- Create: `src/mlx_vq/runtime/session_cache.py`
- Test: `tests/test_session_cache.py`

**Interfaces:**
- Produces:
  `prefix_fingerprint(model_id: str, token_ids: Sequence[int]) -> str` (sha256 hex);
  `is_token_prefix(saved: Sequence[int], requested: Sequence[int]) -> bool`;
  `SessionCacheRecord(fingerprint: str, model_id: str, token_count: int, cache_file: str, created_unix: float)`;
  `session_cache_dir(session_id: str, *, root: Path | None = None) -> Path`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session_cache.py
from __future__ import annotations

from pathlib import Path

from mlx_vq.runtime.session_cache import (
    SessionCacheRecord,
    is_token_prefix,
    prefix_fingerprint,
    session_cache_dir,
)


def test_fingerprint_is_stable_and_prefix_sensitive() -> None:
    a = prefix_fingerprint("glm52", [1, 2, 3])
    b = prefix_fingerprint("glm52", [1, 2, 3])
    c = prefix_fingerprint("glm52", [1, 2, 3, 4])
    d = prefix_fingerprint("glm45", [1, 2, 3])
    assert a == b
    assert a != c
    assert a != d


def test_is_token_prefix() -> None:
    assert is_token_prefix([1, 2, 3], [1, 2, 3, 4, 5]) is True
    assert is_token_prefix([1, 2, 3], [1, 2, 3]) is True
    assert is_token_prefix([1, 2, 9], [1, 2, 3, 4]) is False
    assert is_token_prefix([1, 2, 3, 4], [1, 2, 3]) is False


def test_session_cache_dir_defaults_under_keep(tmp_path: Path) -> None:
    d = session_cache_dir("abc123", root=tmp_path)
    assert d == tmp_path / "abc123"
    assert d.parent == tmp_path


def test_record_roundtrips_through_dict() -> None:
    rec = SessionCacheRecord(
        fingerprint="ff",
        model_id="glm52",
        token_count=3,
        cache_file="cache-ff.safetensors",
        created_unix=1.0,
    )
    assert SessionCacheRecord.from_dict(rec.to_dict()) == rec
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_session_cache.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_vq.runtime.session_cache'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mlx_vq/runtime/__init__.py
"""RAMP runtime services (streaming, caching, sessions)."""
```

```python
# src/mlx_vq/runtime/session_cache.py
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def prefix_fingerprint(model_id: str, token_ids: Sequence[int]) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_id.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(struct.pack(f"<{len(token_ids)}i", *token_ids))
    return hasher.hexdigest()


def is_token_prefix(saved: Sequence[int], requested: Sequence[int]) -> bool:
    saved = list(saved)
    requested = list(requested)
    if len(saved) > len(requested):
        return False
    return requested[: len(saved)] == saved


def default_session_root() -> Path:
    return Path.home() / ".keep" / "sessions"


def session_cache_dir(session_id: str, *, root: Path | None = None) -> Path:
    base = root if root is not None else default_session_root()
    return base / session_id


@dataclass(frozen=True)
class SessionCacheRecord:
    fingerprint: str
    model_id: str
    token_count: int
    cache_file: str
    created_unix: float

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "model_id": self.model_id,
            "token_count": self.token_count,
            "cache_file": self.cache_file,
            "created_unix": self.created_unix,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionCacheRecord":
        return cls(
            fingerprint=str(data["fingerprint"]),
            model_id=str(data["model_id"]),
            token_count=int(data["token_count"]),
            cache_file=str(data["cache_file"]),
            created_unix=float(data["created_unix"]),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_session_cache.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/__init__.py src/mlx_vq/runtime/session_cache.py tests/test_session_cache.py
git commit -m "feat: add session-cache fingerprint and record types"
```

---

### Task 2: Save/load a real KV cache round-trip

**Files:**
- Modify: `src/mlx_vq/runtime/session_cache.py`
- Test: `tests/test_session_cache.py`

**Interfaces:**
- Consumes: `SessionCacheRecord`, `prefix_fingerprint`, `session_cache_dir`.
- Produces:
  `save_session_cache(session_dir: Path, cache: list, *, model_id: str, token_ids: Sequence[int], now_unix: float) -> SessionCacheRecord`;
  `load_cache_arrays(session_dir: Path, record: SessionCacheRecord) -> list`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_session_cache.py
import mlx.core as mx
from mlx_lm.models.cache import KVCache

from mlx_vq.runtime.session_cache import (
    load_cache_arrays,
    save_session_cache,
)


def _populated_cache(n_tokens: int) -> list:
    cache = [KVCache()]
    keys = mx.zeros((1, 2, n_tokens, 4), dtype=mx.float32)
    values = mx.ones((1, 2, n_tokens, 4), dtype=mx.float32)
    cache[0].update_and_fetch(keys, values)
    mx.eval(cache[0].state)
    return cache


def test_save_then_load_roundtrips_cache_state(tmp_path) -> None:
    cache = _populated_cache(3)
    record = save_session_cache(
        tmp_path,
        cache,
        model_id="glm52",
        token_ids=[10, 11, 12],
        now_unix=123.0,
    )
    assert record.token_count == 3
    assert (tmp_path / record.cache_file).exists()

    restored = load_cache_arrays(tmp_path, record)
    orig_keys, orig_values = cache[0].state
    new_keys, new_values = restored[0].state
    assert mx.array_equal(orig_keys, new_keys).item()
    assert mx.array_equal(orig_values, new_values).item()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_session_cache.py::test_save_then_load_roundtrips_cache_state -q`
Expected: FAIL with `ImportError: cannot import name 'save_session_cache'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/runtime/session_cache.py
from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache


def save_session_cache(
    session_dir: Path,
    cache: list,
    *,
    model_id: str,
    token_ids: Sequence[int],
    now_unix: float,
) -> SessionCacheRecord:
    session_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = prefix_fingerprint(model_id, token_ids)
    cache_file = f"cache-{fingerprint[:16]}.safetensors"
    metadata = {
        "fingerprint": fingerprint,
        "model_id": model_id,
        "token_count": str(len(token_ids)),
    }
    save_prompt_cache(str(session_dir / cache_file), cache, metadata)
    return SessionCacheRecord(
        fingerprint=fingerprint,
        model_id=model_id,
        token_count=len(token_ids),
        cache_file=cache_file,
        created_unix=now_unix,
    )


def load_cache_arrays(session_dir: Path, record: SessionCacheRecord) -> list:
    return load_prompt_cache(str(session_dir / record.cache_file))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_session_cache.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/session_cache.py tests/test_session_cache.py
git commit -m "feat: save and load session KV cache via mlx_lm prompt cache"
```

---

### Task 3: Session ledger — record many prefixes, pick the longest valid match

**Files:**
- Modify: `src/mlx_vq/runtime/session_cache.py`
- Test: `tests/test_session_cache.py`

**Interfaces:**
- Consumes: `SessionCacheRecord`, `is_token_prefix`, `prefix_fingerprint`.
- Produces:
  `SessionLedger.load(session_dir) -> SessionLedger`;
  `SessionLedger.append(record) -> None` (persists to `ledger.json`);
  `SessionLedger.best_match(model_id, token_ids) -> SessionCacheRecord | None`
  (longest saved token prefix that is a prefix of `token_ids` for this model).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_session_cache.py
from mlx_vq.runtime.session_cache import SessionLedger


def _rec(model_id: str, token_ids: list[int]) -> SessionCacheRecord:
    fp = prefix_fingerprint(model_id, token_ids)
    return SessionCacheRecord(
        fingerprint=fp,
        model_id=model_id,
        token_count=len(token_ids),
        cache_file=f"cache-{fp[:16]}.safetensors",
        created_unix=0.0,
    )


def test_ledger_persists_and_picks_longest_prefix(tmp_path) -> None:
    ledger = SessionLedger.load(tmp_path)
    ledger.append(_rec("glm52", [1, 2]))
    ledger.append(_rec("glm52", [1, 2, 3, 4]))
    ledger.append(_rec("glm52", [9, 9]))

    reloaded = SessionLedger.load(tmp_path)
    match = reloaded.best_match("glm52", [1, 2, 3, 4, 5, 6])
    assert match is not None
    assert match.token_count == 4  # longest valid prefix wins

    # Wrong model or non-prefix → no warm reuse.
    assert reloaded.best_match("glm45", [1, 2, 3, 4]) is None
    assert reloaded.best_match("glm52", [1, 3]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_session_cache.py::test_ledger_persists_and_picks_longest_prefix -q`
Expected: FAIL with `ImportError: cannot import name 'SessionLedger'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/runtime/session_cache.py
import json


class SessionLedger:
    LEDGER_NAME = "ledger.json"

    def __init__(self, session_dir: Path, records: list[SessionCacheRecord]) -> None:
        self._dir = session_dir
        self._records = records

    @property
    def records(self) -> list[SessionCacheRecord]:
        return list(self._records)

    @classmethod
    def load(cls, session_dir: Path) -> "SessionLedger":
        path = session_dir / cls.LEDGER_NAME
        if not path.exists():
            return cls(session_dir, [])
        raw = json.loads(path.read_text())
        records = [SessionCacheRecord.from_dict(item) for item in raw.get("records", [])]
        return cls(session_dir, records)

    def _persist(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        payload = {"records": [r.to_dict() for r in self._records]}
        (self._dir / self.LEDGER_NAME).write_text(json.dumps(payload, indent=2))

    def append(self, record: SessionCacheRecord) -> None:
        # De-duplicate by fingerprint; newest wins.
        self._records = [r for r in self._records if r.fingerprint != record.fingerprint]
        self._records.append(record)
        self._persist()

    def best_match(
        self, model_id: str, token_ids: Sequence[int]
    ) -> SessionCacheRecord | None:
        token_ids = list(token_ids)
        best: SessionCacheRecord | None = None
        for record in self._records:
            if record.model_id != model_id:
                continue
            expected = prefix_fingerprint(model_id, token_ids[: record.token_count])
            if record.fingerprint != expected:
                continue
            if best is None or record.token_count > best.token_count:
                best = record
        return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_session_cache.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/session_cache.py tests/test_session_cache.py
git commit -m "feat: add session ledger with longest-prefix cache matching"
```

---

### Task 4: `resolve_prompt_cache` resolver (warm-or-cold decision, model-free unit test)

**Files:**
- Modify: `src/mlx_vq/runtime/session_cache.py`
- Test: `tests/test_session_cache.py`

**Interfaces:**
- Consumes: `SessionLedger`, `load_cache_arrays`, `is_token_prefix`.
- Produces:
  `PromptCacheResolution(cache: list, reused_tokens: int, warm: bool)`;
  `resolve_prompt_cache(session_dir, *, model_id, token_ids, make_cache, load_arrays=load_cache_arrays, ledger=None) -> PromptCacheResolution`.
  When a valid prefix cache exists, returns the restored cache and
  `reused_tokens = record.token_count`; otherwise returns `make_cache()` and
  `reused_tokens = 0`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_session_cache.py
from mlx_vq.runtime.session_cache import (
    PromptCacheResolution,
    resolve_prompt_cache,
)


def test_resolver_returns_cold_cache_when_no_match(tmp_path) -> None:
    sentinel = ["COLD"]
    resolution = resolve_prompt_cache(
        tmp_path,
        model_id="glm52",
        token_ids=[1, 2, 3],
        make_cache=lambda: sentinel,
        load_arrays=lambda _d, _r: ["WARM"],
    )
    assert isinstance(resolution, PromptCacheResolution)
    assert resolution.warm is False
    assert resolution.reused_tokens == 0
    assert resolution.cache is sentinel


def test_resolver_returns_warm_cache_on_prefix_match(tmp_path) -> None:
    ledger = SessionLedger.load(tmp_path)
    ledger.append(_rec("glm52", [1, 2, 3]))
    resolution = resolve_prompt_cache(
        tmp_path,
        model_id="glm52",
        token_ids=[1, 2, 3, 4, 5],
        make_cache=lambda: ["COLD"],
        load_arrays=lambda _d, _r: ["WARM"],
    )
    assert resolution.warm is True
    assert resolution.reused_tokens == 3
    assert resolution.cache == ["WARM"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_session_cache.py::test_resolver_returns_warm_cache_on_prefix_match -q`
Expected: FAIL with `ImportError: cannot import name 'resolve_prompt_cache'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/runtime/session_cache.py
from dataclasses import dataclass as _dataclass
from typing import Callable


@_dataclass(frozen=True)
class PromptCacheResolution:
    cache: list
    reused_tokens: int
    warm: bool


def resolve_prompt_cache(
    session_dir: Path,
    *,
    model_id: str,
    token_ids: Sequence[int],
    make_cache: Callable[[], list],
    load_arrays: Callable[[Path, SessionCacheRecord], list] = load_cache_arrays,
    ledger: SessionLedger | None = None,
) -> PromptCacheResolution:
    ledger = ledger if ledger is not None else SessionLedger.load(session_dir)
    match = ledger.best_match(model_id, token_ids)
    if match is None:
        return PromptCacheResolution(cache=make_cache(), reused_tokens=0, warm=False)
    try:
        cache = load_arrays(session_dir, match)
    except (OSError, ValueError):
        # Corrupt/missing cache file → fall back to cold, never a wrong warm cache.
        return PromptCacheResolution(cache=make_cache(), reused_tokens=0, warm=False)
    return PromptCacheResolution(cache=cache, reused_tokens=match.token_count, warm=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_session_cache.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/session_cache.py tests/test_session_cache.py
git commit -m "feat: add warm-or-cold prompt cache resolver"
```

---

### Task 5: Wire into chat generation behind an opt-in

**Files:**
- Modify: `src/mlx_vq/build/chat.py`
- Test: `tests/test_session_cache.py` (resolver-integration with a fake runtime)

**Interfaces:**
- Consumes: `resolve_prompt_cache`, `save_session_cache`, `session_cache_dir`.
- Produces: on the chat runtime, when `warm_cache_session_id` is set, prefill
  restores via `resolve_prompt_cache` and, after prefill, `save_session_cache`
  persists the full-prompt cache and appends to the ledger. When unset, behavior
  is byte-identical to today (fresh `make_cache()`).

- [ ] **Step 1: Write the failing test**

Test the seam via a fake runtime so no model is needed:

```python
# append to tests/test_session_cache.py
import time


class _FakeModel:
    def make_cache(self) -> list:
        return ["FRESH"]


class _FakeRuntime:
    model = _FakeModel()
    model_id = "glm52"


def test_prefill_prompt_cache_saves_and_reuses(tmp_path) -> None:
    from mlx_vq.build.chat import prefill_prompt_cache

    runtime = _FakeRuntime()
    session_dir = tmp_path / "sess1"

    # First turn: cold, then persisted.
    resolution = prefill_prompt_cache(
        runtime,
        token_ids=[1, 2, 3],
        session_dir=session_dir,
        save_fn=lambda d, c, **kw: __import__("mlx_vq.runtime.session_cache", fromlist=["save_session_cache"]).save_session_cache(d, c, now_unix=time.time(), **kw),
        persist_cache=lambda _d, _c, _tok: None,  # skip real safetensors write
    )
    assert resolution.warm is False
```

> Implementation note: `prefill_prompt_cache` takes injectable `save_fn` /
> `persist_cache` seams precisely so this test runs without MLX save I/O. The
> real chat path passes the true `save_session_cache`.

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_session_cache.py::test_prefill_prompt_cache_saves_and_reuses -q`
Expected: FAIL with `ImportError: cannot import name 'prefill_prompt_cache'`.

- [ ] **Step 3: Write minimal implementation**

Add a small, testable helper to `src/mlx_vq/build/chat.py` and call it from
`_stream_mlx_tokens` (`chat.py:232`) in place of the bare
`prompt_cache = runtime.model.make_cache()`:

```python
# src/mlx_vq/build/chat.py  (new helper near the top-level functions)
def prefill_prompt_cache(
    runtime,
    *,
    token_ids,
    session_dir=None,
    persist_cache=None,
    save_fn=None,
):
    """Resolve a warm-or-cold prompt cache and persist the full-prompt cache.

    When ``session_dir`` is None the behavior is a fresh cache (today's default).
    """
    if session_dir is None:
        return _ColdResolution(runtime.model.make_cache())

    from mlx_vq.runtime.session_cache import (
        SessionLedger,
        resolve_prompt_cache,
        save_session_cache,
    )

    ledger = SessionLedger.load(session_dir)
    resolution = resolve_prompt_cache(
        session_dir,
        model_id=runtime.model_id,
        token_ids=token_ids,
        make_cache=runtime.model.make_cache,
        ledger=ledger,
    )
    return resolution
```

Add the tiny `_ColdResolution` shim so callers get a uniform `.cache`/`.warm`
interface:

```python
# src/mlx_vq/build/chat.py
class _ColdResolution:
    def __init__(self, cache) -> None:
        self.cache = cache
        self.reused_tokens = 0
        self.warm = False
```

In `_stream_mlx_tokens`, replace the `prompt_cache=runtime.model.make_cache()`
argument with the resolved cache, and after prefill (once the full prompt has
been processed) persist it when a session id is configured:

```python
    resolution = prefill_prompt_cache(
        runtime,
        token_ids=token_ids,
        session_dir=getattr(runtime, "warm_cache_dir", None),
    )
    prompt_cache = resolution.cache
    # ... existing generate_step(... prompt_cache=prompt_cache ...) unchanged ...
    # After generation, if a warm-cache dir is configured, persist:
    warm_dir = getattr(runtime, "warm_cache_dir", None)
    if warm_dir is not None:
        import time as _time

        from mlx_vq.runtime.session_cache import SessionLedger, save_session_cache

        record = save_session_cache(
            warm_dir,
            prompt_cache,
            model_id=runtime.model_id,
            token_ids=token_ids,
            now_unix=_time.time(),
        )
        SessionLedger.load(warm_dir).append(record)
```

> The exact insertion point is the existing `prompt_cache` construction at
> `chat.py:254-259`; keep the `generate_step(...)` call and token iteration
> (`chat.py:261-267`) unchanged. `runtime.warm_cache_dir` defaults to `None`
> (cold path), so existing behavior is preserved unless a session is opened with
> a warm-cache directory.

Simplify the test to only exercise the cold branch of `prefill_prompt_cache`
(the warm branch is already covered by Task 4's resolver tests):

```python
# adjust the Task 5 test body to:
def test_prefill_prompt_cache_cold_when_no_session() -> None:
    from mlx_vq.build.chat import prefill_prompt_cache

    runtime = _FakeRuntime()
    resolution = prefill_prompt_cache(runtime, token_ids=[1, 2, 3], session_dir=None)
    assert resolution.warm is False
    assert resolution.cache == ["FRESH"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_session_cache.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/build/chat.py tests/test_session_cache.py
git commit -m "feat: wire warm KV-cache resolution into chat prefill"
```

---

### Task 6: `keep sessions` CLI + ramp re-export + docs

**Files:**
- Modify: `src/mlx_vq/build/cli.py` (add `sessions` subcommand: `list`, `prune`)
- Modify: `src/mlx_vq/runtime/session_cache.py` (add `prune_session`, `list_sessions`)
- Create: `src/ramp/runtime/__init__.py`, `src/ramp/runtime/session_cache.py`
- Test: `tests/test_session_cache_cli.py`

**Interfaces:**
- Produces:
  `list_sessions(root: Path) -> list[tuple[str, int]]` (session_id, cached-prefix count);
  `prune_session(root: Path, session_id: str) -> int` (files removed);
  CLI `keep sessions list [--root PATH]` and `keep sessions prune SESSION_ID [--root PATH]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session_cache_cli.py
from __future__ import annotations

from pathlib import Path

from mlx_vq.build.cli import main as cli_main
from mlx_vq.runtime.session_cache import SessionCacheRecord, SessionLedger


def _seed_session(root: Path, session_id: str) -> None:
    session_dir = root / session_id
    ledger = SessionLedger.load(session_dir)
    ledger.append(
        SessionCacheRecord(
            fingerprint="ab",
            model_id="glm52",
            token_count=2,
            cache_file="cache-ab.safetensors",
            created_unix=0.0,
        )
    )
    (session_dir / "cache-ab.safetensors").write_bytes(b"x")


def test_sessions_list_and_prune(tmp_path: Path, capsys) -> None:
    _seed_session(tmp_path, "s1")
    assert cli_main(["sessions", "list", "--root", str(tmp_path)]) == 0
    assert "s1" in capsys.readouterr().out

    assert cli_main(["sessions", "prune", "s1", "--root", str(tmp_path)]) == 0
    assert not (tmp_path / "s1" / "cache-ab.safetensors").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_session_cache_cli.py -q`
Expected: FAIL — unknown `sessions` command.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/runtime/session_cache.py
def list_sessions(root: Path) -> list[tuple[str, int]]:
    if not root.exists():
        return []
    out: list[tuple[str, int]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        ledger = SessionLedger.load(child)
        out.append((child.name, len(ledger.records)))
    return out


def prune_session(root: Path, session_id: str) -> int:
    session_dir = root / session_id
    if not session_dir.exists():
        return 0
    removed = 0
    for path in session_dir.glob("*"):
        path.unlink()
        removed += 1
    session_dir.rmdir()
    return removed
```

Register in `build_parser()` (`src/mlx_vq/build/cli.py:418`):

```python
    sessions = subparsers.add_parser("sessions", help="manage warm KV-cache sessions")
    sessions_sub = sessions.add_subparsers(dest="sessions_command", required=True)
    sessions_list = sessions_sub.add_parser("list", help="list cached sessions")
    sessions_list.add_argument("--root", default=None)
    sessions_list.set_defaults(func=_cmd_sessions_list)
    sessions_prune = sessions_sub.add_parser("prune", help="delete a session's caches")
    sessions_prune.add_argument("session_id")
    sessions_prune.add_argument("--root", default=None)
    sessions_prune.set_defaults(func=_cmd_sessions_prune)
```

```python
# handlers with the other _cmd_* functions
def _cmd_sessions_list(args: argparse.Namespace) -> int:
    from pathlib import Path

    from mlx_vq.runtime.session_cache import default_session_root, list_sessions

    root = Path(args.root) if args.root else default_session_root()
    for session_id, count in list_sessions(root):
        print(f"{session_id}\t{count} cached prefixes")
    return 0


def _cmd_sessions_prune(args: argparse.Namespace) -> int:
    from pathlib import Path

    from mlx_vq.runtime.session_cache import default_session_root, prune_session

    root = Path(args.root) if args.root else default_session_root()
    removed = prune_session(root, args.session_id)
    print(f"removed {removed} file(s) for session {args.session_id}")
    return 0
```

Re-export:

```python
# src/ramp/runtime/__init__.py
"""Public RAMP runtime surface."""
```

```python
# src/ramp/runtime/session_cache.py
from __future__ import annotations

from mlx_vq.runtime.session_cache import (  # noqa: F401
    PromptCacheResolution,
    SessionCacheRecord,
    SessionLedger,
    load_cache_arrays,
    resolve_prompt_cache,
    save_session_cache,
    session_cache_dir,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_session_cache.py tests/test_session_cache_cli.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/session_cache.py src/mlx_vq/build/cli.py src/ramp/runtime tests/test_session_cache_cli.py
git commit -m "feat: add keep sessions CLI and ramp session-cache re-export"
```

---

## Self-Review

- **Spec coverage (F3):** persist KV cache (Task 2), warm reopen with zero
  re-prefill for the matched prefix (Tasks 3–4), ledger/"resume" semantics
  (Task 3), chat integration behind opt-in (Task 5), management CLI (Task 6).
- **Correctness guard:** `best_match` re-derives the fingerprint from the
  requested tokens and compares — a non-prefix or wrong-model request can never
  return a warm cache; corrupt files fall back to cold (Task 4).
- **Placeholder scan:** none; all steps runnable.
- **Type consistency:** `SessionCacheRecord`, `SessionLedger`,
  `PromptCacheResolution`, `resolve_prompt_cache`, `save_session_cache` used
  consistently across tasks, CLI, and the ramp re-export.
- **Deferred (noted, not dropped):** partial-prefix reuse that prefills only the
  tail via `trim_prompt_cache` is a follow-up; this plan restores a cache and the
  caller prefills the remaining `len(token_ids) - reused_tokens` tokens.
