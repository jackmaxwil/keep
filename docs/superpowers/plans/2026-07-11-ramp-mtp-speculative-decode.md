# RAMP MTP Speculative Decoding (F5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use GLM-5.2's built-in multi-token-prediction (MTP) head as a draft
model for speculative decoding — propose k tokens from the MTP head, verify them
in one batched main-model forward, and accept the longest greedy-correct prefix —
raising decode throughput toward colibri's 2.2–2.8 tokens/forward without
changing what greedy decoding would have produced.

**Architecture:** Split into a pure, fully-tested core and a host-only
integration. The core is the greedy acceptance rule and the generation
orchestration, both driven by injected `draft_step`/`verify_step` callables. The
integration re-admits the MTP head (currently excluded at load and *forbidden* by
the artifact validator) behind an explicit opt-in, builds `draft_step` from the
MTP head and `verify_step` from the main model, and manages the draft/main KV
caches. The default acceptance/validation pipeline's MTP forbid-guard is left
intact.

**Tech Stack:** Python 3.11+, MLX, `mlx_lm` cache, pytest.

## Global Constraints

Inherited from [`2026-07-11-ramp-beyond-ram-roadmap.md`](2026-07-11-ramp-beyond-ram-roadmap.md).
Load-bearing here:

- **Exactness:** speculative decoding must yield exactly the greedy-decoding
  token sequence the main model alone would produce. The acceptance rule is the
  standard "accept while main-model argmax matches the draft, else take the
  correction." A test asserts equivalence to plain greedy on fakes.
- **Do not weaken the default MTP guard.** `glm52_artifact.py:554` forbids
  layer-78 MTP artifacts and binding excludes MTP tensors
  (`glm52_vq_adapter.py:840-845`). F5 adds an *opt-in* admission path only; the
  default pipeline still excludes/forbids MTP. A test proves the default stays
  excluded.
- **Preserve the NAX path** for the main model's routed experts; the MTP head is
  a small extra head, not a change to routed-expert dispatch.
- Core tests run with **no Metal and no model** (injected `draft_step`/
  `verify_step`). MTP head loading + real decode are host-only (`# pragma: no
  cover`), gated on the model + `.keep-heavy-job.lock`.
- `from __future__ import annotations`; pytest; `tmp_path`. New code under
  `src/mlx_vq/runtime/`; `ramp` re-export as the final task. One commit/task.

## File Structure

- Create `src/mlx_vq/runtime/speculative.py` — `accept_draft_tokens`,
  `speculative_generate`, `SpeculationStats`.
- Create `src/mlx_vq/models/glm52_mtp.py` — opt-in MTP head admission +
  `MTPDraftHead` + `build_draft_verify_steps`.
- Modify `src/mlx_vq/models/glm52_vq_adapter.py` — thread `allow_mtp` through
  non-VQ binding.
- Modify `src/mlx_vq/build/chat.py` — `--speculative`/`draft_len` opt-in.
- Create `tests/test_speculative.py`, `tests/test_glm52_mtp_admission.py`.
- Create `benchmarks/bench_glm52_speculative.py` (labeled acceptance benchmark).
- Create `src/ramp/runtime/speculative.py` — re-export (final task).

## Verified facts (2026-07-11)

- MTP present in source, excluded at load: `_is_mtp_tensor(name, args)`
  (`glm52_vq_adapter.py:287`) flags `layer_idx >= num_hidden_layers`; binding
  appends to `skipped_mtp` (`:840-845`); `NON_VQ_EXPECTED_MTP_TENSORS = 2039`
  (`convert/glm52_non_vq.py:58`); validator forbids layer 78
  (`validate/glm52_artifact.py:554`).
- Generation: `_stream_mlx_tokens(runtime, messages)` (`build/chat.py:232`) uses
  `mlx_lm.generate.generate_step(prompt_tokens, model, prompt_cache=..., max_tokens=...)`.
- Cache: `make_cache()` returns `list[CacheList]` (GLM52,
  `glm52_vq_adapter.py:567`). Model forward `model(inputs, cache=...)` returns
  logits; decode step feeds `[[next_token]]` (`benchmark/glm45_air.py:513`).

---

### Task 1: Greedy acceptance rule (pure core)

**Files:**
- Create: `src/mlx_vq/runtime/speculative.py`
- Test: `tests/test_speculative.py`

**Interfaces:**
- Produces:
  `AcceptResult(new_tokens: list[int], accepted_draft: int, rejected_at: int | None)`;
  `accept_draft_tokens(draft: list[int], verify_argmax: list[int]) -> AcceptResult`
  where `len(verify_argmax) == len(draft) + 1` (one bonus slot). Accept draft
  tokens while `verify_argmax[i] == draft[i]`; at the first mismatch `j` emit the
  correction `verify_argmax[j]` and stop; if all match, emit the bonus
  `verify_argmax[len(draft)]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_speculative.py
from __future__ import annotations

import pytest

from mlx_vq.runtime.speculative import AcceptResult, accept_draft_tokens


def test_all_draft_accepted_emits_bonus() -> None:
    result = accept_draft_tokens([5, 6, 7], [5, 6, 7, 8])
    assert result == AcceptResult(new_tokens=[5, 6, 7, 8], accepted_draft=3, rejected_at=None)


def test_mismatch_midway_takes_correction() -> None:
    result = accept_draft_tokens([5, 6, 7], [5, 6, 9, 8])
    assert result.new_tokens == [5, 6, 9]  # 9 is the correction at index 2
    assert result.accepted_draft == 2
    assert result.rejected_at == 2


def test_first_token_mismatch_emits_single_correction() -> None:
    result = accept_draft_tokens([5], [3, 4])
    assert result.new_tokens == [3]
    assert result.accepted_draft == 0
    assert result.rejected_at == 0


def test_empty_draft_emits_single_bonus() -> None:
    result = accept_draft_tokens([], [42])
    assert result.new_tokens == [42]
    assert result.accepted_draft == 0
    assert result.rejected_at is None


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="verify_argmax"):
        accept_draft_tokens([1, 2], [1, 2])  # needs len 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_speculative.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_vq.runtime.speculative'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mlx_vq/runtime/speculative.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcceptResult:
    new_tokens: list[int]
    accepted_draft: int
    rejected_at: int | None


def accept_draft_tokens(draft: list[int], verify_argmax: list[int]) -> AcceptResult:
    if len(verify_argmax) != len(draft) + 1:
        raise ValueError("verify_argmax must have length len(draft) + 1")
    for i, proposed in enumerate(draft):
        if verify_argmax[i] != proposed:
            return AcceptResult(
                new_tokens=list(draft[:i]) + [verify_argmax[i]],
                accepted_draft=i,
                rejected_at=i,
            )
    return AcceptResult(
        new_tokens=list(draft) + [verify_argmax[len(draft)]],
        accepted_draft=len(draft),
        rejected_at=None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_speculative.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/speculative.py tests/test_speculative.py
git commit -m "feat: add greedy speculative-decode acceptance rule"
```

---

### Task 2: `speculative_generate` orchestration + stats

**Files:**
- Modify: `src/mlx_vq/runtime/speculative.py`
- Test: `tests/test_speculative.py`

**Interfaces:**
- Consumes: `accept_draft_tokens`.
- Produces:
  `SpeculationStats(verify_steps: int, draft_proposed: int, draft_accepted: int, tokens_emitted: int)` with `.tokens_per_forward` and `.acceptance_rate` properties;
  `speculative_generate(prompt_tokens, *, draft_step, verify_step, max_new_tokens, eos_id=None) -> tuple[list[int], SpeculationStats]`.
  `draft_step(context: list[int]) -> list[int]` proposes up to k tokens;
  `verify_step(context: list[int], draft: list[int]) -> list[int]` returns the
  main model's argmax for each draft position plus one bonus (length
  `len(draft)+1`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_speculative.py
from mlx_vq.runtime.speculative import SpeculationStats, speculative_generate


def test_speculative_matches_pure_greedy_and_counts_stats() -> None:
    # Oracle greedy target the main model would produce after the prompt.
    greedy = [10, 11, 12, 13, 14]

    def verify_step(context: list[int], draft: list[int]) -> list[int]:
        # Main model argmax at each position: the next greedy token given how
        # many correct tokens are already in context.
        start = len(context) - 3  # prompt length is 3 in the test
        return [greedy[start + i] for i in range(len(draft) + 1)]

    def draft_step(context: list[int]) -> list[int]:
        # A draft that is right for 2 tokens then wrong, to exercise rejection.
        start = len(context) - 3
        proposal = []
        for i in range(2):
            idx = start + i
            proposal.append(greedy[idx] if idx < len(greedy) else 0)
        proposal[-1] = 999 if len(proposal) == 2 else proposal[-1]  # force a miss
        return proposal

    out, stats = speculative_generate(
        [1, 2, 3],
        draft_step=draft_step,
        verify_step=verify_step,
        max_new_tokens=5,
    )
    assert out == greedy  # exactness vs greedy
    assert isinstance(stats, SpeculationStats)
    assert stats.tokens_emitted == 5
    assert stats.tokens_per_forward >= 1.0


def test_speculative_stops_at_eos() -> None:
    def verify_step(context, draft):
        return [7] * (len(draft) + 1)

    def draft_step(context):
        return []

    out, stats = speculative_generate(
        [1], draft_step=draft_step, verify_step=verify_step, max_new_tokens=10, eos_id=7
    )
    assert out == [7]
    assert stats.tokens_emitted == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_speculative.py -k speculative -q`
Expected: FAIL with `ImportError: cannot import name 'speculative_generate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/runtime/speculative.py
from typing import Callable


@dataclass
class SpeculationStats:
    verify_steps: int = 0
    draft_proposed: int = 0
    draft_accepted: int = 0
    tokens_emitted: int = 0

    @property
    def tokens_per_forward(self) -> float:
        return self.tokens_emitted / self.verify_steps if self.verify_steps else 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.draft_accepted / self.draft_proposed if self.draft_proposed else 0.0


def speculative_generate(
    prompt_tokens,
    *,
    draft_step: Callable[[list[int]], list[int]],
    verify_step: Callable[[list[int], list[int]], list[int]],
    max_new_tokens: int,
    eos_id: int | None = None,
) -> tuple[list[int], SpeculationStats]:
    context = list(prompt_tokens)
    generated: list[int] = []
    stats = SpeculationStats()
    while len(generated) < max_new_tokens:
        draft = list(draft_step(context))
        verify_argmax = verify_step(context, draft)
        result = accept_draft_tokens(draft, verify_argmax)
        stats.verify_steps += 1
        stats.draft_proposed += len(draft)
        stats.draft_accepted += result.accepted_draft
        for token in result.new_tokens:
            generated.append(token)
            context.append(token)
            stats.tokens_emitted += 1
            if eos_id is not None and token == eos_id:
                return generated, stats
            if len(generated) >= max_new_tokens:
                return generated[:max_new_tokens], stats
    return generated[:max_new_tokens], stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_speculative.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/runtime/speculative.py tests/test_speculative.py
git commit -m "feat: add speculative generation loop with acceptance stats"
```

---

### Task 3: Opt-in MTP admission gate (keep the default forbid intact)

**Files:**
- Create: `src/mlx_vq/models/glm52_mtp.py`
- Test: `tests/test_glm52_mtp_admission.py`

**Interfaces:**
- Produces:
  `should_admit_mtp_tensor(name: str, num_hidden_layers: int, *, allow_mtp: bool) -> bool`
  — returns `True` only for MTP tensors AND only when `allow_mtp` is set;
  `partition_mtp_tensors(names: Iterable[str], num_hidden_layers: int, *, allow_mtp: bool) -> tuple[list[str], list[str]]`
  returning `(admitted_mtp, excluded_mtp)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_glm52_mtp_admission.py
from __future__ import annotations

from mlx_vq.models.glm52_mtp import (
    partition_mtp_tensors,
    should_admit_mtp_tensor,
)

# GLM-5.2 has 78 hidden layers; MTP tensors live at layer index 78+.
NUM_HIDDEN = 78


def test_default_never_admits_mtp() -> None:
    name = "model.layers.78.embed_tokens.weight"
    assert should_admit_mtp_tensor(name, NUM_HIDDEN, allow_mtp=False) is False


def test_opt_in_admits_only_mtp_tensors() -> None:
    mtp = "model.layers.78.mtp.norm.weight"
    main = "model.layers.40.self_attn.q_proj.weight"
    assert should_admit_mtp_tensor(mtp, NUM_HIDDEN, allow_mtp=True) is True
    assert should_admit_mtp_tensor(main, NUM_HIDDEN, allow_mtp=True) is False


def test_partition_splits_admitted_from_excluded() -> None:
    names = [
        "model.layers.40.mlp.gate.weight",   # main → neither list
        "model.layers.78.mtp.a.weight",      # mtp
        "model.layers.79.mtp.b.weight",      # mtp
    ]
    admitted, excluded = partition_mtp_tensors(names, NUM_HIDDEN, allow_mtp=True)
    assert admitted == ["model.layers.78.mtp.a.weight", "model.layers.79.mtp.b.weight"]
    assert excluded == []

    admitted2, excluded2 = partition_mtp_tensors(names, NUM_HIDDEN, allow_mtp=False)
    assert admitted2 == []
    assert excluded2 == ["model.layers.78.mtp.a.weight", "model.layers.79.mtp.b.weight"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_glm52_mtp_admission.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_vq.models.glm52_mtp'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mlx_vq/models/glm52_mtp.py
from __future__ import annotations

import re
from typing import Iterable

_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def _layer_index(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _is_mtp_tensor(name: str, num_hidden_layers: int) -> bool:
    layer_idx = _layer_index(name)
    return layer_idx is not None and layer_idx >= num_hidden_layers


def should_admit_mtp_tensor(name: str, num_hidden_layers: int, *, allow_mtp: bool) -> bool:
    return allow_mtp and _is_mtp_tensor(name, num_hidden_layers)


def partition_mtp_tensors(
    names: Iterable[str],
    num_hidden_layers: int,
    *,
    allow_mtp: bool,
) -> tuple[list[str], list[str]]:
    admitted: list[str] = []
    excluded: list[str] = []
    for name in names:
        if not _is_mtp_tensor(name, num_hidden_layers):
            continue
        if allow_mtp:
            admitted.append(name)
        else:
            excluded.append(name)
    return admitted, excluded
```

> The admission gate mirrors `_is_mtp_tensor` from `glm52_vq_adapter.py:287` so
> the definition of "MTP tensor" stays identical. Task 5 routes non-VQ binding
> through `partition_mtp_tensors`; when `allow_mtp=False` (the default and the
> only path the acceptance/validation pipeline uses) MTP stays excluded exactly
> as today, so `glm52_artifact.py:554`'s forbid-guard is never contradicted.

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_glm52_mtp_admission.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/models/glm52_mtp.py tests/test_glm52_mtp_admission.py
git commit -m "feat: add opt-in MTP tensor admission gate"
```

---

### Task 4: KV-cache offset sync helper (pure)

**Files:**
- Modify: `src/mlx_vq/models/glm52_mtp.py`
- Test: `tests/test_glm52_mtp_admission.py`

**Interfaces:**
- Produces:
  `tokens_to_rollback(main_offset: int, draft_len: int, accepted: int) -> int`
  — how many positions the main cache must trim after a verify step that
  proposed `draft_len` tokens and accepted `accepted` of them (main model saw
  `draft_len+1` positions; keeps `accepted+1`). Used to keep the main KV cache
  consistent when some drafted tokens are rejected.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_glm52_mtp_admission.py
from mlx_vq.models.glm52_mtp import tokens_to_rollback


def test_rollback_when_all_accepted_is_zero() -> None:
    # Proposed 3, accepted 3 → main advanced 4 (3 draft + bonus), keep all 4.
    assert tokens_to_rollback(main_offset=100, draft_len=3, accepted=3) == 0


def test_rollback_discards_rejected_tail() -> None:
    # Proposed 3, accepted 1 → main saw 4 positions, keep accepted+1=2, trim 2.
    assert tokens_to_rollback(main_offset=100, draft_len=3, accepted=1) == 2


def test_rollback_never_negative() -> None:
    assert tokens_to_rollback(main_offset=0, draft_len=0, accepted=0) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_glm52_mtp_admission.py -k rollback -q`
Expected: FAIL with `ImportError: cannot import name 'tokens_to_rollback'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/models/glm52_mtp.py
def tokens_to_rollback(main_offset: int, draft_len: int, accepted: int) -> int:
    # Main model consumed draft_len + 1 positions during verification; a correct
    # run keeps accepted + 1 (accepted draft tokens plus the emitted correction/
    # bonus). Everything past that must be trimmed from the KV cache.
    consumed = draft_len + 1
    kept = accepted + 1
    return max(0, consumed - kept)
```

> The real decode loop applies this with
> `mlx_lm.models.cache.trim_prompt_cache(main_cache, tokens_to_rollback(...))`
> after each verify step (see Task 5). `main_offset` is accepted for symmetry
> with the caller and future assertions; the rollback amount does not depend on
> it under greedy verification.

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_glm52_mtp_admission.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/models/glm52_mtp.py tests/test_glm52_mtp_admission.py
git commit -m "feat: add KV-cache rollback math for speculative verify"
```

---

### Task 5: MTP head loading + draft/verify adapters (host-only integration)

**Files:**
- Modify: `src/mlx_vq/models/glm52_mtp.py` (`MTPDraftHead`,
  `build_draft_verify_steps`)
- Modify: `src/mlx_vq/models/glm52_vq_adapter.py` (thread `allow_mtp` through the
  non-VQ binder so MTP tensors are admitted when requested)
- Test: `tests/test_glm52_mtp_admission.py` (adapter shape with fakes)

**Interfaces:**
- Consumes: `partition_mtp_tensors`, `tokens_to_rollback`, `accept_draft_tokens`.
- Produces:
  `build_draft_verify_steps(main_model, mtp_head, *, draft_len, argmax_fn) -> tuple[draft_step, verify_step]`
  where `argmax_fn(logits) -> int` isolates the array backend so the adapter is
  unit-testable with fake models. `draft_step(context)` runs the MTP head
  autoregressively for `draft_len` tokens; `verify_step(context, draft)` runs the
  main model over `context[-1:] + draft` and returns argmax per position.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_glm52_mtp_admission.py
from mlx_vq.models.glm52_mtp import build_draft_verify_steps


class _FakeMain:
    """Predicts token = last_context_token + 1 at every position (deterministic)."""

    def next_argmax(self, context: list[int], k: int) -> list[int]:
        base = context[-1]
        return [base + 1 + i for i in range(k)]


class _FakeMTP:
    """Draft head that guesses the same rule but drifts after 1 token."""

    def propose(self, context: list[int], draft_len: int) -> list[int]:
        base = context[-1]
        out = [base + 1]
        out += [base + 100 + i for i in range(1, draft_len)]  # wrong tail
        return out[:draft_len]


def test_build_draft_verify_steps_shapes(monkeypatch) -> None:
    main, mtp = _FakeMain(), _FakeMTP()
    draft_step, verify_step = build_draft_verify_steps(
        main, mtp, draft_len=3, argmax_fn=lambda logits: int(logits),
        _fake_draft=lambda ctx, k: mtp.propose(ctx, k),
        _fake_verify=lambda ctx, draft: main.next_argmax(ctx, len(draft) + 1),
    )
    context = [1, 2, 3]
    draft = draft_step(context)
    assert len(draft) == 3
    verify = verify_step(context, draft)
    assert len(verify) == len(draft) + 1
```

> The `_fake_draft`/`_fake_verify` seams exist so this test needs no MLX model.
> In production those parameters are `None` and the real MLX forward paths run.

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_glm52_mtp_admission.py -k draft_verify -q`
Expected: FAIL with `ImportError: cannot import name 'build_draft_verify_steps'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/models/glm52_mtp.py
from typing import Callable


class MTPDraftHead:
    """Wraps the admitted GLM-5.2 MTP layer as a draft-token proposer.

    Real construction (host-only) binds the layer-78 tensors admitted by
    partition_mtp_tensors and shares the main model's embedding + hidden state.
    """

    def __init__(self, module, argmax_fn: Callable[[object], int]) -> None:
        self._module = module
        self._argmax_fn = argmax_fn

    def propose(self, hidden_state, draft_len: int) -> list[int]:  # pragma: no cover
        tokens: list[int] = []
        state = hidden_state
        for _ in range(draft_len):
            logits, state = self._module.step(state)
            tokens.append(self._argmax_fn(logits))
        return tokens


def build_draft_verify_steps(
    main_model,
    mtp_head,
    *,
    draft_len: int,
    argmax_fn: Callable[[object], int],
    _fake_draft: Callable[[list[int], int], list[int]] | None = None,
    _fake_verify: Callable[[list[int], list[int]], list[int]] | None = None,
):
    def draft_step(context: list[int]) -> list[int]:
        if _fake_draft is not None:
            return _fake_draft(context, draft_len)
        # Host path: encode context tail, run the MTP head autoregressively.
        return mtp_head.propose(main_model.hidden_for(context), draft_len)  # pragma: no cover

    def verify_step(context: list[int], draft: list[int]) -> list[int]:
        if _fake_verify is not None:
            return _fake_verify(context, draft)
        # Host path: one batched main-model forward over context tail + draft,
        # returning argmax per position (len(draft)+1). pragma: no cover
        logits = main_model.verify_logits(context, draft)  # pragma: no cover
        return [argmax_fn(row) for row in logits]  # pragma: no cover

    return draft_step, verify_step
```

Thread `allow_mtp` through non-VQ binding in
`src/mlx_vq/models/glm52_vq_adapter.py` (the loop that currently skips MTP at
`:840-845`). Replace the unconditional skip with the gate:

```python
# src/mlx_vq/models/glm52_vq_adapter.py  (inside the non-VQ bind loop, ~:840)
from mlx_vq.models.glm52_mtp import should_admit_mtp_tensor

# ... where the code currently does: if _is_mtp_tensor(name, args): skipped_mtp.append(name); continue
if _is_mtp_tensor(name, args):
    if not should_admit_mtp_tensor(name, args.num_hidden_layers, allow_mtp=allow_mtp):
        skipped_mtp.append(name)
        continue
    admitted_mtp.append(name)
    # fall through to bind the MTP tensor
```

Add `allow_mtp: bool = False` to the non-VQ binder signature and initialize
`admitted_mtp: list[str] = []`, returning it on the bind report. Default
`allow_mtp=False` reproduces today's behavior exactly.

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_glm52_mtp_admission.py -q`
Expected: PASS (all). Real MTP loading is verified host-only (Task 7).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/models/glm52_mtp.py src/mlx_vq/models/glm52_vq_adapter.py tests/test_glm52_mtp_admission.py
git commit -m "feat: add MTP draft head adapters and opt-in admission wiring"
```

---

### Task 6: Chat opt-in + ramp re-export + docs

**Files:**
- Modify: `src/mlx_vq/build/chat.py` (opt-in `speculative`/`draft_len` on runtime)
- Create: `src/ramp/runtime/speculative.py`
- Modify: `README.md` (note the speculative decode path)
- Test: `tests/test_speculative.py` (runtime selection with fakes)

**Interfaces:**
- Consumes: `speculative_generate`, `build_draft_verify_steps`.
- Produces: `generate_tokens(runtime, prompt_tokens, *, max_new_tokens)` that
  dispatches to `speculative_generate` when `runtime.speculative` is set and a
  `runtime.mtp_head` exists, else the existing single-token path. Default
  behavior unchanged.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_speculative.py
def test_generate_tokens_uses_speculative_when_enabled() -> None:
    from mlx_vq.build.chat import generate_tokens

    class _Runtime:
        speculative = True
        draft_len = 2

        def draft_verify(self):
            # returns (draft_step, verify_step); greedy target [9,9,9]
            def verify_step(ctx, draft):
                return [9] * (len(draft) + 1)

            def draft_step(ctx):
                return [9, 0]  # first right, second wrong

            return draft_step, verify_step

    out = generate_tokens(_Runtime(), [1], max_new_tokens=3)
    assert out == [9, 9, 9]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_speculative.py::test_generate_tokens_uses_speculative_when_enabled -q`
Expected: FAIL with `ImportError: cannot import name 'generate_tokens'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mlx_vq/build/chat.py  (new dispatcher helper)
def generate_tokens(runtime, prompt_tokens, *, max_new_tokens):
    if getattr(runtime, "speculative", False) and hasattr(runtime, "draft_verify"):
        from mlx_vq.runtime.speculative import speculative_generate

        draft_step, verify_step = runtime.draft_verify()
        tokens, _stats = speculative_generate(
            prompt_tokens,
            draft_step=draft_step,
            verify_step=verify_step,
            max_new_tokens=max_new_tokens,
            eos_id=getattr(runtime, "eos_id", None),
        )
        return tokens
    # Fall back to the existing single-token generate path.
    return _generate_tokens_single(runtime, prompt_tokens, max_new_tokens=max_new_tokens)
```

> `_generate_tokens_single` is the extraction of today's
> `generate_step(...)`-driven loop (`chat.py:254-267`) into a helper returning
> the token list. Extract it verbatim; the streaming iterator behavior is
> unchanged when `speculative` is off.

```python
# src/ramp/runtime/speculative.py
from __future__ import annotations

from mlx_vq.runtime.speculative import (  # noqa: F401
    AcceptResult,
    SpeculationStats,
    accept_draft_tokens,
    speculative_generate,
)
```

Add to `README.md`:

```markdown
- **Speculative decoding (optional):** with `--speculative`, RAMP uses the
  GLM-5.2 MTP head as a draft model, verifying k drafted tokens per main-model
  forward. Output is identical to greedy decoding; throughput approaches
  2–3 tokens/forward. The MTP head is admitted only on this opt-in path; the
  default acceptance/validation pipeline still excludes it.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_speculative.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/build/chat.py src/ramp/runtime/speculative.py README.md tests/test_speculative.py
git commit -m "feat: dispatch chat generation through optional MTP speculation"
```

---

### Task 7: Acceptance benchmark (labeled, host-only)

**Files:**
- Create: `benchmarks/bench_glm52_speculative.py`
- Test: `tests/test_speculative.py` (report shape)

**Interfaces:**
- Produces: `summarize_speculative_run(stats, elapsed_seconds) -> dict` with
  `tokens_per_forward`, `acceptance_rate`, `tokens_per_second`, `mode:
  "speculative"`; a host-only `main()` that loads the model with
  `allow_mtp=True`, runs `speculative_generate`, and reports.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_speculative.py
def test_speculative_bench_report_shape() -> None:
    from benchmarks.bench_glm52_speculative import summarize_speculative_run
    from mlx_vq.runtime.speculative import SpeculationStats

    stats = SpeculationStats(verify_steps=4, draft_proposed=8, draft_accepted=5, tokens_emitted=9)
    row = summarize_speculative_run(stats, elapsed_seconds=3.0)
    assert row["mode"] == "speculative"
    assert row["tokens_per_forward"] == 9 / 4
    assert row["acceptance_rate"] == 5 / 8
    assert row["tokens_per_second"] == 9 / 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_speculative.py::test_speculative_bench_report_shape -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks.bench_glm52_speculative'`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/bench_glm52_speculative.py
from __future__ import annotations

import argparse
import json


def summarize_speculative_run(stats, elapsed_seconds: float) -> dict:
    return {
        "mode": "speculative",
        "verify_steps": stats.verify_steps,
        "tokens_emitted": stats.tokens_emitted,
        "tokens_per_forward": stats.tokens_per_forward,
        "acceptance_rate": stats.acceptance_rate,
        "tokens_per_second": stats.tokens_emitted / elapsed_seconds if elapsed_seconds else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench_glm52_speculative")
    parser.add_argument("artifact_dir")
    parser.add_argument("--draft-len", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Explain mixture-of-experts briefly.")
    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - heavy path
    args = build_parser().parse_args(argv)
    # Heavy/host-only: acquire .keep-heavy-job.lock; load with allow_mtp=True;
    # build_draft_verify_steps(...); speculative_generate(...); time it; print.
    print(json.dumps({"note": "run on a Metal host with the MTP-admitted model"}))
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_speculative.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/bench_glm52_speculative.py tests/test_speculative.py
git commit -m "feat: add labeled speculative-decode acceptance benchmark"
```

---

## Self-Review

- **Spec coverage (F5):** acceptance rule (Task 1), generation loop + stats
  (Task 2), opt-in MTP admission that preserves the default forbid (Task 3),
  KV rollback math (Task 4), head/adapters + binder wiring (Task 5), chat opt-in
  (Task 6), acceptance benchmark (Task 7).
- **Exactness:** Task 2 asserts speculative output equals the greedy target on
  fakes; the acceptance rule (Task 1) is the standard correction rule.
- **Guard preserved:** Task 3 default `allow_mtp=False` keeps MTP excluded, so
  `glm52_artifact.py:554` is never contradicted; a test proves it.
- **Placeholder scan:** none in the tested core. Host-only real forward paths are
  isolated behind `# pragma: no cover` and `_fake_*` seams, with the exact
  integration points named (`chat.py:254-267`, binder `:840-845`,
  `trim_prompt_cache`).
- **Type consistency:** `AcceptResult`, `SpeculationStats`,
  `speculative_generate`, `build_draft_verify_steps`, `partition_mtp_tensors`,
  `tokens_to_rollback` used consistently across tasks and re-exports.
- **Research risk (flagged):** the exact GLM-5.2 MTP tensor names/step API and
  whether the head shares the main embedding are confirmed only host-side; Task 5
  isolates that behind `MTPDraftHead.step`/`main_model.verify_logits`, so a name
  change is a localized edit, not a plan rewrite.
