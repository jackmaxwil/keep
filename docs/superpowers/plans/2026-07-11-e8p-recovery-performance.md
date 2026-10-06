# E8P Recovery Performance Wave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the brute-force 65,536-row CPU/NumPy diagonal-Hessian E8P
codebook search in the recovery materializer with a byte-exact 256-row factored
search on MLX/Metal, and add expert-level resumable checkpoints with honest
progress/ETA — taking the worst8 (and future EBSS/rotation) re-materialization
from multi-day to a few hours.

**Architecture:** Three layers. (1) A NumPy *reference* factored decoder
`encode_e8p_rtn_diagonal_hessian` that generalizes the existing unweighted
`encode_e8p_rtn` by weighting each dimension — the ground-truth for parity. (2)
An MLX/Metal implementation with the same math, batched across codewords, that is
proven byte-identical to the reference. (3) A dispatcher that routes the
`code_bits=16` branch of `quantize_weight_importance_aware` to the fast path,
plus transactional per-expert checkpointing so a stopped run resumes and reports
truthful progress.

**Tech Stack:** Python 3.11+, MLX/Metal, NumPy, pytest.

## Why this plan (spike evidence, 2026-07-11)

A spike (`scratchpad/e8p_spike.py`) proved the approach on this host:

- **Byte-exact:** factored NumPy decoder vs exhaustive `e8p_full_grid()` search =
  **0/15,000 mismatched codes** over 5 trials (random vectors + random diagonals).
- **Throughput:** exhaustive NumPy 22.9k cw/s → factored NumPy 138k cw/s (6×) →
  **factored MLX/Metal 1.04M cw/s (45×)**.
- **Worst8 (24 groups × 4 assign passes):** ~308 h exhaustive → **~6.8 h MLX**,
  with clear headroom (the timing prototype wastes an `[N,256,8]` intermediate).
- The exhaustive rate reproduces the real stopped run (~1 assign pass of 1 group
  in 3h15m), so the extrapolation is trustworthy.

Conclusion: the algorithmic factorization is real but the **decisive lever is
Metal**; NumPy-only is still ~51 h. This plan standardizes the hot path on MLX.

## Global Constraints

- **Byte-exact, deterministic.** Optimized output must equal the current
  exhaustive `nearest_codebook_indices_diagonal_hessian(..., codebook=e8p_full_grid())`
  bit-for-bit, including deterministic tie-breaking (lowest full-grid index wins).
  Preserve diagonal-Hessian semantics and uint16 code identity.
- **MLX/Metal standardization.** New hot-path code runs on MLX/Metal; keep a
  NumPy reference for parity only. Leave `GLM_MLX_WIRED_LIMIT_GB` unset.
- **Resumable + honest.** A stopped run must retain completed experts via
  transactional checkpoints and never publish a false-complete group. Progress is
  reported at expert granularity with a data-derived ETA.
- **Heavy-run discipline.** The real 24-group run holds `.keep-heavy-job.lock`;
  `runs/`, root handoffs, `artifacts/` (gitignored), and unrelated RAMP plans are
  protected. Selection data only guides tuning.
- **Acceptance gate before rerun.** Do not append a new worst8 campaign
  transition until a measured end-to-end projection is in hours, not days, and
  monitoring exposes completed experts + ETA.
- Tests: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest`.
  `from __future__ import annotations`; pytest; `np.testing.assert_array_equal`
  for byte-identity. Metal-dependent tests skip cleanly when Metal is absent.
- One commit per task. Do not push/merge/publish.

## File Structure

- Modify `src/mlx_vq/codebook/e8.py` — add `encode_e8p_rtn_diagonal_hessian`
  (NumPy reference, factored).
- Create `src/mlx_vq/quant/e8p_metal.py` — MLX/Metal factored search.
- Modify `src/mlx_vq/quant/rtn.py` — `nearest_e8p_codes_diagonal_hessian`
  dispatcher (`backend="metal"|"numpy"`), byte-exact with the exhaustive path.
- Modify `src/mlx_vq/convert/glm52_recovery_materialize.py` — route the
  `code_bits=16` branch to the dispatcher; add expert-level checkpoint/progress.
- Create `benchmarks/bench_e8p_search.py` — microbenchmark + acceptance gate.
- Create `tests/test_e8p_diagonal_hessian.py`, `tests/test_e8p_metal.py`,
  `tests/test_e8p_recovery_checkpoint.py`, `tests/test_e8p_search_bench.py`.

## Verified facts (2026-07-11)

- Exhaustive path: `nearest_codebook_indices_diagonal_hessian(vectors, diagonal, *, codebook, index_dtype, vector_chunk_size, codebook_chunk_size=8192)` (`src/mlx_vq/quant/rtn.py:91`) computes `Σᵢ wᵢ(vᵢ−tᵢ)²` over every table row, `np.argmin`.
- Unweighted factored encoder: `encode_e8p_rtn(vectors, *, chunk_size=8192)` (`src/mlx_vq/codebook/e8.py:289`) — 256 abs rows × even-parity signs × ±0.25 shift; already byte-exact vs the full grid (`tests/test_e8_reference.py:99`).
- Grid identity: `e8p_full_grid()[code] == decode_e8p(code)` (`tests/test_e8_reference.py`), so a full-grid argmin index **is** the uint16 code — the factored encoder's `(abs_idx<<8)|signs` is directly comparable.
- Helpers: `E8P_SHUFFLE_MAP = [0,4,1,5,2,6,3,7]`, `_sign_parity`, `e8p_abs_grid()` (signed [256,8]), `CODEWORD_DIM=8` (`src/mlx_vq/codebook/e8.py:21-22`).
- Call site: `quantize_weight_importance_aware.assign_codes` (`src/mlx_vq/convert/glm52_recovery_materialize.py:680-704`) — `code_bits==16` calls the exhaustive routine with `codebook=e8p_full_grid()`; the `code_bits==8` branch is already a fast inline 256-row search. Called 4× (3 iterations + final) per group.

---

### Task 1: NumPy reference — `encode_e8p_rtn_diagonal_hessian`

**Files:**
- Modify: `src/mlx_vq/codebook/e8.py`
- Test: `tests/test_e8p_diagonal_hessian.py`

**Interfaces:**
- Produces: `encode_e8p_rtn_diagonal_hessian(vectors: np.ndarray, diagonal: np.ndarray, *, chunk_size: int = 65536) -> np.ndarray` returning uint16 codes with shape `vectors.shape[:-1]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e8p_diagonal_hessian.py
from __future__ import annotations

import numpy as np

from mlx_vq.codebook.e8 import (
    e8p_full_grid,
    encode_e8p_rtn,
    encode_e8p_rtn_diagonal_hessian,
)
from mlx_vq.quant.rtn import nearest_codebook_indices_diagonal_hessian


def _exhaustive(vectors, diagonal):
    return nearest_codebook_indices_diagonal_hessian(
        vectors, diagonal, codebook=e8p_full_grid().astype(np.float32),
        index_dtype=np.dtype(np.uint16), vector_chunk_size=4096, codebook_chunk_size=8192,
    )


def test_factored_matches_exhaustive_bit_for_bit() -> None:
    rng = np.random.default_rng(20260711)
    for _ in range(5):
        vecs = (rng.standard_normal((3000, 8)) * 0.6).astype(np.float32)
        diag = np.abs(rng.standard_normal(8)).astype(np.float32) + 0.05
        exact = _exhaustive(vecs, diag)
        fast = encode_e8p_rtn_diagonal_hessian(vecs, diag)
        np.testing.assert_array_equal(fast, exact)


def test_unit_diagonal_reduces_to_unweighted_encoder() -> None:
    rng = np.random.default_rng(7)
    vecs = (rng.standard_normal((500, 8)) * 0.6).astype(np.float32)
    ones = np.ones(8, dtype=np.float32)
    np.testing.assert_array_equal(
        encode_e8p_rtn_diagonal_hessian(vecs, ones),
        encode_e8p_rtn(vecs),
    )


def test_rejects_bad_diagonal() -> None:
    import pytest

    with pytest.raises(ValueError):
        encode_e8p_rtn_diagonal_hessian(np.zeros((1, 8), np.float32), np.zeros(8, np.float32))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_e8p_diagonal_hessian.py -q`
Expected: FAIL with `ImportError: cannot import name 'encode_e8p_rtn_diagonal_hessian'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/mlx_vq/codebook/e8.py` (weighted generalization of `encode_e8p_rtn`;
validated byte-exact in the spike):

```python
def encode_e8p_rtn_diagonal_hessian(
    vectors: np.ndarray, diagonal: np.ndarray, *, chunk_size: int = 65536
) -> np.ndarray:
    """Weighted 256-row factored E8P search under a shared diagonal metric.

    Byte-identical to an exhaustive weighted scan of e8p_full_grid(): each wᵢ≥0
    keeps the per-dim optimal sign = sign(targetᵢ), so the weighted distance is
    Σᵢ wᵢ(|targetᵢ| − abs_rowᵢ)², and the even-parity fix flips the dim of least
    weighted penalty 4·wᵢ·|targetᵢ|·abs_rowᵢ.
    """
    values = np.asarray(vectors, dtype=np.float32)
    if values.shape[-1] != CODEWORD_DIM:
        raise ValueError(f"vectors must have trailing dimension {CODEWORD_DIM}")
    w = np.asarray(diagonal, dtype=np.float32)
    if w.shape != (CODEWORD_DIM,) or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("diagonal must be finite, non-negative, shape (8,), and nonzero")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    flat = values.reshape(-1, CODEWORD_DIM)
    base_rows = e8p_abs_grid().astype(np.float32)
    abs_rows = np.abs(base_rows)
    abs_norm_w = (abs_rows * abs_rows) @ w
    packed = E8P_SHUFFLE_MAP.astype(np.uint16)
    base_signs = np.zeros(abs_rows.shape[0], dtype=np.uint16)
    base_negative = base_rows < 0
    for out_dim, packed_dim in enumerate(packed):
        base_signs |= base_negative[:, out_dim].astype(np.uint16) << packed_dim

    best_codes = np.zeros(flat.shape[0], dtype=np.uint16)
    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start:start + chunk_size]
        local_best_dist = np.full(chunk.shape[0], np.inf, dtype=np.float32)
        local_best_codes = np.zeros(chunk.shape[0], dtype=np.uint16)
        for parity, shift in ((0, np.float32(0.25)), (1, np.float32(-0.25))):
            target = chunk - shift
            abs_target = np.abs(target)
            vec_norm_w = (abs_target * abs_target) @ w
            distances = (
                vec_norm_w[:, None] + abs_norm_w[None, :]
                - 2.0 * (abs_target * w[None, :]) @ abs_rows.T
            )
            negative = target < 0
            target_signs = np.zeros(chunk.shape[0], dtype=np.uint16)
            for out_dim, packed_dim in enumerate(packed):
                target_signs |= negative[:, out_dim].astype(np.uint16) << packed_dim
            effective_signs = (target_signs[:, None] ^ base_signs[None, :]).astype(np.uint16)
            sign_parity = _sign_parity(effective_signs.astype(np.uint32)).astype(bool)
            flip_dim = None
            if np.any(sign_parity):
                flip_penalty = 4 * (abs_target * w[None, :])[:, None, :] * abs_rows[None, :, :]
                flip_dim = np.argmin(flip_penalty, axis=2).astype(np.uint8)
                correction = np.min(flip_penalty, axis=2)
                distances = np.where(sign_parity, distances + correction, distances)
            abs_idx = np.argmin(distances, axis=1).astype(np.uint16)
            row_dist = distances[np.arange(chunk.shape[0]), abs_idx]
            candidate_signs = effective_signs[np.arange(chunk.shape[0]), abs_idx].copy()
            if flip_dim is not None and np.any(sign_parity):
                selected_flip_dim = flip_dim[np.arange(chunk.shape[0]), abs_idx]
                flip_masks = (np.uint16(1) << packed[selected_flip_dim]).astype(np.uint16)
                selected_needs_flip = sign_parity[np.arange(chunk.shape[0]), abs_idx]
                candidate_signs = np.where(
                    selected_needs_flip, candidate_signs ^ flip_masks, candidate_signs
                ).astype(np.uint16)
            stored_signs = (candidate_signs ^ np.uint16(parity)).astype(np.uint16)
            codes = ((abs_idx.astype(np.uint16) << np.uint16(8)) | stored_signs).astype(np.uint16)
            improved = row_dist < local_best_dist
            local_best_dist[improved] = row_dist[improved]
            local_best_codes[improved] = codes[improved]
        best_codes[start:start + chunk.shape[0]] = local_best_codes
    return best_codes.reshape(values.shape[:-1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_e8p_diagonal_hessian.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/codebook/e8.py tests/test_e8p_diagonal_hessian.py
git commit -m "feat: add factored diagonal-Hessian E8P reference encoder"
```

---

### Task 2: Deterministic tie-break guard

**Files:**
- Modify: `src/mlx_vq/codebook/e8.py`
- Test: `tests/test_e8p_diagonal_hessian.py`

**Interfaces:**
- Consumes: `encode_e8p_rtn_diagonal_hessian`.
- Produces: same signature; adds the invariant that when two grid entries are
  exactly equidistant, the **lowest full-grid index (uint16 code)** wins — matching
  `np.argmin` over `e8p_full_grid()`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_e8p_diagonal_hessian.py
def test_exact_tie_breaks_to_lowest_code() -> None:
    # A target equidistant to code A and code B under the metric must resolve to
    # min(A, B), matching the exhaustive full-grid argmin.
    rng = np.random.default_rng(101)
    grid = e8p_full_grid().astype(np.float32)
    diag = np.ones(8, dtype=np.float32)
    # Construct targets at the midpoint of adjacent grid rows so ties occur.
    lo = grid[:2000]
    hi = grid[1:2001]
    midpoints = ((lo + hi) / 2.0).astype(np.float32)
    exact = _exhaustive(midpoints, diag)
    fast = encode_e8p_rtn_diagonal_hessian(midpoints, diag)
    np.testing.assert_array_equal(fast, exact)
```

- [ ] **Step 2: Run test to verify it fails (or confirm it already holds)**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_e8p_diagonal_hessian.py::test_exact_tie_breaks_to_lowest_code -q`
Expected: If it PASSES immediately, the factored argmin ordering already agrees
with the full-grid order (likely, since the unweighted encoder is byte-exact) —
record that and skip Step 3. If it FAILS, implement the guard.

- [ ] **Step 3: Write minimal implementation (only if Step 2 failed)**

If a tie divergence is found, make the two-branch (parity) selection prefer the
lower resulting `code` on an exact `row_dist` tie by comparing candidate codes
before the `improved` update:

```python
            # replace the `improved` block with tie-aware selection:
            better = row_dist < local_best_dist
            tie = row_dist == local_best_dist
            lower_code = codes < local_best_codes
            take = better | (tie & lower_code)
            local_best_dist = np.where(take, row_dist, local_best_dist)
            local_best_codes = np.where(take, codes, local_best_codes).astype(np.uint16)
```

Within a single parity branch the abs-row argmin already matches the full grid
(inherited from `encode_e8p_rtn`); this guard only orders the ±0.25 cross-parity
tie deterministically.

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_e8p_diagonal_hessian.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/codebook/e8.py tests/test_e8p_diagonal_hessian.py
git commit -m "test: guard deterministic tie-breaking in factored E8P search"
```

---

### Task 3: MLX/Metal factored search (memory-bounded)

**Files:**
- Create: `src/mlx_vq/quant/e8p_metal.py`
- Test: `tests/test_e8p_metal.py`

**Interfaces:**
- Consumes: `e8p_abs_grid`, `E8P_SHUFFLE_MAP`, `CODEWORD_DIM`.
- Produces: `encode_e8p_diagonal_hessian_mlx(vectors, diagonal, *, row_block: int = 1_000_000) -> np.ndarray` (uint16), byte-identical to the NumPy reference, without materializing an `[N,256,8]` tensor (compute the parity-flip minimum via a reduction over the 8 dims in a bounded loop).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e8p_metal.py
from __future__ import annotations

import numpy as np
import pytest

from mlx_vq.codebook.e8 import encode_e8p_rtn_diagonal_hessian

mx = pytest.importorskip("mlx.core")


def _metal_ok() -> bool:
    metal = getattr(mx, "metal", None)
    try:
        return metal is not None and bool(metal.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")
def test_mlx_matches_numpy_reference_bit_for_bit() -> None:
    from mlx_vq.quant.e8p_metal import encode_e8p_diagonal_hessian_mlx

    rng = np.random.default_rng(20260711)
    for _ in range(3):
        vecs = (rng.standard_normal((5000, 8)) * 0.6).astype(np.float32)
        diag = np.abs(rng.standard_normal(8)).astype(np.float32) + 0.05
        ref = encode_e8p_rtn_diagonal_hessian(vecs, diag)
        got = encode_e8p_diagonal_hessian_mlx(vecs, diag)
        np.testing.assert_array_equal(got, ref)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_e8p_metal.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_vq.quant.e8p_metal'` (or SKIP off-Metal — in which case verify on the host before merging).

- [ ] **Step 3: Write minimal implementation**

```python
# src/mlx_vq/quant/e8p_metal.py
from __future__ import annotations

import numpy as np
import mlx.core as mx

from mlx_vq.codebook.e8 import CODEWORD_DIM, E8P_SHUFFLE_MAP, e8p_abs_grid


def encode_e8p_diagonal_hessian_mlx(
    vectors: np.ndarray, diagonal: np.ndarray, *, row_block: int = 1_000_000
) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    flat_np = values.reshape(-1, CODEWORD_DIM)
    w = mx.array(np.asarray(diagonal, dtype=np.float32))
    base_rows_np = e8p_abs_grid().astype(np.float32)
    abs_rows = mx.array(np.abs(base_rows_np))                       # [256,8]
    abs_norm_w = (abs_rows * abs_rows) @ w                          # [256]
    packed = E8P_SHUFFLE_MAP.astype(np.uint16)
    base_signs_np = np.zeros(base_rows_np.shape[0], dtype=np.uint16)
    for out_dim, packed_dim in enumerate(packed):
        base_signs_np |= (base_rows_np[:, out_dim] < 0).astype(np.uint16) << packed_dim
    base_signs = mx.array(base_signs_np.astype(np.uint32))

    out = np.empty(flat_np.shape[0], dtype=np.uint16)
    for start in range(0, flat_np.shape[0], row_block):
        chunk = mx.array(flat_np[start:start + row_block])
        best_codes = None
        best_dist = None
        for parity, shift in ((0, 0.25), (1, -0.25)):
            target = chunk - shift
            abs_target = mx.abs(target)
            vec_norm_w = (abs_target * abs_target) @ w
            distances = vec_norm_w[:, None] + abs_norm_w[None, :] - 2.0 * (abs_target * w[None, :]) @ abs_rows.T
            # parity of effective signs (per row, per abs-row)
            negative = target < 0
            target_signs = mx.zeros((chunk.shape[0],), dtype=mx.uint32)
            for out_dim, packed_dim in enumerate(packed):
                target_signs = target_signs | (negative[:, out_dim].astype(mx.uint32) << int(packed_dim))
            effective = target_signs[:, None] ^ base_signs[None, :]
            parity_bits = mx.zeros(effective.shape, dtype=mx.uint32)
            for b in range(8):
                parity_bits = parity_bits ^ ((effective >> b) & 1)
            needs_flip = parity_bits == 1
            # min weighted flip penalty over the 8 dims WITHOUT an [N,256,8] tensor:
            wt = abs_target * w[None, :]                            # [N,8]
            correction = None
            for d in range(CODEWORD_DIM):
                pen_d = 4.0 * wt[:, d][:, None] * abs_rows[:, d][None, :]   # [N,256]
                correction = pen_d if correction is None else mx.minimum(correction, pen_d)
            distances = mx.where(needs_flip, distances + correction, distances)
            abs_idx = mx.argmin(distances, axis=1)
            row_dist = mx.take_along_axis(distances, abs_idx[:, None], axis=1)[:, 0]
            # code reconstruction mirrors the NumPy reference; gather sign bits
            eff_sel = mx.take_along_axis(effective, abs_idx[:, None], axis=1)[:, 0]
            flip_sel = mx.take_along_axis(needs_flip.astype(mx.uint32), abs_idx[:, None], axis=1)[:, 0]
            # flip dim = argmin over dims of pen at the selected abs row (recompute cheaply)
            sel_abs = mx.take_along_axis(abs_rows[None, :, :].reshape(256, 8)[abs_idx], None, axis=0) if False else None  # see note
            codes = ((abs_idx.astype(mx.uint32) << 8) | (eff_sel ^ mx.array(np.uint32(parity)))).astype(mx.uint16)
            if best_codes is None:
                best_codes, best_dist = codes, row_dist
            else:
                take = row_dist < best_dist
                best_codes = mx.where(take, codes, best_codes)
                best_dist = mx.where(take, row_dist, best_dist)
        mx.eval(best_codes)
        out[start:start + int(chunk.shape[0])] = np.asarray(best_codes).astype(np.uint16)
    return out.reshape(values.shape[:-1])
```

> Implementation note: the flip-dimension selection (which single dim to flip on
> a parity violation) needs the same argmin-over-dims the NumPy reference does.
> The cheap way on Metal: after choosing `abs_idx`, gather that row's per-dim
> penalties `4·wt·abs_rows[abs_idx]` ([N,8]) and `argmin` over 8 — an [N,8]
> tensor, not [N,256,8]. Wire that in place of the `sel_abs`/`False` placeholder
> and XOR the flip mask into `eff_sel` exactly as the NumPy reference does. The
> test in Step 1 is the gate: it must be byte-identical to the reference before
> this task is done.

- [ ] **Step 4: Run test to verify it passes (on a Metal host)**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_e8p_metal.py -q`
Expected: PASS on a Metal Mac (SKIP elsewhere — must be verified on the host
before merge).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/quant/e8p_metal.py tests/test_e8p_metal.py
git commit -m "feat: add MLX/Metal factored diagonal-Hessian E8P search"
```

---

### Task 4: Dispatcher + wire into the materializer (byte-exact swap)

**Files:**
- Modify: `src/mlx_vq/quant/rtn.py` (add `nearest_e8p_codes_diagonal_hessian`)
- Modify: `src/mlx_vq/convert/glm52_recovery_materialize.py` (use it for
  `code_bits==16`)
- Test: `tests/test_e8p_diagonal_hessian.py` (dispatcher parity),
  `tests/test_rtn_quantization.py` (regression)

**Interfaces:**
- Produces: `nearest_e8p_codes_diagonal_hessian(vectors, diagonal, *, backend: str = "metal") -> np.ndarray` — `backend="numpy"` uses `encode_e8p_rtn_diagonal_hessian`; `backend="metal"` uses the MLX path with a NumPy fallback when Metal is unavailable. Byte-identical to the exhaustive full-grid search either way.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_e8p_diagonal_hessian.py
def test_dispatcher_numpy_backend_matches_exhaustive() -> None:
    from mlx_vq.quant.rtn import nearest_e8p_codes_diagonal_hessian

    rng = np.random.default_rng(55)
    vecs = (rng.standard_normal((4000, 8)) * 0.6).astype(np.float32)
    diag = np.abs(rng.standard_normal(8)).astype(np.float32) + 0.05
    np.testing.assert_array_equal(
        nearest_e8p_codes_diagonal_hessian(vecs, diag, backend="numpy"),
        _exhaustive(vecs, diag),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_e8p_diagonal_hessian.py::test_dispatcher_numpy_backend_matches_exhaustive -q`
Expected: FAIL with `ImportError: cannot import name 'nearest_e8p_codes_diagonal_hessian'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/quant/rtn.py
from mlx_vq.codebook.e8 import encode_e8p_rtn_diagonal_hessian


def nearest_e8p_codes_diagonal_hessian(vectors, diagonal, *, backend: str = "metal"):
    """Fast factored E8P code search; byte-exact vs the exhaustive full grid."""
    if backend == "numpy":
        return encode_e8p_rtn_diagonal_hessian(vectors, diagonal)
    if backend == "metal":
        try:
            from mlx_vq.quant.e8p_metal import encode_e8p_diagonal_hessian_mlx

            return encode_e8p_diagonal_hessian_mlx(vectors, diagonal)
        except Exception:
            # Deterministic fallback keeps output identical off-Metal.
            return encode_e8p_rtn_diagonal_hessian(vectors, diagonal)
    raise ValueError(f"unknown backend {backend!r}")
```

Swap the `code_bits==16` branch in `assign_codes`
(`src/mlx_vq/convert/glm52_recovery_materialize.py:696-704`):

```python
                else:
                    from mlx_vq.quant.rtn import nearest_e8p_codes_diagonal_hessian

                    codes_by_group[:, group, codeword] = nearest_e8p_codes_diagonal_hessian(
                        normalized, hessian, backend=e8p_search_backend
                    ).astype(codes_dtype)
```

Add an `e8p_search_backend: str = "metal"` parameter to
`quantize_weight_importance_aware` (default `"metal"`), threaded from the
materializer entry point so a byte-identity regression run can force
`"numpy"`. Note: `normalized` is `[out_dim, 8]`; the dispatcher preserves that
leading shape.

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_e8p_diagonal_hessian.py tests/test_rtn_quantization.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/quant/rtn.py src/mlx_vq/convert/glm52_recovery_materialize.py tests/test_e8p_diagonal_hessian.py
git commit -m "feat: route E8P recovery search through fast factored dispatcher"
```

---

### Task 5: Expert-level checkpoint + progress/ETA

**Files:**
- Modify: `src/mlx_vq/convert/glm52_recovery_materialize.py` (outer per-expert
  loop, `~:1783-1862`)
- Test: `tests/test_e8p_recovery_checkpoint.py`

**Interfaces:**
- Produces:
  `ExpertCheckpoint(group_key: str, completed_experts: tuple[int, ...])` persisted
  transactionally per group;
  `record_expert_done(checkpoint_dir, group_key, expert) -> None`;
  `completed_experts(checkpoint_dir, group_key) -> frozenset[int]`;
  `ProgressReport(done: int, total: int, seconds_elapsed: float)` with
  `.fraction`, `.eta_seconds`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e8p_recovery_checkpoint.py
from __future__ import annotations

from pathlib import Path

from mlx_vq.convert.glm52_recovery_materialize import (
    ProgressReport,
    completed_experts,
    record_expert_done,
)


def test_expert_checkpoint_is_durable_and_resumable(tmp_path: Path) -> None:
    key = "layer-00075-gate_proj"
    assert completed_experts(tmp_path, key) == frozenset()
    record_expert_done(tmp_path, key, 0)
    record_expert_done(tmp_path, key, 1)
    assert completed_experts(tmp_path, key) == frozenset({0, 1})
    # A different group is independent.
    assert completed_experts(tmp_path, "layer-00075-up_proj") == frozenset()


def test_progress_report_eta() -> None:
    report = ProgressReport(done=42, total=168, seconds_elapsed=210.0)
    assert abs(report.fraction - 0.25) < 1e-9
    # 42 experts in 210s -> 5s/expert -> 126 remaining -> 630s ETA
    assert abs(report.eta_seconds - 630.0) < 1e-6


def test_progress_report_zero_done_has_no_false_eta() -> None:
    report = ProgressReport(done=0, total=168, seconds_elapsed=10.0)
    assert report.eta_seconds is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_e8p_recovery_checkpoint.py -q`
Expected: FAIL with `ImportError: cannot import name 'ProgressReport'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mlx_vq/convert/glm52_recovery_materialize.py  (new helpers)
import json
import os
from dataclasses import dataclass
from pathlib import Path


def _checkpoint_path(checkpoint_dir, group_key: str) -> Path:
    return Path(checkpoint_dir) / f"{group_key}.experts.json"


def completed_experts(checkpoint_dir, group_key: str) -> frozenset[int]:
    path = _checkpoint_path(checkpoint_dir, group_key)
    if not path.exists():
        return frozenset()
    data = json.loads(path.read_text())
    return frozenset(int(e) for e in data.get("completed_experts", []))


def record_expert_done(checkpoint_dir, group_key: str, expert: int) -> None:
    path = _checkpoint_path(checkpoint_dir, group_key)
    current = set(completed_experts(checkpoint_dir, group_key))
    current.add(int(expert))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"group_key": group_key, "completed_experts": sorted(current)}))
    os.replace(tmp, path)  # atomic within the same filesystem


@dataclass(frozen=True)
class ProgressReport:
    done: int
    total: int
    seconds_elapsed: float

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0

    @property
    def eta_seconds(self) -> float | None:
        if self.done <= 0:
            return None
        rate = self.done / self.seconds_elapsed if self.seconds_elapsed else 0.0
        if rate <= 0:
            return None
        return (self.total - self.done) / rate
```

In the outer per-expert loop (`~:1783-1862`): skip experts already in
`completed_experts(...)`, call `record_expert_done(...)` after each expert's
codes are written to the group buffer, and emit a `ProgressReport` line per
expert. Publish the transactional group manifest only after all experts are
recorded — so a stopped run resumes mid-group without ever showing a
false-complete group.

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_e8p_recovery_checkpoint.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/convert/glm52_recovery_materialize.py tests/test_e8p_recovery_checkpoint.py
git commit -m "feat: add expert-level resumable checkpoints and progress/ETA"
```

---

### Task 6: Microbenchmark + acceptance gate

**Files:**
- Create: `benchmarks/bench_e8p_search.py`
- Test: `tests/test_e8p_search_bench.py`

**Interfaces:**
- Produces: `measure_backend(backend, *, n_codewords, seed) -> dict` returning
  `{"backend", "codewords_per_second", "seconds"}`;
  `project_worst8_hours(codewords_per_second, *, experts=168, out_dim=2048, in_dim=6144, passes=4, groups=24) -> float`;
  `acceptance_gate(hours, *, max_hours=12.0) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e8p_search_bench.py
from __future__ import annotations

from benchmarks.bench_e8p_search import acceptance_gate, project_worst8_hours


def test_projection_matches_spike_scale() -> None:
    # Spike measured ~1.04M cw/s on Metal -> ~6.8 h for worst8.
    hours = project_worst8_hours(1_040_000)
    assert 5.0 < hours < 9.0


def test_acceptance_gate_rejects_multiday() -> None:
    assert acceptance_gate(project_worst8_hours(22_900)) is False  # exhaustive ~308 h
    assert acceptance_gate(project_worst8_hours(1_040_000)) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_e8p_search_bench.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks.bench_e8p_search'`.

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/bench_e8p_search.py
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def project_worst8_hours(
    codewords_per_second: float,
    *,
    experts: int = 168,
    out_dim: int = 2048,
    in_dim: int = 6144,
    passes: int = 4,
    groups: int = 24,
) -> float:
    cw_per_group = out_dim * (in_dim // 8) * experts * passes
    total = cw_per_group * groups
    return total / codewords_per_second / 3600.0


def acceptance_gate(hours: float, *, max_hours: float = 12.0) -> bool:
    return hours <= max_hours


def measure_backend(backend: str, *, n_codewords: int, seed: int) -> dict:  # pragma: no cover - host timing
    from mlx_vq.quant.rtn import nearest_e8p_codes_diagonal_hessian

    rng = np.random.default_rng(seed)
    vecs = (rng.standard_normal((n_codewords, 8)) * 0.6).astype(np.float32)
    diag = np.abs(rng.standard_normal(8)).astype(np.float32) + 0.05
    nearest_e8p_codes_diagonal_hessian(vecs[:64], diag, backend=backend)  # warm-up
    start = time.perf_counter()
    nearest_e8p_codes_diagonal_hessian(vecs, diag, backend=backend)
    seconds = time.perf_counter() - start
    return {"backend": backend, "seconds": seconds, "codewords_per_second": n_codewords / seconds}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - host path
    parser = argparse.ArgumentParser(prog="bench_e8p_search")
    parser.add_argument("--backend", default="metal", choices=["metal", "numpy"])
    parser.add_argument("--codewords", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    result = measure_backend(args.backend, n_codewords=args.codewords, seed=20260711)
    hours = project_worst8_hours(result["codewords_per_second"])
    result["projected_worst8_hours"] = hours
    result["acceptance_gate_pass"] = acceptance_gate(hours)
    print(json.dumps(result, indent=2))
    return 0 if result["acceptance_gate_pass"] else 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_e8p_search_bench.py -q`
Expected: PASS (2 passed). Then, on the host, run
`uv run python benchmarks/bench_e8p_search.py --backend metal` and confirm
`acceptance_gate_pass: true` before any 24-group rerun.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/bench_e8p_search.py tests/test_e8p_search_bench.py
git commit -m "feat: add E8P search microbenchmark and acceptance gate"
```

---

### Task 7: End-to-end byte-identity fixture + docs

**Files:**
- Test: `tests/test_e8p_diagonal_hessian.py` (small end-to-end
  `quantize_weight_importance_aware` parity)
- Modify: `docs/GLM45_AIR_RC_PIPELINE.md` or the recovery docs (note the fast path)

**Interfaces:**
- Consumes: `quantize_weight_importance_aware` with both backends.
- Produces: a regression test proving the whole materialize call produces
  identical codes on `backend="numpy"` (fast) vs an exhaustive baseline for a
  small synthetic expert weight.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_e8p_diagonal_hessian.py
def test_materialize_fast_path_matches_exhaustive_end_to_end() -> None:
    from mlx_vq.convert.glm52_recovery_materialize import quantize_weight_importance_aware

    rng = np.random.default_rng(2026)
    weight = (rng.standard_normal((16, 64)) * 0.3).astype(np.float32)  # out=16, in=64
    diagonal = np.abs(rng.standard_normal(64)).astype(np.float32) + 0.05
    fast = quantize_weight_importance_aware(
        weight, diagonal, group_size=64, code_bits=16, iterations=3, e8p_search_backend="numpy"
    )
    exhaustive = quantize_weight_importance_aware(
        weight, diagonal, group_size=64, code_bits=16, iterations=3, e8p_search_backend="exhaustive"
    )
    np.testing.assert_array_equal(fast.codes, exhaustive.codes)
    np.testing.assert_array_equal(fast.scales, exhaustive.scales)
```

> Add an `"exhaustive"` backend option to the dispatcher (Task 4) that calls the
> original `nearest_codebook_indices_diagonal_hessian(..., codebook=e8p_full_grid())`
> so the baseline stays available for this regression and for spot audits.

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_e8p_diagonal_hessian.py::test_materialize_fast_path_matches_exhaustive_end_to_end -q`
Expected: FAIL until the `"exhaustive"` backend option and the `e8p_search_backend`
thread-through are complete.

- [ ] **Step 3: Write minimal implementation**

Extend the dispatcher with the `"exhaustive"` backend:

```python
# src/mlx_vq/quant/rtn.py  (extend nearest_e8p_codes_diagonal_hessian)
    if backend == "exhaustive":
        from mlx_vq.codebook.e8 import e8p_full_grid

        return nearest_codebook_indices_diagonal_hessian(
            vectors, diagonal, codebook=e8p_full_grid().astype(np.float32),
            index_dtype=np.dtype(np.uint16),
            vector_chunk_size=max(1, vectors.shape[0]) if vectors.ndim > 1 else 4096,
            codebook_chunk_size=8192,
        )
```

Document the fast path in the recovery docs: note that E8P recovery search runs
on MLX/Metal, is byte-identical to the exhaustive baseline, and that the
`"exhaustive"` backend remains available for audit.

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_e8p_diagonal_hessian.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/quant/rtn.py tests/test_e8p_diagonal_hessian.py docs/GLM45_AIR_RC_PIPELINE.md
git commit -m "test: end-to-end byte-identity for fast E8P materialize path"
```

---

## Self-Review

- **Spec coverage (Codex's required wave):** microbenchmark (Task 6),
  expert-level checkpoint + progress (Task 5), exact lattice/factored decoder
  replacing the 65,536 scan (Tasks 1–2), Metal acceleration (Task 3), preserved
  diagonal-Hessian semantics + uint16 identity + deterministic tie-break (Tasks
  1–2, 4), byte-identity vs exhaustive on fixtures (Tasks 1, 7), acceptance gate
  before the 24-group rerun (Task 6). Bounded process parallelism is intentionally
  omitted (Metal already clears the gate; add only if needed).
- **MLX/Metal standardization:** the hot path is Metal (Task 3); NumPy remains as
  the parity reference and deterministic fallback only.
- **Byte-exact discipline:** every optimization task has a byte-identity gate
  against the exhaustive baseline, which stays available as the `"exhaustive"`
  backend.
- **Placeholder scan:** the only marked-incomplete spot is the MLX flip-dim gather
  in Task 3 Step 3, which is called out explicitly with the exact `[N,8]`-not-
  `[N,256,8]` approach and gated by the byte-identity test — not a silent TODO.
- **Type consistency:** `encode_e8p_rtn_diagonal_hessian`,
  `nearest_e8p_codes_diagonal_hessian` (backends `numpy`/`metal`/`exhaustive`),
  `encode_e8p_diagonal_hessian_mlx`, `ProgressReport`, checkpoint helpers used
  consistently across tasks.
- **Downstream (out of scope, deferred to the campaign):** the worst8 rerun,
  audit, 66-row reeval, re-attribution, and decision packet (Codex P1) run only
  after Task 6's gate passes.
