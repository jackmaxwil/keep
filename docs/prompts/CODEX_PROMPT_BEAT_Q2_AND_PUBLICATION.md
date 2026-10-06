# Prompt for Codex — Beat q2 (INT8-NAX) + Publication-Grade Benchmarks

> Paste below the line to Codex. Two workstreams: (A) lock publication-grade numbers for the current FP16-NAX win, and (B) make INT8-NAX actually beat q2. FP16-NAX stays the default until INT8 passes both a speed and a quality gate.

---

## Where we are (grounded, do not re-derive)
- Public `vq_e1_routed` defaults to `prefill_engine="auto"` → sorted-A Steel NAX FP16 path; resident `prefill_1k` median **1.6777s vs q2 1.1985s = 1.40×**, `decode_128` ≈4.428s (~29 tok/s), accepted rows pageouts/swapouts `0/0`. Commit `522d86f` on `codex/nax-prefill-parity`.
- q2 baseline is confirmed genuine NAX via the N0 LLDB tap: it launches `affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2`, **no `bk32` request** — the #3632 landmine did not bite. Keep using this audited sorted-q2 (`mlx_q2_routed_g128`) as the only baseline.
- INT8 (`NaxE8Int8RoutedMatmul`, variant `nax_e8_int8`) is **correct but slower than FP16**: gate/up M=1024 ≈10.36ms full (incl A8 quant), ≈9.83ms prequantized native-only, vs FP16 staged native-only ≈8.97ms. Per DISCOVERY, the overhead is **per-group int32 accumulation/rescale + A8 activation quantization**, NOT weight mapping (which is exact: `q_w = codebook[code]-8`, `s_w = 0.5*scale`).
- Best FP16 geometry today: q2-like `64 routes × 64 cols`, `BK=64`, direct device codebook reads, staged codeword tile, `route_tile=64` (`nax_e8_fp16_sorted_steel`).
- Down projection `group_size=352` is unaligned to 32/64 — already the fiddliest case for FP16 and worse for INT8 per-group rescale.
- Tooling gap: `xctrace`/`metal`/`metallib` are unavailable via `xcrun`; N0 used CLT LLDB taps.

## Read first
`docs/PLAN_PREFILL_NAX.md` (N6 INT8 + risk ledger), `docs/plans/glm-4-5-vq-parity-plan.md`, `WORK_LOG.md` (N4/N6 loops, the publication variance caveat at the N5 gate), `DISCOVERY.md` (the N6 INT8 overhead rows, down-352 facts), `native/vq_nax_ext/`, `src/mlx_vq/kernels/nax.py`, `src/mlx_vq/benchmark/projection_kernels.py` and `glm45_air.py`.

## Your task
Author the plan updates and then execute, in this order: **Workstream A first (cheap, certain), Workstream B in parallel/after.** Do not change the public default away from FP16-NAX unless B passes both gates. This is real implementation, gated in our Ralph-loop style.

---

## Workstream A — Publication-grade benchmarks (lock the FP16 win)
**Goal:** credible, repeatable numbers for the current default, including the historically-elusive clean 4K resident row.

- **Engines × scenarios:** `vq_e1_routed` (auto default), `vq_e1_routed_nax_e8` (pinned NAX), `vq_e1_routed_vq_metal` (pinned Metal), and `mlx_q2_routed_g128` (baseline) × `{decode_128, prefill_1k, prefill_4k}`. Plus projection microbench rows at `M=1024/2048/4096` for the same engines.
- **Statistics:** ≥5 clean repetitions per cell; report **median + min + p90 + stddev**, not just median. Fresh process per row.
- **Clean 4K is a first-class deliverable.** It has never landed clean (P1 swapouts 50612; R3 `pageouts_delta=429`). Use the existing context caps/preflight; only accept rows with `pageouts_delta==0 && swapouts_delta==0`. If 4K cannot be made clean on this host, record exactly why (memory headroom, caps) rather than reporting a dirty number.
- **Methodology capture** (for a writeup): host (`Apple M5 Max`, `applegpu_g17s`), MLX `0.31.2`/mlx-lm `0.31.3`, macOS version, the audited q2/VQ kernel names (reuse the LLDB tap), warmup/iteration counts, memory discipline, commit hash.
- **Optional GPU-counter evidence:** if you want capture beyond LLDB kernel names, install/select full Xcode for `xctrace`; otherwise document the LLDB-tap method as the reproducible path. Don't block on it.
- **Output:** a results table/figure artifact (e.g. `artifacts/benchmarks/glm45-air-nax-publication.jsonl` + a short `docs/research/NAX_PARITY_RESULTS.md` with the tables and methodology).
- **Ledger nit:** update the `WORK_LOG.md` "Known good commit/state" line from "uncommitted" to `522d86f`.

**Gate A:** every published row is `0/0` on pageouts/swapouts with ≥5 reps and reported spread; a clean 4K row exists or its absence is explained with evidence.

---

## Workstream B — Beat q2 via INT8-NAX
**Goal:** make INT8-NAX materially faster than FP16-NAX and ideally `<1.0×` q2, without unacceptable quality loss. The weight side is lossless; the work is killing the A8/rescale overhead.

Sequence:
1. **Sanity-isolate the ceiling first.** Microbench raw `matmul2d` INT8×INT8→INT32 vs FP16×FP16→FP32 on the exact Air tiles (`64×64×64`). Confirm the ~2× INT8 TOPS is actually reachable on this host before optimizing the wrapper. If raw INT8 is **not** ~2× FP16, the INT8 thesis is weak — record and consider stopping B.
2. **Cut A8 activation-quant overhead.** Quantize x once per token and **reuse across gate+up** (shared input) and across routed copies; fuse the A8 quant into the producing op's epilogue where possible rather than a standalone pass. Measure the standalone-quant cost you remove.
3. **Remove per-group int32 rescale stalls.** Restructure so int32 accumulation runs over a full K-tile with scale applied to int32 partials in registers/fp32 at group boundaries, minimizing boundary flushes. gate/up `gs=512` over `K=4096` = 8 groups; down `gs=352` over `K=1408` is unaligned.
4. **Resolve the down-352 hazard.** Decide explicitly: specialize the 352 K-tail, **or re-export the down artifact at a 32/64-aligned group size (e.g. 128)** to make INT8 (and FP16) cleaner. This is a conversion-param change, not an on-disk-format change — record the decision and quality impact.
5. **Activation outlier handling if needed.** If A8 garbles MoE quality, apply SmoothQuant/per-channel mitigation (cider warns W8A8 needs it). Router/shared experts stay high precision.

**Gate B (both required to displace FP16 default):**
- **Speed:** INT8-NAX beats FP16-NAX on gate/up and down at `M=1024/2048/4096`, and lands `<1.0×` sorted q2 on at least gate/up; pageouts/swapouts `0/0`.
- **Quality:** one-layer PPL degradation `<1%` vs the FP16-NAX VQ path (extend toward a small multi-layer / full-model PPL check before shipping). Projection cosine stays acceptable.
- If either fails: keep FP16-NAX default, leave INT8 as `nax_int8` pinned/diagnostic, and record the stop reason.

---

## Invariants (both workstreams)
- Baseline is always the audited **sorted `mlx_q2_routed_g128`**. Never unsorted q2.
- **Do not regress the banked wins:** decode M=1 path untouched; FP16-NAX prefill stays default until Gate B passes. Re-check `decode_128` tok/s after any engine change.
- Correctness before throughput: cosine ≥0.99999 vs dequant-then-dense (projections) / vs FP16-NAX VQ output (engine swaps).
- Every headline number: ≥5 reps, median+spread, `pageouts_delta==0 && swapouts_delta==0`; fresh process.
- Keep the on-disk VQ format; the only conversion change on the table is the down group-size re-export (B4), recorded as a decision.
- Ralph-loop discipline: per-slice objective, gate, cheapest verification, evidence (commands + artifact JSONL), GO/STOP, recheck triggers. Update `docs/PLAN_PREFILL_NAX.md`, the parity plan, `WORK_LOG.md`, and `DISCOVERY.md`.

## Deliverables
1. `docs/research/NAX_PARITY_RESULTS.md` — publication tables (engines × scenarios, median+spread), clean-4K result or explained absence, methodology, commit hash.
2. Plan/ledger updates: N6 INT8 firmed into speed+quality sub-gates; a "Publication" phase row; `WORK_LOG.md` known-good-state → `522d86f`; DISCOVERY rows for the raw-INT8 ceiling result and the down-352 decision.
3. Code: the INT8 optimizations (only merged behind the pinned engine; default unchanged unless Gate B passes), the publication benchmark harness/variants, and tests kept green (`uv run pytest … && git diff --check`).

## Do NOT
- Do not flip the public default to INT8 unless Gate B (speed AND quality) passes.
- Do not report any 4K/headline number with nonzero pageouts/swapouts.
- Do not let INT8 work regress decode or the FP16-NAX prefill default.
