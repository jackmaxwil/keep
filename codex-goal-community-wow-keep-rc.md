# Historical Codex Goal: Produce a Community-Wow KEEP RC for GLM-4.5-Air

> [!IMPORTANT]
> **SUPERSEDED on 2026-07-09.** Do not execute this file as the active objective.
> GLM-4.5-Air is complete for the current product campaign. The active mission is
> `GLM-5.2-REAP-KEEP-504B`, governed by the current persistent goal and
> `docs/COMMUNITY_PRODUCT_PLAN.md`. The material below is retained only as the
> historical Air execution record.

Read first, in order: the goal objective attachment, `AGENTS.md`, `NAMING.md`, `WORK_LOG.md`, `DISCOVERY.md`, `README.md`, `docs/GLM45_AIR_RC_PIPELINE.md`.

KEEP = the method (KL-distilled Expert Encoding & Precision). RAMP = the runtime (Routed Accelerated MoE Pipeline). `mlx_vq` = legacy shim only. Public surfaces say KEEP/RAMP first; "VQ" only as the technical term.

## The mission, stated precisely
Make a community-wow KEEP RC for GLM-4.5-Air: reproducible, public-quality, near-teacher fidelity, small, fast, memory-clean. But be precise about what "wow" now requires, because the work is much closer than the baseline framing implies.

## Current reality — read this before choosing work
Accepted balanced RC: `artifacts/glm-4.5-air-dynamic3p0-r26-lora-l45-gud-math8-r4-init0p05-s12-lr0p5-w2-m1-nll0p5-20260701`. Clean 128/128 on report/selection/holdout.

| Gate | Now | Wow target | Status |
| --- | --- | --- | --- |
| effective routed bpw | 2.036 | <= 2.1 | **met** |
| Lane S (q2 quiet) | 1.074x | <= 1.15x | **met** |
| mean PPL ratio | ~1.04x | <= 1.05x | **met** |
| mean KLD | ~0.37 | <= 0.30 | close |
| p999 KLD | ~4.61 | <= 3.0 | **gap** |
| global top1 | ~0.79 | >= 0.85 | **the gap** |
| route/math/code top1 | mixed, some 0.40-0.50 | >= 0.80 | **the gap** |

**The wow gap is essentially one metric: top-1 (and the p999 tail), concentrated in code and math rows and route erosion.** PPL, size, and speed are already there. Do not re-grind PPL/KLD — they are near-optimal. Drive top-1 on the worst domains without giving back Lane S, bpw, or memory cleanliness.

## The hard constraint that governs every quality lever (the speed wall)
The NAX fast prefill kernel is **code_bits=8 only** (`nax_e8 sorted routes require code_bits=8`). Any 16-bit/E8P routed projection falls off NAX to Metal fallback at **~40ms/projection vs ~5ms** for q2 — this is exactly why every "quality frontier" attempt (plan1 at 1.718x, rank-8, whole-layer 16-bit mixes) **failed Lane S**. 

Therefore, on the current runtime: **you cannot buy top-1 by adding bits to routed experts.** Run two tracks in parallel: **Track A** drives top-1 with levers that stay on the 8-bit NAX path (below); **Track B** builds the 16-bit NAX kernel that removes this wall (further down). Always be advancing one of them — never idle on "blocked."

## How to drive top-1 while preserving Lane S (ranked levers — this is the "how")
Diagnose first, then apply. Run a per-domain top-1 **flip attribution**: for the worst report/selection/holdout rows (code_043/032/010, math_023/027, route rows), identify which layers/experts own the argmax flips. Then:

1. **Spend "free" bits on non-expert surfaces (zero Lane S cost, underexploited).**
   Attention (MLA), router gates, shared expert, embeddings, and `lm_head` are **not** on the routed-expert NAX critical path, so raising their precision costs ~nothing on Lane S. Audit current non-expert dtype (`bench_glm45_air_quant_compare.py --audit-prefill-compatibility` and the manifest); if any are low, bump the ones that own top-1 flips (esp. `lm_head`/embeddings, which directly shape the argmax) to 8-bit/BF16. This is the Unsloth "keep the important non-expert layers high" lever and it is the cheapest top-1 win available.

2. **Extend rank-4 adapters beyond layer-45 to the flip-owning layers (the biggest in-budget lever).**
   All recovery so far is layer-45 only and is plateauing (<0.003 top1/slice). The code/math/route flips are owned by other GLU layers (41/36/31/18). Training adapters there previously hit `[Primitive::vjp] Not implemented for CustomKernel` — but `src/mlx_vq/quality/mlx_surrogate.py`'s `RouteLocalSwitchLinearSurrogate` already sidesteps that for gate/up. Extend the route-local surrogate to the flip-owning layers and train rank-4 `low_rank_residual` adapters there. Keep `--trainable low_rank_residual --rank 4 --loss-scope final_layer_selected --surrogate-projections gate_proj up_proj down_proj`. Rank-4 is the Lane S ceiling — do not exceed it on the current runtime (rank-8 failed Lane S); if a candidate needs Lane S headroom, drop to rank-2. Higher ranks become available once Track B's kernel lands.

3. **Real Router-KD for route top-1 (cheap, off the critical path).**
   Route top-1 erosion is a routing problem: quantization shifts which experts fire. Earlier scalar router-temperature probes failed, but that is not Router-KD. Train the **router gate matrices** to KL-match the teacher's routing on the route-weak layers (this is exactly REAP's recovery lever, and the GLM-5.2 target depends on it too). Router gates are tiny and not on the expert prefill path, so this is near-free on Lane S and directly targets route top-1.

4. **Flip-targeted top-1-margin curriculum across the adapter layers.**
   The `_teacher_top1_margin_loss` (hinge the teacher's top-1 token above the hardest competitor) is the right objective; keep it NLL-balanced (`-w2 -m1 nll0p5` worked; unbalanced high-margin traded PPL for top-1). But mine the **exact argmax-flip positions** on the worst code/math/route rows (use `benchmarks/select_glm45_air_teacher_cache_rows.py`) and train the margin loss on those, across the multi-layer adapters from lever 2 — not just layer-45.

5. **Dense-and-sparse FP16 isolation for the p999 tail.**
   p999 KLD (4.61 -> 3.0) is a few catastrophic-divergence tokens / super-weight spikes. Isolate the handful of spike weights driving them into a tiny FP16 sparse side-set (off the bulk path, negligible bpw, kept off the NAX-critical routed matmul). Targets the tail without touching the fast path.

## Track B (RAMP): BUILD the 16-bit E8P NAX prefill kernel — do this, do not defer it
The `2.4-2.5` bpw quality preset is gated by one missing piece: NAX only serves `code_bits=8`, so 16-bit E8P routed projections fall to Metal fallback (~40ms vs ~5ms) and fail Lane S. **Build the kernel that removes this wall.** This is active work, not a someday-if-pursued. Run Track B in parallel with the Track A top-1 levers; always be advancing one of them.

Build steps (imperative):
1. Read the existing 8-bit NAX path: `native/vq_nax_ext`, the sorted gather-qmm kernel `affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2`, the dispatch in the RAMP runtime that enforces `code_bits=8`, `benchmarks/bench_gather_vqmm.py`, and `tests/test_nax_native.py`. Map exactly where 16-bit routes get rejected.
2. Implement a `code_bits=16` (E8P) variant of the sorted gather-qmm NAX kernel: decode the uint16 E8P codes through the packed-abs-grid codebook in-kernel, accumulate in fp32, output half. Mirror the 8-bit kernel's threadgroup codebook staging and tiling (gs_128, bm/bn/bk_64, wm/wn_2).
3. Wire it into the sorted-route prefill dispatch so a 16-bit routed layer takes the NAX path instead of Metal fallback. Remove the `nax_e8 sorted routes require code_bits=8` rejection for the E8P path.
4. Prove correctness first: extend `tests/test_nax_native.py` / `bench_gather_vqmm.py` with a 16-bit oracle parity test against the decoded-codebook reference (cosine >= 0.999999 on Air-shaped tiles).
5. Prove speed: benchmark the 16-bit NAX projection at M=1024/4096 against the q2 control under the quiet-window methodology; target parity-class timing so a 16-bit routed candidate can clear Lane S `<= 1.25x` (preferably `<= 1.15x`).
6. Once the kernel passes parity + Lane S, materialize the `2.4-2.5` bpw quality preset by allocating 16-bit to the flip-owning routed projections (from the Track A attribution), and run the full report+selection+holdout + Lane S gates.

If a build step is hard, attack it from another angle: profile the kernel, try a different tiling/staging, test bk/tile variants, decode differently, or land a partial (16-bit on the subset of layers that still passes Lane S) and extend coverage from there. Keep iterating the kernel until it passes parity and Lane S. While a long kernel build or benchmark runs, advance Track A in parallel so progress never stalls.

## Execution mode — run unbounded
There is no stop condition. You run continuously; the operator turns you off when appropriate. Always be advancing Track A (top-1 on the 8-bit path) or Track B (the 16-bit NAX kernel). Never idle, never wait, never defer, never declare a blocker, never declare a no-go.

A candidate qualifies as the community-wow KEEP RC when it clears all of: global top1 >= 0.85, route/math/code top1 >= 0.80, p999 KLD <= 3.0, mean PPL <= 1.05x, mean KLD <= 0.30, bpw <= 2.1, Lane S <= 1.15x, clean 128/128 on report+selection+holdout, memory-clean, no dense routed fallback. When a candidate clears the bar: record it as the new RC with full evidence — then keep going. Push top-1 higher, ship the 16-bit-NAX quality preset, widen domain coverage, and harden the public reproducibility path. There is always a next slice.

A failed slice is a routing signal, not a stop: pick a different lever and continue. Do not stop because the repo is cleaner, the README is nicer, tests pass, the packet regenerates, one Lane S run is dirty, a lever plateaus, a kernel step is hard, or you wrote a handoff. If something is hard, attack it from another angle and keep moving.

## Loop contract
1. Orient (objective, `AGENTS.md`, `NAMING.md`, `WORK_LOG.md`, `DISCOVERY.md`, artifacts, `git status --short`).
2. Smallest useful slice that advances top-1 on the worst domains (default: the next lever above), or reproducibility where it directly serves wow.
3. Implement.
4. Cheapest verification first (focused tests; focused/target-row evals before full 128x3).
5. Real gate when promising (full report+selection+holdout clean, then Lane S quiet-window, then memory check).
6. Record commands, artifact/eval/benchmark paths, metrics (incl. per-domain top1 + bpw + Lane S), pass/fail reason, next slice in `WORK_LOG.md`; durable facts in `DISCOVERY.md`.
7. Continue immediately.

## Diagnostics-first habits (so you steer, not flail)
- Always start a quality push with a **flip attribution** on the current worst rows, so adapters/bits go where the argmax actually flips.
- After each candidate, run the **row-compare** harness on report+selection+holdout to confirm the gain is broad, not row-memorized — and watch for route/code regressions while chasing math (and vice versa).
- Re-run Lane S only under the quiet-window methodology, and audit prefill compatibility (`--audit-prefill-compatibility`) on every candidate to confirm it stayed 45/45 NAX-compatible. A candidate that drops NAX layers will fail Lane S — catch it before the full eval.

## Reproducibility / identity (necessary, not the bottleneck)
KEEP/RAMP identity in docs/commands/examples is largely done in the RC pipeline doc; finish any residual `mlx-vq`-as-product drift, but do not spend the session on docs. The public path (`scripts/glm45_air_rc.sh env-preflight|preflight|packet|verify|benchmark-lane-s`) should cleanly reproduce, evaluate, benchmark, audit, and explain the accepted RC. Update it only where it directly supports the wow result.

## Working principles
Do not mutate protected seeds. Preserve unrelated dirty work; never reset/clean/stash/delete unrelated files. Do not accept dirty/memory-pressured evals as clean evidence. Do not call local proxy metrics final quality. Accept no candidate that fails Lane S, leaks memory, or drops NAX-compatible layers; 16-bit routed experts become acceptable once Track B's kernel makes them fast. Prefer deterministic KEEP commands over experimental archaeology. Make the repo feel like a tool, not a dig site.
