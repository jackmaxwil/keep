# Wave 4 increment 1 — wide-M VQ verify kernels

**Date:** 2026-08-11 · **Branch:** `worktree-agent-aa460ac30049b6cf4`
· **Machine:** Apple M5 Max, 40-core GPU, 128 GB, `applegpu_g17s`, MLX 0.31.2
· **Campaign:** `docs/deepseek-v4-flash/2026-08-11-campaign-plan.md`, Wave 4
· **Revision:** post-adversarial-review (2 blockers + 8 findings addressed)

## Status

**Delivered and parity-gated. The performance target was missed, and the miss is
the finding.** A wide-M-capable VQ verify kernel exists, is byte-exact against
the M=1 decode kernel looped per verify row across 193 test cases, and is wired
into dispatch behind an explicit `route_strategy="per_route_decoded"`. Measured
against a looped M=1 baseline it is **performance-neutral** — medians 0.98x /
1.03x / 0.98x at M=2/4/8 — not the ≥1.8x the Wave-4 brief targeted.

The increment's value is elsewhere: it found that **verify widths currently land
on a dispatch path 3.2–4.5x slower than looping the M=1 kernel**, and it measured
out three hypotheses about where the remaining 5–10x actually lives.

## Commits

| Commit | Contents |
| --- | --- |
| `2a846f54` | `docs/deepseek-v4-flash/research/wave4-wide-m-vqmm-survey.md` — survey + baseline table |
| `c2ce0229` | `src/mlx_vq/kernels/gather_vqmm_verify_mrows.metal`, host wrapper in `gather_vqmm.py`, ops seam in `vq_switch.py` |
| `05a86cdb` | `tests/test_gather_vqmm_verify_rows.py` — byte-exact parity gate |
| `7e29f45e` | `benchmarks/bench_gather_vqmm_verify_rows.py` |
| `61c58c73` | this report |
| review fixes | row-packing exactness guard, helper-sourced row packing, `per_route_decoded` rename, reproducible bandwidth columns, deterministic seeds, route cap, doc corrections |

Commits are ordered survey → kernel → harness → bench rather than the brief's
survey → harness → kernel → bench, so the test suite passes at every commit
rather than at three of four.

Files touched: `src/mlx_vq/kernels/`, `src/mlx_vq/ops/vq_switch.py`,
`src/mlx_vq/nn/switch_linear.py` (one validation tuple, review finding M3),
`tests/`, `benchmarks/`, `docs/`, and this report. No model weights were loaded.

## Parity contract satisfied

**Byte-exact, the strong contract.** Every verify variant produces bit-identical
output (compared on `tobytes()`, not a tolerance) to
`gather_vqmm_m1_kernel_unchecked(..., use_decoded_codebook=True)` at the **same**
row packing, run once per verify row. Exact because the verify kernel is a
literal row-tiling of the M=1 lane discipline: identical lane-to-codeword
partition, identical float32 accumulator, identical `simd_shuffle_down` tree, no
reassociation. Verified across M ∈ {1,2,3,4,8}, all three V4 shapes, four
routing patterns at the small shape and three at the V4 shapes, both groupings,
all `m_rows` tile widths, all accepted row packings, float16 and float32
activations, the per-route down-projection layout, and every `codeword_unroll`
setting.

Three exactness details worth recording, because each is a place a reasonable
implementer would assume otherwise:

- `simd_broadcast` of the group scale is a bit-copy, so reading `scales[...]`
  directly per lane is exact against the reference's broadcast.
- Replacing the reference's eight scalar activation loads with two aligned
  `half4` loads changes neither the values nor the dot, so it is exact.
- **Not every row packing is exact, and the unsafe one now raises.**
  `gather_vqmm_m1_decoded.metal` selects its reduction by lane count: `simd_sum`
  at exactly 32 lanes per row, a shuffle-down tree at 16 or fewer. The verify
  kernel implements only the tree, so `rows_per_threadgroup=8` (32 lanes) reduces
  in a different order — measured **46/49152 elements at gate/up and 74/98304 at
  down, max delta ~1e-3**. Accepted set is `{16, 32, 64}`
  (`VERIFY_ROWS_PER_THREADGROUP_VALUES`); 8 raises with a message naming
  `simd_sum`. 16 and 64 were verified exact at the real V4 shapes, not assumed.

**Not byte-exact, and stated as such.** A float32 dequantize-then-matmul
reference is checked with `rtol=atol=2e-2` plus a 0.9999 cosine floor, purely to
catch layout and indexing errors a self-consistent byte comparison cannot see. It
*cannot* be exact: the kernel evaluates each 8D codeword as `dot(half4, half4)`,
a half-precision 4-term dot, then scales in float32. The fallback contract the
brief offered ("per-element exactness of dequantized-weight matmul in float32
accumulate") was **not needed and is not claimed** — the strong contract holds.

Test results: `tests/test_gather_vqmm_verify_rows.py` 193 passed;
pre-existing `test_gather_vqmm.py`, `test_quantized_vq_switch_linear.py`,
`test_switch_routing.py`, `test_vq_switch_nax_e8p_selector.py` 59 passed.

## Measured results

GPU-amortized microseconds: `min` over 10 samples of 64 enqueued calls after 2
warmups (`--repeat 64 --iterations 21`; the amortized statistic takes
`max(5, iterations // 2)` samples). 256 experts, top-6, `group_size=512`.
Reproduce with:

```
UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python \
  benchmarks/bench_gather_vqmm_verify_rows.py \
  --verify-rows 2 4 8 --patterns uniform skewed duplicate \
  --repeat 64 --iterations 21 --include-current-dispatch
```

### Baseline: M=1 decode kernel called M times (uniform routing)

| Projection | K | N | M=2 | M=4 | M=8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate | 4096 | 2048 | 64.2 | 241.2 | 470.9 |
| up | 4096 | 2048 | 123.2 | 225.6 | 521.9 |
| down | 2048 | 4096 | 79.0 | 259.2 | 487.9 |

gate and up are the *same shape*; their 64.2 vs 123.2 µs at M=2 is a direct read
on the noise floor. Treat sub-1.5x differences here as unresolved.

### Verify variant vs that baseline (speedup medians over 3 shapes × 3 patterns)

| | M=2 | M=4 | M=8 |
| --- | ---: | ---: | ---: |
| verify flat, raw kernel | **0.98x** (0.89–1.27) | **1.03x** (0.97–1.10) | **0.98x** (0.92–1.04) |
| verify flat, op path | 0.96x | 0.95x | 0.96x |
| verify expert-grouped, op path | 0.42x | 0.66x | 0.75x |

Expert grouping at maximum overlap (every verify row selecting the same 6
experts): 0.42–0.55x at M=2, 0.90–1.00x at M=4, **1.25–1.50x at M=8**. Weight
reuse does work, but only when duplication is near-total and only at M=8.

### The finding that matters: current dispatch is the bottleneck

`gather_vqmm(route_strategy="auto")` at verify widths, versus the same baseline:

| | M=2 | M=4 | M=8 |
| --- | ---: | ---: | ---: |
| current auto dispatch | 0.10–0.36x | 0.14–0.26x | 0.15–0.24x |
| **verify op path vs current auto** | **3.24x** (2.96–9.73) | **4.32x** (3.69–6.27) | **4.53x** (4.01–6.27) |

Cause: with top-6 routing, `route_count >= 128` requires `tokens >= 22`, so
M=2–8 resolves to `"direct"`, and `"direct"` has only an M=1 fast path.
Everything else falls to `gather_vqmm.metal`, which spends one 256-thread
threadgroup per *output element per route*. The brief's premise — "verify falls
back to running M=1 kernels M times" — describes a fallback 3.2–4.5x *better*
than what actually happens today.

### Why the M dimension was the wrong lever

Median achieved code-read bandwidth (now emitted by the bench as `_code_gbps`,
computed from `code_bytes = distinct_experts × N × K/8`):

| | M=2 | M=4 | M=8 |
| --- | ---: | ---: | ---: |
| looped M=1 | 71 GB/s | 50 GB/s | 32 GB/s |
| verify flat | 70 GB/s | 55 GB/s | 32 GB/s |

Against roughly 500 GB/s available. Bandwidth *falls* as M rises, and an
operation count puts the kernel at ~15–20% of ALU throughput, so neither limit
binds — the inner loop is latency-bound on a code-byte load → dependent
threadgroup codebook lookup → dot → FMA chain.

Three hypotheses were implemented and measured rather than assumed:

- **Launch fusion** (one wide launch instead of M): neutral. MLX already batches
  the M `metal_kernel` calls of a looped baseline into one command buffer, so
  there was never M× launch overhead to reclaim.
- **Vectorized activation loads** (two aligned `half4` instead of eight scalars):
  negligible. Kept — free and byte-exact.
- **Codeword-walk unrolling** (2/4/8 codewords per lane iteration, to overlap the
  dependency chain): **slower at every setting on every shape**, degrading
  monotonically with depth. Register pressure from the live
  `codeword[U]`/`w0[U]`/`w1[U]` arrays costs more occupancy than the ILP
  recovers. Retained as `codeword_unroll` default 1, with a test pinning
  byte-exactness at every setting and another pinning the default, so the
  negative result is re-testable rather than folklore.

### Measurement-quality caveat

Identical configurations re-measured within one process differed by up to 1.8x.
Worse, **two full authoritative runs of the same committed code disagreed**: an
earlier run read 1.03/1.08/1.07 for the flat kernel where the final one reads
0.98/1.03/0.98. That is why this report calls the variant neutral rather than a
small win — the earlier, more flattering numbers did not survive re-measurement.
**No finer kernel tuning should be attempted on this machine without a
quiet-window harness first.**

## Review findings addressed

| ID | Fix |
| --- | --- |
| BLOCKER 1 | `rows_per_threadgroup` restricted to `VERIFY_ROWS_PER_THREADGROUP_VALUES = (16, 32, 64)`; 8 raises naming the `simd_sum` divergence. Reproduced the divergence first (46/49152 gate/up, 74/98304 down) — **but only at 8**; 16 and 64 measured byte-exact, contrary to the review's claim that all three diverge, so they are accepted and pinned by a new V4-shape parity sweep rather than rejected. Added `test_verify_rows_rejects_row_packings_that_break_exactness` and `test_verify_rows_byte_exact_across_row_packings_at_v4_shapes` (the small fixture provably cannot expose this class — 2 codewords per group over 8 lanes). |
| BLOCKER 2 | All row-packing call sites now source from `m1_rows_per_threadgroup(input_dims, output_dims)`: the op default (`rows_per_threadgroup="auto"`), the `per_route_decoded` dispatch branch, the test reference `_looped_m1_reference`, and the benchmark baseline. `test_verify_row_packing_defaults_to_the_m1_decode_helper` asserts the wiring and that the helper's value stays inside the accepted set. |
| M1 | Seeds derive from `zlib.crc32` of a canonical string via `_seed(...)`; no `hash()` on str anywhere. |
| M2 | `_verify_row_tile_descriptors` raises above `_VERIFY_MAX_ROUTES = 256`, pointing the caller at `sorted_tiled`; pinned by test. |
| M3 | `switch_linear.py` accepts `per_route_decoded` with a comment explaining that the M=1 fast-path guards deliberately exclude it, and that this is the strategy Wave 5 should select. |
| M4 | Sampling described correctly everywhere: min over `max(5, iterations // 2)` samples after `--amortized-warmup` (default 2) warmups. Added the `--amortized-warmup` flag so it is not a magic constant. |
| M5 | Bench emits `code_bytes`, `bytes_moved`, `_gbps`, `_code_gbps` per variant. All bandwidth claims are now derived by committed code; survey and report both carry the reproduction command. |
| L1 | `test_measured_negative_result_defaults_stay_pinned` asserts `codeword_unroll` default 1, `rows_per_threadgroup` default 32, op `grouping` default `"flat"`, and — by spying on `_verify_rows_routed` — that the dispatch branch passes `grouping="flat"`, `m_rows=1`, and the helper's row packing. |
| L2 | `wide_m` → `per_route_decoded` across ops, `switch_linear`, tests, bench, and docs. The name now states what it does: one route per tile, `m_rows=1`, no M-widening. The kernel remains wide-M capable via `gather_vqmm_verify_rows(grouping="expert")`. Test and bench files renamed to `*_verify_rows.py`. |
| L6 | Kernel count corrected to fifteen (fourteen pre-existing + one). Routing-pattern coverage stated precisely per suite. This report's file-touch line now names `switch_linear.py`. |

All headline numbers were re-measured against the final post-review code; the
tables above are from that run, not from the pre-review run.

## Recommended next increment

Ranked by measured value, not by plan order.

1. **Flip `route_strategy="auto"` to `per_route_decoded` for 2 ≤ tokens ≤ 21 with
   `code_bits=8`.** Worth **3.2–4.5x** on verify projections. Pure dispatch
   change, byte-exact, no new kernel. Held out of this increment only because it
   changes an existing shipped path and deserves its own regression pass. This is
   the highest-value change available to Wave 4 and should land before any
   further kernel work.
2. **Build the quiet-window benchmark harness** (Lane-S discipline: fresh
   process, lock held, pageout/swapout gates) and re-baseline. Everything below
   is unmeasurable until this exists — this increment's own headline number moved
   0.05–0.09x between two runs of identical code.
3. **Attack the lane discipline, not the M dimension.** The one untried lever
   with a real mechanism is **widening the per-lane code read** — `uint` or
   `uint2` instead of one byte, so each lane owns 4–8 consecutive codewords and
   the dependency chain amortizes over more work. This reassociates the per-lane
   codeword walk, so it moves the byte-exact reference: change the M=1 decode
   kernel in lockstep (preferred — decode benefits equally and the reference
   moves with it) or declare a second contract explicitly. Upside is the 7–15x
   gap to hardware bandwidth, which dwarfs everything else here.
4. **Do not build the NAX M=16 verify path.** The hardware gate passes
   (`applegpu_g17s`, Darwin 25.4.0) and the plumbing exists in
   `src/mlx_vq/kernels/nax.py`, but its 16-row cooperative-tensor tile is
   mismatched to 12–48 routes over 12–46 distinct experts, it is fp16 throughout
   so cannot be byte-exact, and omlx's own measurement (prefill 828 → 400 tok/s
   when custom block kernels displaced NAX on M5 Max) plus its 1024-route floor
   both say NAX is a prefill tool. Revisit only if Wave 5 measures much wider
   effective verify passes.
5. **Revive expert grouping only on evidence.** It needs two things Wave 5 can
   supply or refute: an in-kernel descriptor builder on the omlx pattern (one
   threadgroup, binary search per expert, atomic slot allocation) to remove the
   ~90–120 µs host-graph descriptor cost that currently exceeds the kernel time,
   and a *measured* mean same-expert route depth from real MTP draft routing to
   choose `m_rows` from — rather than from the verify width, which this increment
   shows is the wrong proxy.

## Open question for the campaign owner

Wave 4's gate is "parity byte-exact on all shapes; microbenchmark table
committed" — both satisfied. But the wave's *purpose* was the ≥1.8–2.5x that
makes the compounding thesis work, and that is not in the M dimension: fusing M
launches is neutral, and widening M is a loss at verify widths.

The compounding argument survives via item 1 — 3.2–4.5x from dispatch alone,
more than the wave asked for — but the attribution changes materially: **verify
does not need wide-M kernels, it needs to stop using the wrong kernel.**
Recommend restating the Wave-4 gate around item 1's dispatch win plus item 3's
bandwidth target before Wave 5 integration depends on a number from this wave.
