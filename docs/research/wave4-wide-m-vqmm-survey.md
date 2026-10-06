# Wave 4 survey: wide-M gather-VQMM kernels for MTP verify

**Date:** 2026-08-11 · **Campaign:** `keep-v4flash-compounding-campaign-20260811.md`, Wave 4
· **Machine:** Apple M5 Max, 40-core GPU, 128 GB, `applegpu_g17s`, MLX 0.31.2, Darwin 25.4.0

MTP verify pushes M=2–8 rows through the same expert-gathered VQ projections
decode pushes at M=1. This survey documents what the existing `gather_vqmm`
kernel family does, where a wide-M variant plugs into its dispatch, and what the
M=1-looped baseline actually costs on the DeepSeek-V4-Flash routed-expert shapes.

Target shapes throughout: 256 routed experts per layer, top-6 routing,
gate/up (`w1`/`w3`) K=4096 → N=2048, down (`w2`) K=2048 → N=4096, E8 8-bit codes,
`group_size=512`. All measurements use synthetic weights and codebooks; no model
weights are involved.

Every number below, including the GB/s figures the next-increment ranking rests
on, is reproduced by:

```
UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python \
  benchmarks/bench_gather_vqmm_verify_rows.py \
  --verify-rows 2 4 8 --patterns uniform skewed duplicate \
  --repeat 64 --iterations 21 --include-current-dispatch
```

Each emitted row carries `code_bytes`, `bytes_moved`, and per-variant `_gbps` /
`_code_gbps`, computed from the shapes, dtypes, and distinct-expert count, so the
bandwidth claims are derived by committed code rather than asserted here. Parity
claims are reproduced by
`uv run --group dev python -m pytest tests/test_gather_vqmm_verify_rows.py -q`.

## 1. Headline findings

1. **The campaign's stated Wave-4 premise is optimistic about today's code.** The
   plan assumes verify "falls back to running M=1 kernels M times". It does not:
   `gather_vqmm` at 2 ≤ tokens ≤ 21 with `route_strategy="auto"` selects
   `gather_vqmm.metal`, a scalar kernel that spends one 256-thread threadgroup
   per *output element per route*. That path is **3.2–4.5x slower** (median by M;
   range 3.0–9.7x) than simply looping the M=1 decode kernel M times. The first
   real Wave-4 win is a dispatch decision, not a new kernel.
2. **Fusing the M launches into one launch is performance-neutral, nowhere near
   the ≥1.8x the plan targets.** MLX already batches the M `metal_kernel` calls
   of a looped M=1 baseline into a single command buffer, so there is no
   per-launch overhead left to reclaim. Measured medians against the looped
   baseline: **0.98x (M=2), 1.03x (M=4), 0.98x (M=8)** — indistinguishable from
   1.0 given this machine's noise floor. Byte-exact, but not faster.
3. **Expert-grouped route tiles (the mtplx weight-reuse discipline) lose at
   verify widths.** With top-6 of 256 experts, M ≤ 8 verify rows rarely share an
   expert, so `M_ROWS > 1` tiles run mostly one-route-deep and pay register
   pressure for nothing: median 0.42x/0.66x/0.75x at M=2/4/8. It only wins when
   routing is heavily duplicated (1.25–1.50x at M=8 when all rows pick the same 6
   experts). Its on-device descriptor build costs ~90–120 µs, more than the whole
   kernel.
4. **The real headroom is not in the M dimension.** Both baseline and verify
   kernel sit at **32–71 GB/s** of code-read bandwidth (median by M: 71 at M=2,
   50 at M=4, 32 at M=8) on a machine with roughly 500 GB/s,
   and neither ALU nor bandwidth is saturated. The lane discipline is
   latency-bound. Two ILP fixes were tried and measured: vectorizing the
   activation load (2 aligned `half4` loads instead of 8 scalar loads) gained
   almost nothing, and unrolling the codeword walk 2/4/8-deep made things
   *worse* at every setting on every shape — register pressure costs more
   occupancy than the ILP recovers. Both stay byte-exact, so the exactness
   reasoning is sound; the bottleneck is simply elsewhere.
5. **Measurement noise on this machine exceeds the effect size.** Identical
   configurations re-measured within one process differ by up to 1.8x, and two
   full authoritative runs of the same committed code disagreed by up to 0.1x on
   the flat-kernel ratio (an earlier run read 1.03/1.08/1.07 where the final one
   reads 0.98/1.03/0.98). This is precisely why item 2 is reported as neutral
   rather than as a small win. Every number below is a `min`-over-repeats
   statistic for GPU time and a median for latency, and the next increment
   should not attempt finer kernel tuning without a quiet-window harness
   (Lane-S discipline).

## 2. The existing kernel family

`src/mlx_vq/kernels/gather_vqmm.py` registers fifteen `mx.fast.metal_kernel`
objects (fourteen pre-existing plus this increment's verify kernel). Two
families matter for verify.

### 2.1 The M=1 decode family

All are launched `grid=(256, row_tiles, routes)`, `threadgroup=(256, 1, 1)`:
one 256-thread threadgroup per (output-row tile × route), where
`row_tiles = ceil(output_dims / ROWS_PER_TG)`.

| Kernel | File | Lane layout | Notes |
| --- | --- | --- | --- |
| `mlx_vq_gather_vqmm_m1` | `gather_vqmm_m1.metal` | `ROWS_PER_TG` rows × `LANES_PER_ROW` lanes | Generic; `CODE_BITS`, threadgroup-vs-device codebook, decoded-vs-packed all template-gated |
| `mlx_vq_gather_vqmm_m1_decoded` | `gather_vqmm_m1_decoded.metal` | same | **The production decode path.** Stages a 512-entry `half4` decoded codebook in threadgroup memory (4 KB), `scale` broadcast from lane 0 via `simd_broadcast` |
| `mlx_vq_gather_vqmm_m1_rowpair` | `gather_vqmm_m1_rowpair.metal` | 16 row-pairs × 16 lanes | Two output rows per lane group, amortizing the *activation* load across N. Selected when `input_dims > output_dims` |
| `mlx_vq_gather_vqmm_m1_per_route_decoded` | `gather_vqmm_m1_per_route_decoded.metal` | `ROWS_PER_TG` × `LANES_PER_ROW` | Per-route activation layout for the down projection |

Shape-keyed selection lives in three one-line helpers
(`gather_vqmm.py:126-141`), currently constant:
`m1_rows_per_threadgroup → 32`, `m1_use_threadgroup_codebook → False`,
`m1_use_decoded_codebook → True`. So production decode is
`ROWS_PER_TG=32`, `LANES_PER_ROW=8`, decoded codebook. **The verify path and its
parity gate both source the row packing from `m1_rows_per_threadgroup` rather
than from a literal 32**, so if that helper is ever tuned the two move together
instead of silently diverging bit-for-bit while a literal-pinned gate passes.

One row packing is *not* exactly reproducible and is rejected rather than
approximated. `gather_vqmm_m1_decoded.metal` chooses its reduction by lane
count: `simd_sum` at exactly `LANES_PER_ROW == 32`, a `simd_shuffle_down` tree at
16 or fewer, threadgroup partials above 32. The verify kernel implements only the
shuffle-down tree, so `ROWS_PER_TG=8` (which yields 32 lanes per row) reduces in
a different order. Measured divergence at the V4 shapes: **46/49152 elements
(gate/up) and 74/98304 (down), max delta ~1e-3.** `ROWS_PER_TG` 16, 32, and 64
are byte-exact and are the accepted set
(`VERIFY_ROWS_PER_THREADGROUP_VALUES`); 8 raises. Note the small test fixture
cannot detect this class of bug at all — 2 codewords per scale group spread over
8 lanes leaves most lanes contributing nothing — which is why the row-packing
parity sweep runs at the real V4 shapes.

**Inner loop of `gather_vqmm_m1_decoded.metal`,** which is the shape the wide-M
variant generalizes. Per output row `row` and expert `expert`:

- `code_base = (expert * out_dim + row) * codewords`, `codewords = K/8`.
- For each scale group, lane `row_lane` walks codewords
  `group_start + row_lane, +LANES_PER_ROW, …`.
- Per codeword: load one `uint8` code byte, two `half4` reads from the
  threadgroup decoded codebook, eight scalar activation loads, two
  `dot(half4, half4)` (a **half-precision** 4-term dot), two `float` FMAs into a
  `float` accumulator.
- Reduction: `simd_shuffle_down` tree over `LANES_PER_ROW`, lane 0 writes
  `out[route * out_dim + row]`.

Expert gather is fed by `rhs_indices[route]` read directly in the kernel — there
is no gathered weight tensor and no staging of codes in threadgroup memory. Only
the codebook is staged.

**Measured baseline — the M=1 decode kernel called M times** (deliverable 1's
required table). GPU-amortized µs, `min` over 10 samples of 64 enqueued calls
after 2 warmups (`--repeat 64 --iterations 21`; the amortized statistic takes
`max(5, iterations // 2)` samples),
uniform routing over 256 experts, top-6:

| Projection | K | N | M=2 | M=4 | M=8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate | 4096 | 2048 | 64.2 | 241.2 | 470.9 |
| up | 4096 | 2048 | 123.2 | 225.6 | 521.9 |
| down | 2048 | 4096 | 79.0 | 259.2 | 487.9 |

Single-shot latency medians (one call per `mx.eval`, so ~0.2 ms of MLX dispatch
is included): M=2 ~0.30 ms, M=4 ~0.43 ms, M=8 ~0.70 ms.

gate and up are the same shape, so the spread between their rows at equal M
(64.2 vs 123.2 µs at M=2) is a direct read on this machine's noise floor; treat
sub-1.5x differences in this table as unresolved.

Note the sub-linear scaling into M=2: at M=1 only
`top_k × row_tiles = 6 × 64 = 384` threadgroups are launched, which does not fill
40 cores, so the first extra verify rows are nearly free. Achieved code-read
bandwidth *falls* with M — median 71 GB/s at M=2, 50 at M=4, 32 at M=8 —
which is the clearest single indicator that this lane discipline is
latency-limited rather than bandwidth-limited.

### 2.2 The simdgroup-MMA block family

Six kernels, all taking the descriptor triple
`(tile_experts, tile_offsets, tile_counts)` plus `lhs_indices`, launched
`grid=(threads, row_tiles, tile_count)` with `row_tiles = ceil(output_dims/32)`.
Routes are pre-sorted by expert; each tile is a contiguous run of routes sharing
one expert.

| Kernel | Threads | Route tile | Row tile | K per barrier | Fragment |
| --- | ---: | ---: | ---: | ---: | --- |
| `mma4x4_blocks` | 512 (16 simdgroups) | 32 | 32 | 8 | `simdgroup_half8x8` × `simdgroup_float8x8` acc |
| `mma8x4_blocks` | 1024 (32 simdgroups) | 64 | 32 | 8 | same |
| `mma_cwdecode_blocks` | `route_groups × 128` | 32 or 64 | 32 | 8 | one full 8D codeword decoded per lane |
| `mma_k32_cwdecode_blocks` | — | 32 or 64 | 32 | 32 | staged K tile |
| `mma_k64_cwdecode_blocks` | — | 32 or 64 | 32 or 8 rows×8 | 64 | `row_groups` selectable |
| `mma_k128_cwdecode_blocks` | 512 | 32 | 32 | 128 | deepest K staging |

Staging in `mma4x4_blocks`: `threadgroup half a_tile[256]` (32 routes × 8 K),
`half b_tile[256]` (32 rows × 8 K of dequantized weights), `float c_tile[1024]`,
`uint local_codebook[256]`. Simdgroup `id` splits into
`route_group = id & 3` and `row_group = id >> 2`, i.e. a 32×32 output tile from
4×4 8×8 fragments.

**Why this family cannot serve verify well as-is:** its M dimension is *routes*,
and a 32-route tile is 4–16x wider than a verify pass has routes per expert.
More fundamentally its accumulation is `simdgroup_multiply_accumulate` over
`half` operands (both `a_tile` and `b_tile` are `half`), so it can never be
byte-exact against the M=1 float32 path — the repo's own parity tests for these
kernels use `max abs < 2e-4` plus a cosine floor rather than exactness
(`tests/test_gather_vqmm.py:1035-1039`). Extending it to M=2/4 row tiles would
therefore forfeit the parity contract while also wasting 6/8 of every A
fragment. This survey's conclusion is that the correct wide-M kernel at verify
widths is a row-tiled generalization of the **M=1 SIMD-reduction discipline**,
not a narrowed MMA block kernel — and the measurements in §1 bear that out.

## 3. Dispatch seams

### 3.1 Where verify traffic goes today

`ops/vq_switch.py::gather_vqmm` is the contract entry point: `x` is
`[tokens, hidden]`, `rhs_indices` is `[tokens, top_k]`, result is
`[tokens, top_k, out]`. Strategy resolution:

```
route_count = tokens * top_k
selected = "sorted_tiled" if (sorted_indices or route_count >= 128) else "direct"
```

Then, for `implementation="metal"` and `selected == "direct"`:

- `tokens == 1 and code_bits == 8` → `gather_vqmm_m1_kernel` (the decode path).
- **otherwise → `gather_vqmm_kernel`** (`gather_vqmm.metal`,
  `grid=(256, output_dims, tokens*top_k)`).

With top-6 routing, `route_count >= 128` needs `tokens >= 22`. So **every verify
width the campaign cares about (M=2–8, 12–48 routes) lands on
`gather_vqmm_kernel`** — one threadgroup of 256 threads per output element,
2048–4096 × 48 threadgroups at M=8. That is the 3.2–4.5x regression in §1.1.

Threshold constants (`vq_switch.py:29-35`) for reference:
`_GATE_UP_MMA_ROUTE_THRESHOLD = 8192`, `_DOWN_BLOCK_MMA_ROUTE_THRESHOLD = 128`,
`_LARGE_BLOCK_MMA_ROUTE_THRESHOLD = 16384`, tiles 32/64.

### 3.2 Where the wide-M variant plugs in

Three seams, in increasing order of blast radius:

1. **Kernel layer** — `gather_vqmm.py` now registers
   `_GATHER_VQMM_VERIFY_MROWS_KERNEL` alongside the existing fourteen, with host
   wrapper `gather_vqmm_verify_mrows_kernel(...)` and the shape-keyed helper
   `verify_m_rows(route_count, top_k)`, mirroring the existing
   `m1_rows_per_threadgroup` convention. It consumes the *same* descriptor
   triple as the MMA block family, so flat and expert-grouped dispatch are one
   kernel with different descriptors.
2. **Ops layer** — `vq_switch.py` adds `gather_vqmm_verify_rows(...)` as the
   public verify entry point (token layout or per-route layout via
   `lhs_indices`), plus `_verify_row_tile_descriptors` (expert-grouped tiles,
   bounded by route count rather than `ceil(routes/tile) + num_experts`, which
   matters because the 256-expert term would otherwise inflate the tile grid
   ~10x at verify sizes) and cached `_flat_route_descriptors` /
   `_token_route_lhs`.
3. **Strategy layer** — `route_strategy="per_route_decoded"` is added to
   `gather_vqmm` and to `QuantizedVQSwitchLinear`. The name states what it does:
   it routes every route through the per-route decoded kernel, one route per
   tile, `m_rows=1` — it does **not** widen M, because §1.3 measures widening as
   a loss at verify widths. The kernel remains wide-M capable via
   `gather_vqmm_verify_rows(grouping="expert")`.
   **`"auto"` is deliberately unchanged in this increment.** Flipping `auto` to
   route 2 ≤ tokens ≤ 21 through the wide-M path is the single highest-value
   remaining change (worth 3.2–4.5x per §1.1) but it alters the behaviour of an
   existing shipped dispatch and belongs in its own increment with its own
   regression pass.

## 4. Reference architecture notes

Studied for imitation, per the Wave-4 brief.

**dflash-mlx `verify_qmm.py`** — a cascade, not a table: eligibility
(`bits ∈ {4,8}`, `group_size ∈ {32,64,128}`, `M ∈ {4,16}` *exactly*) → M-specific
family → shape moduli → magnitude thresholds → arch gate last. Two ideas worth
taking: (a) K compiled in as a template constant (`KCONST`) so every loop bound
is `constexpr`, with the kernel cache keyed on K; (b) variant selection done
**once at layer construction** with both dtype pipelines prebuilt, so the
per-call path is a two-predicate branch. Its `_auto_variant(K, N)` is
`("mma2big_pipe", 8) if K >= 8192 or N <= 8192 else ("mma2big", 1)` — both V4
shapes have N ≤ 8192, so the analogous choice here would be K-split with 8 parts.
`_m4_ksplit_np` splits K across simdgroups (2 parts if N ≥ 4096 else 4) and
combines in threadgroup memory with a **sequential ascending fp32 loop**, i.e.
deterministic by construction; `_m16_ktmpl`'s NAX variant gates on
`arch.startswith("applegpu_g17")` and macOS ≥ 26.2 and uses
`matmul2d_descriptor(16, 32, 16, …)` cooperative tensors with a 16/32/16 tile.

**mtplx `deepseek_v4_attn_proj_wide_m3.py`** — the wide-M weight-reuse lane
discipline this increment's kernel copies: 32 lanes partition **K** (not M, not
N), each simdgroup owns 4 output rows, accumulators are `float result[M][4]`,
and **one packed weight word plus its scale and bias are hoisted above the M
loop** and consumed by all M rows before the kernel advances, with the
activation re-read per row into a single 8-element staging buffer. Its exactness
technique is also directly relevant: it reproduces MLX's affine-Q4 association
operation-for-operation so a load-time gate can assert `mx.array_equal` against
stock rather than a tolerance. Note it hard-wires M=3 and leaves M ∈ {2,4} on
the stock path.

**omlx `patches/deepseek_v4/switch_layers.py`** — crossovers keyed on
`num_routes = tokens × top_k`, three stacked: **< 64 routes → no sort, stock
gather-mv**; ≥ 1024 routes on NAX → route back to stock `mx.gather_qmm` (which
dispatches `gather_qmm_rhs_nax`); ≥ 8192 (affine) / 16384 (mxfp4) → BM 16 → 32.
The 64-route floor is independent corroboration of this survey's finding: below
it, sorting routes by expert does not pay for itself, and verify at M ≤ 8 with
top-6 is 12–48 routes — under the floor. Its block-list builder is one
threadgroup, one thread per expert, two binary searches over the sorted indices
and an atomic slot counter; that is the pattern to copy if the expert-grouped
descriptor build is ever revived (see §5).

## 5. NAX M=16 feasibility (not built this increment)

The hardware gate passes: `mx.device_info()["architecture"]` is
`applegpu_g17s`, which satisfies both dflash's `applegpu_g17*` prefix test and
omlx's stricter `applegpu_g(\d+)([a-z])` with `gen >= 17`, on Darwin 25.4.0
(macOS 26.x, ≥ 26.2 required). `src/mlx_vq/kernels/nax.py` already exposes
`nax_e8_fp16_sorted_steel_matmul` and `nax_e8p_fp16_sorted_steel_matmul`, wired
into `gather_vqmm_sorted_routes` under `implementation="nax_e8"/"nax_e8p"`, so
the plumbing exists.

Feasibility assessment, given this increment's measurements:

- A NAX M=16 verify path is **shape-mismatched to verify at M ≤ 8**. The
  cooperative-tensor tile is 16 rows; a verify pass supplies 12–48 *routes*
  spread across 12–46 *distinct experts*, so the 16-row A operand would be
  mostly padding unless routes are expert-grouped — which §1.3 measures as a
  loss at these widths.
- It also cannot satisfy the byte-exact contract: the NAX path is fp16
  throughout (`sorted_x = x_mx.astype(mx.float16)`), like the MMA block family.
- omlx's own note is a direct warning: on NAX the *custom* block kernels
  regressed prefill from 828 to 400 tok/s versus stock, which is why
  `_nax_prefers_stock` routes ≥ 1024 routes back to `mx.gather_qmm`. Verify
  route counts are two orders of magnitude below that threshold.

**Recommendation: do not build the NAX M=16 verify path.** It is the right tool
for prefill-sized route counts, not verify-sized ones. If Wave 5 measures MTP
verify at much larger effective widths (deep trees, many candidates), revisit.

## 6. What to do next

Ranked by measured value:

1. **Route 2 ≤ tokens ≤ 21 away from `gather_vqmm_kernel`** — flip `auto` to
   `per_route_decoded` for `code_bits=8`. Worth **3.2–4.5x** on verify
   projections (median by M; range 3.0–9.7x). Pure
   dispatch change, byte-exact, no new kernel.
2. **Build a quiet-window benchmark harness** before any further kernel tuning.
   Current run-to-run spread (up to 1.8x on identical configs) exceeds the
   effect size of every remaining micro-optimization.
3. **Attack the latency-bound lane discipline, not the M dimension.** Achieved
   32–71 GB/s against ~500 GB/s available. ILP unrolling and activation
   vectorization are both measured dead ends. The untried lever with a real
   mechanism is **widening the per-lane code read** (`uint`/`uint2` instead of
   one byte, so a lane owns 4–8 consecutive codewords). That reassociates the
   per-lane codeword walk, so it changes the byte-exact reference — it would need
   the M=1 kernel changed in lockstep, or a second declared contract.
4. **Only if Wave 5 measures high route duplication:** revive expert grouping
   with an in-kernel descriptor builder on the omlx pattern (one threadgroup,
   binary search per expert, atomic slot allocation) to kill the ~90–120 µs
   host-graph descriptor cost, and pick `m_rows` from *measured* mean
   same-expert route depth rather than from the verify width.
