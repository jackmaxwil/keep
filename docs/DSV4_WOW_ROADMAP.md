# DSV4-Flash road to a publishable result

Date: 2026-08-21. Supersedes the "also queued, lower priority" list in
[HANDOFF.md](HANDOFF.md), whose Task 7 was measured to have no input (see
[the compounded verdict](DSV4_COMPOUNDED_SPEED_VERDICT.md)).

Goal: a community-shareable KEEP/RAMP result on `deepseek-ai/DeepSeek-V4-Flash-0731`
with figures that survive the first skeptical comment.

## Measured baseline

Everything here is committed evidence, not estimate.

| Fact | Value | Source |
|---|---|---|
| Parameters | 304.18 B | revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Source on disk | 163 GB, native FP4 (~4 bpw) | 48 shards |
| KEEP build | **84.687 GB @ 2.031 bpw** (1.925x) | 75.25 GB routed + 9.442 GB resident |
| Fits in 128 GB unified? | **KEEP yes, source no** (163 > 128) | machine is M5 Max / 128 GB |
| Top-1 teacher agreement | **91.80% median** (0.795157–0.957605) | 30 report sessions, 316,617 positions |
| Mean KL | 0.141742 median (0.085750–0.287086) | same |
| NLL delta | 0.157549 median | same |
| Harness self-check | source control scores top-1 `1.000000`, mean KL ~0 | same rows |
| MTP speculative | works, token-identical output; acceptance invalidated | Task 4 |
| **Prefill** throughput, streamed | VQ **25.1** tok/s vs source **60.95** tok/s | `dsv4-quality-gate` |
| **Decode** throughput, resident | **0.2409363565 tok/s** (4.15 s/token), clean row | `dsv4-headline-downfix-20260821` |
| Teacher cache | 153.94 GB, 1,502,378 supervised positions, **unused** | Task 5 |
| Holdout | 37 sessions, **sealed** | never opened |

## What sinks a post today

1. **Slower than the model it compresses.** 25.1 vs 60.95 tok/s streamed — but see
   the staleness warning below; that VQ figure was measured with the import bug in
   the loop. This is the first comment, and it ends the thread.
2. **91.80% top-1 is visible degradation.** ~8 tokens in 100 differ from teacher.
   Reads as lossy demo, not as a win.

The footprint claim on its own — 2x below an already-4-bit frontier model, fully
resident on one laptop, which the source cannot be — is genuinely first-of-kind.
It is not *extremely impressive* while carrying those two numbers.

## Gate figures

Absolute throughput targets set by the operator 2026-08-21: **50 tok/s is the bar,
100 tok/s is the wow.**

| Metric | Now | Bar | Wow |
|---|---|---|---|
| **decode tok/s** | **0.2409363565** (measured, clean) | **>= 50** | **>= 100** |
| prefill tok/s vs source | 0.41x pre-fix | >= 1.0x | >= 2x |
| Top-1 agreement | 91.80% (prefill, valid) | >= 95% | >= 97% |
| MTP acceptance | 57.89% (11/19), re-confirmed post-fix | >= 70% | >= 80% |
| Standard benchmarks | none | within 2 pts of source | within 1 pt |
| bpw | 2.031 | hold | hold |

> **The down-projection defect fixed in `e8df26cd` is latent in production, not
> active. An earlier revision of this document wrongly called the acceptance
> figure invalidated.** The defect is real at the `gather_vqmm` API level: with
> unsorted `sorted_rhs` the down projection was computed against permuted
> activations (cosine -0.01351581 against the reference). But the DSV4 router
> selects experts with `mx.argpartition(-biased, kth=top_k-1)[..., :top_k]`
> (`src/ramp/models/deepseek_v4_flash_adapter.py:892`), and argpartition returns
> the selected prefix in ascending **index** order — 200/200 draws in a direct
> test. So `argsort(rhs)` is always the identity in production, the skipped
> gather was a no-op, and nothing was ever wrong on this path.
>
> Confirmed empirically rather than argued: the post-fix run emits tokens
> byte-identical to Task 4's pre-fix run (all 16), and acceptance is exactly
> 11/19 = 0.5789473684210527 again. So the 57.89% figure stands, and Task 4's
> quality and acceptance evidence was valid all along.
>
> The fix is kept as defense in depth — any caller that passes unsorted routes
> would hit it, and `_stable_topk_indices` (`deepseek_v4_flash_adapter.py:1273`)
> is a second selection path with its own ordering. The regression test pins the
> contract at the API level where the defect is reachable.

### Measured memory headroom

From the post-fix bind proof (`~/keep-artifacts/dsv4-headline-downfix-20260821/bind-proof.json`),
1,271 residents bound, zero unbound VQ experts, wired-limit environment absent:

| Quantity | Bytes | GB |
|---|---|---|
| MLX active / peak | 90,990,767,620 | 90.99 |
| MLX cache | 3,556,781,384 | 3.56 |
| runtime parameter storage | 90,990,750,712 | 90.99 |
| **total in flight** | | **~94.5 of 128** |

`swapouts_delta` during the bind itself was 535,792. That is the headroom any
future footprint decision has to fit inside — notably the step 2 drafter rebind,
which would add 5.36 GB on top of the 90.99 GB figure.

> **PREFILL AND DECODE ARE NOT THE SAME NUMBER, and an earlier revision of this
> document conflated them.** The 25.1 / 60.95 tok/s pair is *prefill* — tokens
> consumed through a 1024-token chunked forward, where a single weight read
> amortizes across many positions. The 50/100 target is about tokens *generated*,
> one at a time, where every weight is re-read per token. On this model the two
> differ by roughly 100x. No decode measurement in this campaign ever approached
> 25 tok/s: Task 4's own accepted baselines were 16 tokens in 5.34-42.17 s, i.e.
> 0.38-3.0 tok/s.
>
> ### The decode ceiling is set by bytes, and residents are held at bf16 in memory
>
> **Residents are not fp8 at runtime.** The manifest's 9.441 GB counts fp8 *disk*
> bytes; the binder dequantizes to `mx.bfloat16`
> (`src/ramp/models/deepseek_v4_flash_adapter.py:2995`,
> `src/keep/convert/fp8_block.py:83-133`). Confirmed by arithmetic against the bind
> proof: runtime parameter storage 90.991 GB minus the 75.25 GB VQ artifact leaves
> **15.741 GB of residents in memory**, a 1.67x expansion over disk. Doubling only
> the `F8_E4M3` groups predicts 15.461 GB, which matches.
>
> Per-token decode memory traffic, corrected:
>
> | Component | disk GB | **in-memory GB / token** | Share |
> |---|---|---|---|
> | attention (`F8_E4M3` -> bf16) | 5.401 | **10.18** | 67% |
> | shared experts (`F8_E4M3` -> bf16) | 1.082 | 2.16 | 14% |
> | routed experts, 43 layers x top-6 x **3** projections @ 2.03125 bpw | 1.649 | 1.65 | 11% |
> | head (`[129280, 4096]` bf16; markov is a gather + draft-only) | 1.059 | 1.06 | 7% |
> | norms / router / hyper-connections | 0.244 | 0.23 | 2% |
> | **total** | | **15.27** | |
>
> Corrections folded in: the routed term is 43 x 6 x **3** x 2,129,920 B =
> 1,648,558,080 B; an earlier revision of this table multiplied two projections
> instead of three. The head term is 1.06 GB, not 1.192: `markov_w1` is bound as an
> `nn.Embedding` and gathered (~512 B/token) and `markov_w2` runs only on MTP draft
> cycles (`src/ramp/models/deepseek_v4_flash_adapter.py:1918-1932`, `:2489-2499`).
>
> ### The bandwidth number, and a same-machine reference point
>
> No memory-bandwidth microbenchmark exists in this repo, and the 500 GB/s figure in
> `docs/research/wave4-wide-m-vqmm-survey.md:55-56` is an assumption. The real
> anchor is `.repos/ds4/speed-bench/m5_max.csv`: a reference C/Metal engine running
> **this model on an M5 Max**.
>
> | ctx | prefill tps | gen tps | gen steady tps |
> |---|---|---|---|
> | 2048 | 790.18 | 39.35 | **40.00** |
> | 4096 | 710.50 | 37.50 | 38.09 |
>
> That engine holds residents at Q8_0 (~8.5 bpw, ~9.4 GB/token), so 40 tok/s implies
> **~376 GB/s achieved**, degrading roughly 30% by 64k context. Plan on **~370
> GB/s**; treat 500 as unproven upside.
>
> **40 tok/s AR is demonstrably achievable on this hardware.** That is the number to
> beat, and beating it with fewer bytes is a far stronger public claim than any
> abstract tok/s figure.
>
> ### What each target actually requires
>
> Measured output error on real resident data, f32-referenced matmul (bf16 baseline
> 0.21%): affine-8 g64 **0.81%**, affine-4 g64 **10.2%**, mxfp4 14.1%, mxfp8 **5.2%**.
> mxfp8 is rejected on measurement — it costs the same bytes as affine-8 for 6x the
> error, because it quantizes activations too.
>
> | scenario | GB/tok | AR tok/s @370 | x1.7701 MTP |
> |---|---|---|---|
> | S0 today (fp8 -> bf16 at bind) | 15.27 | **24** | 43 |
> | S1 fp8 groups -> affine-8 | 9.78 | **38** | 67 |
> | S2 S1 + head a8 + compressor/gate a8 + hc bf16 | 8.88 | **42** | 74 |
> | S3 attention core + shared -> affine-4, head a8 | 5.96 | **62** | **110** |
>
> - **100 tok/s AR is not reachable at any acceptable quality.** At 370 GB/s it needs
>   <= 3.7 GB/token; the routed experts alone are 1.65 GB and 4-bit-everything still
>   totals ~5.5 GB. It exists only as AR x MTP, and only from S3.
> - **50 tok/s AR needs S3** at 370 GB/s. S2 reaches it only if bandwidth is truly
>   >= 450.
> - **Best AR-only figure at defensible quality is S2, ~42 tok/s** — which lands
>   right on the reference engine's 40.00, a useful sign the physics is right.
>
> ### The mechanism already exists
>
> MLX affine quantization has a complete contract for dense weights:
> `nn.quantize(model, mode=, class_predicate=)` swaps `nn.Linear` ->
> `QuantizedLinear` and `nn.Embedding` -> `QuantizedEmbedding`, and even the
> `MultiLinear` used for `wo_a` has `to_quantized()` -> `QuantizedMultiLinear`.
> **Zero new kernels.** What is needed is quantize-at-bind (or a repacked artifact
> plus a manifest/identity extension in `src/mlx_vq/models/dsv4_composite_loader.py`)
> and a quality-gate rerun.
>
> By contrast the E8P VQ dense path (`src/mlx_vq/nn/linear.py:17`) exists but runs at
> the known 32-71 GB/s latency-bound rate, so byte cuts there would not convert to
> tok/s until that kernel gap closes.
>
> ### Recommended sequencing
>
> S1/S2 first — near-safe, gateable, ~1.7x cut, and quality-identical in the fp8 ->
> affine-8 step. Then per-layer-group affine-4 on `wq_b`/`wo_a`/`wo_b` (94% of
> attention-core bytes) and shared experts, with the teacher-agreement gate deciding
> how far down to go. Keep the head at affine-8 and no lower — it shapes every logit.
> Do not compress embed (a gather), norms, or the router gate, and keep the KV
> compressor at bf16 initially: DeepSeek shipped it bf16 while putting the rest of
> attention at fp8, which reads as a deliberate sensitivity signal.
>
> Honest risk: 4-bit attention across 43 layers at a spot-measured 10.2%
> per-projection perturbation is a real gamble at 91.80% top-1 needing >= 95%. The
> recovery machinery was built for routed-VQ damage and has never been validated on
> attention damage.
>
> ### One red flag checked and cleared
>
> A reported risk of `inf` weights from e8m0 scale bytes overflowing float32 in
> `dequantize_fp8_block` **does not reproduce**. All 390 `F8_E8M0` scale tensors in
> the resident artifact were scanned: the maximum scale byte is 121, implying 2^-6,
> against a float32 limit of 2^127. The reported 247-valued bytes were almost
> certainly `F8_E4M3` weight bytes read as scales.
>
> ### The measured rows are memory weather, not decode
>
> The quiet gate is **one-sided**: `wait_for_memory_quiet` and
> `collect_metric_snapshot` gate on `pageouts_delta` / `swapouts_delta` only
> (`src/mlx_vq/benchmark/metrics.py:103-112`, `:188-191`). `pageins_delta` is
> computed at `benchmarks/bench_dsv4_mtp_headline.py:882-886` but `_row_clean`
> (`:207-256`) never checks it, and swap-ins and compressor decompressions are never
> collected at all. A row can fault gigabytes of weights back in mid-window and
> still stamp itself clean.
>
> Byte-identical work therefore spans **16.5x** across rows:
>
> | Row | elapsed | pageins in window | gate |
> |---|---|---|---|
> | Task 4 baseline pair-03 | **5.58 s** | 144 | PASS |
> | Task 4 baseline pair-00 | 7.03 s | 1,119 | PASS |
> | Task 4 baselines pair-04/02/01 | 23.4 / 26.2 / 33.3 s | 1,941 / 553 / 7,865 | PASS |
> | post-fix pair-00-baseline | 66.41 s | 7,296 | PASS |
> | post-fix pair-01-baseline | 92.07 s | 245,255 (3.83 GB) | FAIL |
>
> The harness also manufactures the coldness it is meant to exclude: a warmup
> generate (`bench:846-849`) is followed by `gc`/`clear_cache` (`:850-853`) and then
> a **60-second idle sleep** (`:855-858`), giving macOS a full minute at 91 GB active
> to compress and evict the working set the warmup just heated. The bind itself
> recorded 535,792 swapouts (8.4 GB of anonymous MLX buffers); those pages return
> only when touched, during the timed window.
>
> **So the 0.2409 tok/s row is not a decode measurement.** It implies 3.7 GB/s
> against the corrected 15.501 GB/token — still ~100x below floor, so the
> implementation gap is real — but the same machine has run the identical workload
> in 5.58 s total, about 0.3 s per decode step. The software steady-state floor
> decomposes to roughly 270-390 ms/token: ~168 ms routed-expert kernels (43 x the
> measured 3.904 ms/FFN), ~40-80 ms resident reads, ~30-70 ms of sinkhorn (a
> 19-iteration loop on a [1,1,4,4] fp32 comb matrix,
> `src/ramp/models/deepseek_v4_flash_adapter.py:777-803`, called 86 times per
> token), and ~15-35 ms of descriptor and permutation bookkeeping.
>
> Attention was profiled for the first time and is **not** pathological: no Python
> loop over heads, no per-token fp8 dequantization, no materialized attention
> matrices, and the sparse indexer takes a trivial `arange` branch at these context
> lengths (`deepseek_v4_flash_adapter.py:1521-1620`). Its cost is purely its
> 10.802 GB/token of bf16 reads. Attention is a byte problem, not a kernel problem.
>
> ### The measurement cannot be fixed by gating — the footprint has to come down
>
> A full five-pair headline series was run post-fix
> (`artifacts/benchmarks/dsv4-headline-downfix-20260821/`) at 60-second quiet gates,
> both orders, on the authenticated composite. Result: **`clean_pairs: 0`,
> `dirty_pairs: 5`, `median_baseline_over_speculative: null`, verdict INCOMPLETE.**
>
> | pair | order | baseline tok/s | speculative tok/s | ratio | max pageins |
> |---|---|---|---|---|---|
> | 0 | baseline first | 0.2409 | 0.0804 | 0.3336 | 304,047 |
> | 1 | speculative first | 0.1738 | 0.7531 | 4.3339 | 245,255 |
> | 2 | baseline first | 0.3407 | 0.9622 | 2.8238 | 128,440 |
> | 3 | speculative first | 0.4457 | 1.0734 | 2.4085 | 98,439 |
> | 4 | baseline first | 0.4313 | 0.8766 | 2.0327 | 121,468 |
>
> Every pair was rejected by `_row_clean`, which requires the preflight quiet window
> *and* row `pageouts_delta` *and* row `swapouts_delta` all zero. None of these
> ratios is admissible evidence, and none should be quoted. What does survive: all
> ten rows report `acceptance_rate` 0.5789473684210527 with `acceptance_consistency`
> and `token_parity` true.
>
> The load-bearing conclusion: **zero of five pairs clear even the current
> too-lenient gate at 90.99 GB MLX active on a 128 GB machine.** Tightening the gate
> to also catch inbound paging — which it should, per the section above — yields
> *fewer* admissible rows, not more. Decode is therefore unmeasurable at this
> footprint by any gating strategy.
>
> That converges with the ceiling arithmetic from the opposite direction, and it
> reorders the work:
>
> | | today | fp8 groups -> affine-8 |
> |---|---|---|
> | per-token bytes | 15.27 GB | 9.78 GB |
> | AR ceiling @370 GB/s | 24 tok/s | **38 tok/s** |
> | MLX active | 90.99 GB | ~85 GB |
> | decode measurable? | **no, 0/5 pairs** | plausibly |
>
> **Resident compression now precedes the measurement fix.** It is the only change
> that both raises the ceiling and restores the headroom needed to measure anything.
> The measurement fix still lands — collect and gate on inbound counters, split
> prefill from decode timing, and report a steady-state median from repeated
> generates in one process rather than one row per freshly loaded process — but it
> produces no usable rows until the footprint drops.
>
> Implementation note for that fix, learned the hard way: `_row_clean`
> (`benchmarks/bench_dsv4_mtp_headline.py:216-225`) does exact set-equality on the
> quiet record's keys, so adding a field to `wait_for_memory_quiet` makes every row
> raise "quiet active record is malformed". Keep the change row-level — extend
> `collect_metric_snapshot`, bump the row `schema_version` 1 -> 2, and gate on the
> inbound deltas only for version 2 rows so the committed Task 4 evidence still
> replays under its original rule.

> **The 25.1 tok/s figure is stale as a VQ throughput claim.** It was measured before
> `390c51a4`, i.e. with `load_native()` re-globbing the build tree and re-executing
> the 3.5 MB native extension on every kernel call — which production hits **nine
> times per FFN per token**, because `glm4_moe_adapter.py:160` gates the shared
> route path on `indices.size >= 64` while M=1 top-6 supplies 6. Component-level
> measurement of the same fix moved the E8P projection ratio 4.53x. Full-model
> decode has not been re-measured since. Re-run
> `benchmarks/bench_dsv4_mtp_headline.py` on the resident composite before
> treating any tok/s number as current.

### What the 50/100 targets require

Working from the pre-fix streamed source baseline of 60.95 tok/s and the measured
Task 4 MTP median of 1.7701x on top of autoregressive decode:

| Path to target | Needed AR decode | Needed MTP |
|---|---|---|
| 50 tok/s via MTP median | ~28 tok/s | 1.7701x (have it) |
| 100 tok/s via MTP median | ~56 tok/s | 1.7701x (have it) |
| 100 tok/s via best observed pair | ~30 tok/s | 3.2973x (observed once, clean) |

So 50 tok/s is plausibly already reachable and 100 tok/s is a question of whether
post-fix AR decode clears roughly 56 tok/s, or whether MTP acceptance rises enough
(step 3, on the unused teacher cache) to carry a lower AR figure. Measure AR decode
first — it decides which lever matters.

## Ordered plan

### 1. Fix the E8P kernel — the whole campaign
**Step 1a is DONE. The 10.70x was never a kernel problem.** It was a Python
import bug in `src/mlx_vq/kernels/nax.py`: `load_native()` never registered
`sys.modules["_vqnax"]`, so every kernel call re-globbed the 78-file build tree
and re-executed the 3.5 MB native extension — three times per M=1 projection. A
4-line fix moved the ratio from **0.0934906005x to 0.4239291863x** (9/9 clean
rows, same committed rail, 9 fresh processes): 10.70x slower becomes 2.36x
slower, a 4.53x improvement. Recomposed, the campaign sits at 1.33x slower than
native source on the MTP median, and the best observed clean MTP pair already
composes to **1.40x faster than source**.

Remaining on this item, both found by the same investigation and neither requiring
any Metal to be written:

- **1b. Cache `_expert_block_tile_descriptors`** (`src/mlx_vq/ops/vq_switch.py:117-152`).
  It rebuilds a 257x256-element descriptor chain per projection to describe 6
  routes, costing about as much as the matmul it feeds (~0.67 ms of the pre-fix
  32.7 ms, a much larger share now). The repo already fixed this exact pattern
  twice elsewhere — `@lru_cache` on `_flat_route_descriptors` (`vq_switch.py:155-165`),
  whose docstring names the problem. Cheaper variant: `gate_proj` and `up_proj`
  get byte-identical descriptors, so build once per FFN and pass through the
  existing `tile_descriptors=` parameter (`vq_switch.py:638`).
- **1c. Reach the 13 unreachable E8P kernel variants.** Thirteen
  `nax_e8p_fp16_sorted_steel_*_matmul` entry points are compiled, bound, and
  callable in `src/mlx_vq/kernels/nax.py`, and none is reachable from production
  dispatch — including variants implementing the decode improvements the shader
  analysis identified. The winner can be found by benchmark.
- **1d. Production is worse than the rail.** `src/mlx_vq/models/glm4_moe_adapter.py:158`
  gates the shared-route path on `indices.size >= 64`; M=1 top-6 gives 6, so three
  independent chains run per FFN per token. Re-measure production, not just the rail.

The arithmetic that says parity is reachable: at M=1 the E8P path reads 38.3 MB
against native FP4's 80.2 MB — roughly *half* the bytes. A memory-bound E8P path
should be near parity or faster, so the remaining 2.36x is still implementation,
not physics.

### 2. Rebind the drafter to native FP4 — now questionable, reassess
**+5.3603 GB**, not +4.757 GB: the first figure dropped the 0.6040 GB of
`F8_E8M0` scales that must be resident for `mx.gather_qmm(mode="mxfp4")`. Steady
state goes to ~96.35 GB, and with the 3.56 GB MLX cache observed in the committed
bind proof that is ~99.9 GB of 128 GB — on a machine where the Task 4 bind already
logged 90,444 swapouts at the *lower* figure.

Step 1a changed the calculus: the drafter's E8P penalty is now 2.36x, not 10.70x,
so this trade buys much less than it did an hour ago while costing the same memory.
Do steps 1b-1d first and re-derive whether it is still worth 5.36 GB. Full scoping,
call sites, and the decode-and-compare test that catches the `w1`/`w3` swap hazard
are in [the verdict doc](DSV4_COMPOUNDED_SPEED_VERDICT.md).

### 3. MTP-head recovery training
The 153.94 GB / 1,502,378-position teacher cache was generated for exactly this
and has never been used. 57.9% -> 75%+ acceptance multiplies straight into
speculative throughput. Rails exist: `src/mlx_vq/recovery_campaign/`,
`src/mlx_vq/quality/glm52_distill_loss.py` (generalizes per the DeepSeek pivot).

### 4. Backbone quality recovery
Same cache, same rails. Target top-1 >= 95%, mean KL <= 0.06. This is what turns
"lossy demo" into "indistinguishable".

### 5. Standard-benchmark harness
There is none in the repo — only vendored tools under `.repos/`. Nobody outside
reads "mean KL 0.141742". MMLU / HumanEval / GSM8K / MBPP, source vs KEEP, same
machine, same quiet discipline. These are the numbers that get shared.

### 6. Quiet-window harness
The old Task 6. This machine shows 1.8–1.9x timing variance and every Task 4 row
proved it (baseline elapsed swung 5.34 s to 42.17 s across rows that both passed
the quiet gate). Do this before publishing **any** tok/s figure — not before the
work that produces one.

### 7. Holdout gate
37 sealed sessions, opened once, at the end, after steps 3 and 4 land.

### 8. Ship
Artifact + repro pack + writeup, carrying the composed-ratio honesty already
recorded rather than quoting the MTP median bare.

## Standing honesty constraint

Never publish the 1.7700938194x MTP median without its compressed-vs-source
qualifier. Composed with the measured component ratio it is 0.1655x of native
source — 6.04x slower. Full derivation in
[DSV4_COMPOUNDED_SPEED_VERDICT.md](DSV4_COMPOUNDED_SPEED_VERDICT.md).
