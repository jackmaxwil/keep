# Routed precision: implementation plan to 100%

Written 2026-11-09. Target headline (Jack, 2026-11-09, after the prior-art search below): *Offload
the Residual, Not the Expert: Unifying Pruning, Quantization and Offloading in MoE Inference*. It
replaces *Precision as a Routed Dimension*, whose first half MatMoE already covers. Progress is scored with the fixed rubric in the
[recovery log](2026-11-08-v41-reap-recovery-log.md) ("Headline progress rubric"); this plan says
what has to be built, run and measured for each rubric item, in what order, and what counts as
done. Score today: **26 / 100**.

## 1. Prior art found today, and what it leaves us

A search on 2026-11-09 found work that already covers parts of the idea. This changes what the
paper can claim, so it comes first.

| Paper | What it does | Overlap with us |
|---|---|---|
| MatMoE (arXiv 2611.01704, 1 Nov 2026, ICCAD 2026) | Nested ("Matryoshka") integer codes per expert; picks each expert's bit-width at run time from its routing score; llama.cpp engines on CPU and GPU | Per-token precision from routing scores with nested codes. This is the phrase "precision as a routed dimension". All weights stay resident; no offload, no pruning. |
| DynaExq (arXiv 2511.15015) | Hot experts resident at high precision, low-precision fallback for the rest, promotions and demotions applied asynchronously so the forward pass never waits | Our no-stall cache with async admission. Separate copies per precision, not nested. GPU memory, not SSD. |
| DyMoE (arXiv 2603.19172) | Run-time mixed precision by expert importance, depth-adaptive schedule, look-ahead prefetch | Depth-dependent importance (our "two routing regimes" observation), prefetch. |
| HOBBIT (arXiv 2411.01433) | On a cache miss, run a low-precision copy of a less critical expert instead of waiting for the load | Low tier as the miss path. Separate copies. |
| MC-MoE (arXiv 2410.06270) | Static mixed-precision per expert plus online per-token expert pruning | Per-token tier 0, but with static precision. |
| PMQ, GEMQ (ICML 2026), EdgeMoE | Static per-expert bit allocation | Our step 2c allocator. |

Only abstracts were read for all but MatMoE (introduction and method overview). Step W0 below reads
them in full.

What none of them appears to do, and what the paper therefore has to stand on:

1. **The offloaded unit is the nested residual.** Tiers span unified memory and SSD inside one
   code: the base is always resident, and a promotion reads only the residual (4.6 MB), never a
   second full copy. What that buys depends on how a separate-copy cache (DynaExq, HOBBIT) is run,
   and both readings are measured (corrected 2026-11-09; the first draft claimed only the first):
   - if it keeps the low copy of a hot expert, a hot expert costs 2 extra units and a nested design
     holds twice as many hot experts at equal memory;
   - if it drops the low copy and reads it back on demotion, memory is equal, but every
     promote/demote cycle moves 3 units (2 in, 1 back) where a residual moves 1, and a demotion is
     free. At an equal SSD rate the nested cache can admit three times as often.
   The separate 2-bit copy is slightly more accurate than base + residual (error 0.090 against
   0.101), so the nested design has to win on memory or on traffic, not on per-expert accuracy.
2. **Tier 0 in the same policy.** Expert pruning (static tier 0), per-token slot dropping, precision
   and residency all come from one rule with one trade-off parameter.
3. **The sub-2-bit lattice regime** (1 and 2 bits per weight, E8 codes), on a 384-expert, top-6
   frontier model, on a 128 GB laptop.

The new title leads with claim 1 and keeps "unifying pruning, quantization and offloading", the
half of the original headline that survives. The paper must cite and beat
MatMoE-style and DynaExq-style policies rather than only static baselines. If W0 finds a paper that
already offloads nested residuals under per-token tiers, the paper becomes an empirical study of
that design at 2 bits and below, and Jack decides whether to continue.

## 2. Where things stand

Done: per-token tiers in `local_eval.py` (`--dynamic`, `--tau`, `--tau-drop`), the residual cache
simulation (`residual_cache`, `--cache-k`, `--cache-warm`), the cost report (`routed_report.py`),
unit tests, and the static frontier to beat (192 units 1.014, 288 units 0.809, 384 units 0.747 KL).

Running on the Mac (detached, scripts in `~/Models/v41-reap-run/jobs/`): the nested residual fit
(`quant/e8-nest-r1`, 8 of 40 layers), the step-3 sidecar fit (restarted from layer 0), and a chain
that runs the sidecar eval and then the first routed sweep (8 variants) when the fits finish.

One unit = one expert at 1 bit per weight = 4.6 MB per layer. All Mac results are inside REAP-50
(192 experts per layer), scored by KL against the full model's logits on 22 held-out sessions.

## 3. Workstreams

Each workstream lists the deliverable, how it is built, the acceptance test, and the fallback.
Times are estimates; the Mac runs one heavy CPU job and one GPU fit at a time, under the 80 GB cap.

### W0. Related work, read in full (gates the claims; no rubric points)

- Read MatMoE, DynaExq, DyMoE, HOBBIT, MC-MoE and GEMQ end to end. Write
  `2026-11-09-routed-precision-related-work.md`: one row per paper with nested or not, per-token or
  not, what is offloaded, pruning, bit range, models, hardware.
- Done when the three claims in section 1 are each confirmed or struck. Half a day, in parallel
  with the fits.

### W1. Nested codes validated (rubric item 1: 5 → 10)

- Finish `quant/e8-nest-r1` (32 layers left, about 6.6 minutes each).
- Accept when (a) every layer's base + residual error is within 1.15× of the flat 2-bit fit, and
  (b) the sweep's endpoints land where they should: all slots at tier 2 scores KL ≤ 0.80 (flat
  2-bit is 0.747), all at tier 1 reproduces flat 1-bit (1.358).
- If (b) fails: one alternating round (refit the base against `W − residual`, then the residual
  again), about 7 h of GPU per round.
- Stretch, for the frontier figure: a third rung (`--residual-of` taking two directories), about
  3 bits per weight, 4.4 h and 34 GB of disk.

### W4. Core result (rubric item 4: 0 → 25)

The claim: at equal resident memory, routed tiers with residual offload give lower KL than any
static assignment and than a separate-copy cache, at an SSD read rate the Mac sustains.

Budgets: B = 240, 288 and 336 units per layer (44, 53 and 62 GB of routed experts). A configuration
keeps N expert bases resident and K residual slots, N + K = B. Policies, all in the same harness on
the same codes:

| Policy | What it is | How it runs |
|---|---|---|
| S, static-nested | hot set = top K by pruned-model routed mass, never changes | `--cache-k K --tau 1e9` with `--cache-warm` |
| R, routed-nested (ours) | LRU of K residuals, async admission at gate ≥ τ | `--cache-k K --tau τ` |
| D, separate-copy cache (DynaExq / HOBBIT style) | high tier is a second flat 2-bit copy. Scored two ways from one run: keeping the low copy (N + 2K units, 2 units per fetch) or swapping it (N + K units, 3 units per cycle) | `--dynamic BASE\|HI`; `routed_report.py` prints both |
| M, all-resident routed (MatMoE style) | every residual resident, tier by routing score | `--tau τ`, no cache; a latency baseline at 384 units |
| O, per-session oracle | hot set = top K by that record's own routed mass; upper bound for any cache | new: `--cache-policy oracle` |

Build first (half a day): the separate-copy mode, the oracle policy, expert masks combined with
`--dynamic` (N < 192), a paired bootstrap over sessions in `routed_report.py --compare A B`, and an
admission trace output for W6. Move the routed forward and `residual_cache` into a shared
`routed.py` so the pod driver can use them.

Runs (each pass is 8 variants, about 3.5 h on the CPU):

1. Pass 1 (chained): N = 192; endpoints, S(96), R(96, τ 0.2 and 0.3), R(48, τ 0.2), M(τ 0.22),
   per-token drop.
2. Pass 2 (queued): O(96), R(96, τ 0.15), N = 144 all at tier 2, N = 160 with K = 128 routed and
   static, S(48), and the two projection-level rungs (X5a).
3. Pass 2b (queued), separate copies: K = 96 at τ 0.2, 0.3 and 0.4 and static; K = 48 and 144.
4. Pass 3 (queued), budgets 240 and 336: R, S and O at K = 144; R(48, τ 0.3), O(48); N = 160 with
   K = 80 routed and static.
5. Pass 4: the winners and their baselines at 4,096 tokens per record (the pod's teacher logits
   cover whole records), so the cache is measured past its warm-up.

Acceptance and scoring inside the item:

- 10 points: R beats S by at least 0.03 KL, and beats D at equal memory and equal SSD rate (under
  whichever of D's two accountings is better for D), bootstrap 95% interval excluding zero, at one
  budget, with simulated SSD traffic at or below 2.5 GB/s at 30 tok/s (half the measured sequential
  rate).
- 15 points: the same at two of three budgets, confirmed at 4,096 tokens.
- 25 points: reproduced on the full 384-expert model on a pod (W4b).

W4b, pod (only after 15 points locally): fit nested codes for all 384 experts and run the same
policies with the full eval set (108 sessions). Needs a torch/CUDA backend for the 1-bit E8 search
(256 codewords, a plain argmin) and the routed forward in `collect.py`. All prep on the Mac; the
code pack uploads while the model downloads. Estimate 6–10 h on 1×H100 at $3.99/h, $25–40 plus
storage; H200 at $5.29/h if H100 stock is still out. Budget left is about $380 of $497.

If R does not beat S, in this order:

1. Admission by prediction instead of after use: score layer l+1's gate on layer l's output and
   fetch ahead (the Fate result, 97% hit rate). Measure the hit rate offline first.
2. A stronger ladder: 2-bit base plus 1-bit residual (3 bits), budgets 384–576. The 1-bit base is
   weak (error 0.31); with a 2-bit base the question becomes how close to FP4 (0.578) the routed
   residual gets.
3. A long-horizon hot set (DynaExq's rule) over nested residuals, which isolates the nesting gain
   from the policy.
4. If all three fail, the honest paper is the negative result plus the selection finding from
   step 2c, under a different title. Say so then; do not stretch the claim.

### W5. Unification (rubric item 5: 0 → 10)

- One rule for all three decisions. For a slot with gate weight g on expert e in layer l, the
  estimated benefit of tier t over t − 1 is g² · (ε_{t−1} − ε_t) · ‖f_e‖², with ε from the fit
  metrics (ε_0 = 1) and ‖f_e‖ from the routing capture; the cost is the bytes that tier adds
  (resident or read). A slot takes the highest tier whose benefit per byte is at least λ. One λ
  yields per-(layer, expert) thresholds for dropping and for the residual, and the same test on
  expected benefit decides which bases and residuals are resident. New `routed_policy.py` writes
  the thresholds; `local_eval.py --tau-file` reads them.
- `routed_frontier.py`: KL against resident GB for static REAP + KEEP, static-nested, separate-copy
  and routed, with SSD rate marked; figure and table committed.
- If the sidecar eval shows at least 0.05 KL, add sidecars as an always-resident layer tier (about
  1.1 units per layer).
- Accept when at least 12 measured configurations are on the figure and the one-λ policy is within
  0.02 KL of the best hand-tuned setting at each budget.

### W6. Real Apple Silicon streaming (rubric item 6: 0 → 10)

Trace-driven, since no V4.1 decoder exists on this Mac.

- `residual_cache` writes its admission trace (record, token, layer, expert).
- `residual_stream_bench.py` replays the trace at 10, 30 and 60 tok/s against the real residual
  bytes on the internal SSD: `pread` with `F_NOCACHE` into page-aligned buffers from a small thread
  pool, reading each expert's extent straight out of the uncompressed npz (no second copy on
  disk). Also an `mmap` + `madvise` variant.
- Measures: latency per admission, the share that lands before the next token and before that
  expert is next routed, sustained GB/s, CPU use, `phys_footprint`, thermal state, bytes read per
  1,000 tokens, and the copy cost into an MLX array.
- Accept when, at the winning W4 configuration, at least 99% of admissions complete within one
  token interval at 30 tok/s and the sustained rate is at most half the device's sequential rate.
  Run with no other heavy job, because the fits and evals also read from this SSD.

### W7. Learned policy and tier-aware codes (rubric item 7: 0 → 10)

- (a) 5 points: the calibrated per-(layer, expert) thresholds from W5 beat a single global τ by at
  least 0.02 KL at equal memory and SSD rate.
- (b) 5 points: tier-aware codes. Capture the importance diagonal separately for low-gate and
  high-gate slots, fit the base to the low-gate statistics and the residual to the high-gate ones
  (each code is fit to the tokens that use it), and gain at least 0.02 KL. Four capture chunks,
  about 7 h of GPU for the refit, one eval pass.
- Not required for 100%: end-to-end distillation of thresholds and router biases with a bits
  penalty. It needs backprop through the streamed model on a pod (over $100) and is worth doing
  only if (a) and (b) leave a visible gap.

### W8. Generality (rubric item 8: 0 → 10)

- 5 points, second MoE: Qwen3-30B-A3B (specs to verify at download: 48 layers, 128 experts, top 8,
  Apache-2.0, about 61 GB in BF16). It fits the Mac for the teacher pass and, unlike V4.1, can
  generate locally. Build its pack with the same domains, capture routing and teacher logits, fit
  nested E8 codes, port the routed forward to its MoE module, and repeat the S / R / D comparison
  at three budgets.
- 5 points, downstream: on Qwen3-30B, two generative benchmarks inside the scope (coding and tool
  use), chosen after a currency check so that nothing saturated is used, at least 200 items each,
  static against routed at equal memory. On V4.1, likelihood-ranked tasks that fit the streamed
  teacher-forced pass (ranking the correct patch or tool call among distractors) built from
  held-out sources.
- Needs about 75 GB of disk and Jack's go for the download.

### W9. Paper (rubric item 9: 1 → 5)

- `docs/paper/` in arXiv LaTeX: method (tiers, nested lattice codes, the one-λ rule, the no-stall
  cache), results (frontier, SSD traffic against τ and K, depth regimes, per-domain), related-work
  table from W0, limitations (teacher-forced KL, one eval length, the tool-use anomaly).
- Sections are drafted as each workstream lands; the full draft follows W8. Publishing the nested
  codes on Hugging Face needs Jack's OK.

## 4. Order and expected score

| When (estimate) | Work | Score after |
|---|---|---:|
| Today | W1 finishes; sidecar eval; sweep pass 1; W0 reading; W4 harness additions | 31 |
| Day 1 | W4 passes 2–3; W6 bench built and run between passes | 41–46 |
| Day 2 | W4 pass 4 (4,096 tokens); W5 policy and figure; W7a | 61–66 |
| Day 3 | W6 final numbers; W7b capture, refit, eval; W4b prep | 76 |
| Day 4 | W4b on a pod; W8 download and teacher pass | 86 |
| Days 5–6 | W8 fit, routed eval, downstream benchmarks | 96 |
| Day 7 | W9 full draft | 100 |

The score moves only on measured results, so a failed acceptance test holds it where it is.

## 5. Resources

- Memory: 80 GB cap. The 8-variant routed sweep is estimated at about 65 GB; if the guard trips it
  splits into two passes of four.
- Disk: 74 GB free after storing 1-bit codes as uint8. The nested fit takes 27 GB more. W8 does not
  fit without freeing space.
- Pod: about $380 left. W4b is the only planned spend ($25–40).

## 6. Decisions for Jack

1. Disk: Jack is cleaning up himself (2026-11-09). Safe to remove: `~/Models/v41-reap75` (273 GB;
   published on Hugging Face, and its experts are a subset of the local REAP-50) and
   `~/Models/v41-reap-run/quant/e8p-prunedimp` (65 GB; step 2 found it changes nothing). In use, do
   not remove: `~/Models/v41-reap50` and, under `~/Models/v41-reap-run/`, `stats`, `pack`,
   `imatrix`, `alloc`, `evals`, `sidecar`, `jobs`, `quant/e8p-podimp`, `quant/e8-1bit-podimp` and
   `quant/e8-nest-r1`.
2. Download Qwen3-30B-A3B for W8.
3. Whether the prior art in section 1 changes the title or the ambition.

## 7. Candidate levers beyond the prior art (2026-11-09)

From this campaign's results and a survey of the older KEEP documents. None is checked against the
literature yet; each gets a search before it is claimed. Ordered by expected value.

| # | Lever | Evidence in hand | Test | Where it runs |
|---|---|---|---|---|
| X1 | Prune twice at 384 → 192. REAP-50 was chosen on full-model routing with the per-route score; its KL (0.578) is the floor under everything here. | The same fix cut KL 15% at 192 → 96 (step 2c ablation); a second pass saturates (step 2d). | Calibration pass on the full model masked to 288 by total mass, re-rank to 192, eval against shipped REAP-50. | Pod, with W4b; about $15–25 extra |
| X2 | ~~Calibration-targeted pruning as specialization.~~ Dropped 2026-11-09. | The NLL advantage sits only where the teacher is surprised by off-format text embedded in the records; on teacher-confident positions the student is worse in every domain. | Done locally from the pod's stored outputs; see the findings note, section 3. | — |
| X3 | Keep every expert, offload residuals, instead of pruning half. | V4.1 plan: each expert kept costs 0.359 GB of RAM at 2 bits "but adds no per-token reads", and the fit "forces a ~50% prune". All 384 one-bit bases cost the same memory as REAP-50 at 2 bits. | 384 bases + K residuals against REAP-50 at 2 bits (0.747) at equal resident memory. This is the title's claim at full scale. | Pod, W4b; the Mac sweep at 192 vs 96 is its proxy |
| X4 | The OS page cache as the precision cache. Residuals are memory-mapped clean file pages; `mincore` asks whether one is resident, `madvise(WILLNEED)` requests it in the background, and the kernel evicts under pressure at no write cost. Precision degrades instead of the machine swapping. | Roadmap, 2026-08-21: at 90.99 GB of MLX-active memory byte-identical work spanned 16.5× in time and the bind alone logged 535,792 swapouts. Anonymous memory at that footprint thrashes; clean file pages cannot. | In W6: replay the trace through `mmap` + `mincore`, with and without a memory hog, and measure hits, latency and swapouts. | Mac |
| X5 | A stronger base and finer rungs. The 1-bit base is the weak link. | Pilot: 1-bit codes gave a 0.56 block cosine; the SwiGLU amplifies projection error about 2.8×; 2 bits is "at its rate-distortion limit". Here, a static 1-bit tier added 0.025 KL at best. | (a) Projection-level tiers from the existing codes: residual for the down projection only (1.33 bits) or gate/up only (1.67 bits), no new fit. (b) The 2-bit base + 1-bit residual ladder, moved up from fallback to W4 pass 3. | Mac |
| X6 | Budget by sensitivity, not routed mass. | Step 1: mass-based reallocation lost to uniform (1.046 vs 0.955 at 128) although layers 20–39 are far more concentrated than 0–19. | Mask a fixed share of experts in one layer band at a time and measure KL; allocate by measured sensitivity. Two eval passes. | Mac, after W4 |
| X7 | A shared base across similar experts (base = one matrix per cluster, residual per expert). | None; speculative. | Functional similarity of experts on shared inputs for 3 layers; stop if experts are near-orthogonal. | Mac, about 1 h |

Smaller findings worth a paragraph in the paper or a note of their own:

- Expert precision is a memory lever, not a speed lever, on this family: on V4-Flash, routed experts
  were 11% of the 15.27 GB read per decoded token and attention 67% (roadmap, 2026-08-21). MatMoE's
  latency argument for dynamic precision does not carry over; the memory argument does. The
  `tok_s_bound` column of `routed_report.py` counts expert bytes only and is not a speed estimate.
- Incoherence processing (the random rotation QuIP# relies on) made importance-aware E8P fits 15%
  worse on the down projection, because it flattens a per-column importance that spans 180×
  (wave-5 pilot).
- A gate/up-swapped artifact loads, gives finite logits and scores cosine 0.9993; only a
  discrimination ratio caught it. Cosine is not a correctness check.
- Negative priors for the old recovery steps 4–5: on GLM-4.5-Air router distillation was rejected
  (top-1 change slightly negative) and a rank-4 sidecar improved top-1 from 0.795 to 0.852 but ran
  1.72× slower and failed the speed gate. Sidecars here are judged on quality per unit of memory and
  on their per-token cost, and router distillation stays off the critical path.
- A quality reference from outside: the Rapid-MLX V4.1 build is reported at KL 0.26 and top-1 85% on
  a 256 GB M3 Ultra (local test plan, 2026-11-08). Our best 2-bit REAP-50 number is 0.747. A result
  that impresses has to close most of that gap at about half the memory.
