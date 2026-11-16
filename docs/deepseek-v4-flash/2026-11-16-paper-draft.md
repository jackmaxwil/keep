# Paper draft (rubric item 9), 2026-11-16

Working title (Jack, 2026-11-09): *Offload the Residual, Not the Expert: Unifying Pruning,
Quantization and Offloading in MoE Inference*.

**Title check before anything else.** The data no longer supports the first half of this title.
Offloading a 1-bit residual of resident 1-bit bases lost to plain 2-bit pruning at equal memory
(+0.025 KL, section 6.1), and the design that works offloads whole 2-bit experts; a resident residual on the hot set wins
in simulation (section 5.3) but is not the serving default (section 5.5). The subtitle holds. Candidates that
state what was measured, for Jack to choose from:

- *Pruning Is a Frozen Cache: Unifying Pruning, Quantization and Offloading in MoE Inference*
  (proposed in the plan, 2026-11-09; close to SwapMoE's comparison, see section 2).
- *Route, Don't Prune: A No-Wait Expert Cache Serves All 384 Experts of DeepSeek-V4.1-Flash on a
  128 GB Mac*.
- *Breadth First, Then Bits: Unifying Pruning, Quantization and Offloading in MoE Inference*.

Status of this draft: every number is from the recovery log or the build plan, with the section
named in brackets where it is not obvious. Sections that wait on unrun experiments say so. Not
for release without Jack's OK.

---

## Abstract

Mixture-of-experts models are pruned, quantized or offloaded to fit small machines, and each
choice is usually studied alone. We treat them as one policy over a pool of experts: how many
experts stay resident (pruning and offload breadth), when a demanded expert is admitted, at what
precision the hottest residents run, and which low-gate slots are skipped. Static pruning is the
corner of this policy where admission never happens. On DeepSeek-V4.1-Flash (40 MoE layers, 384
routed experts per layer, top-6), with every expert coded at about 2 bits on an E8 lattice, a
cache that admits experts by gate weight and re-routes a missed slot among the residents beats
the static set of equal memory at every size we measured, by 0.955 KL at 18 GB of resident
experts and 0.054 at 92 GB (22 sessions; position-bootstrap intervals clear of zero). A cache of 96 experts per
layer (44 GiB in all, estimated) is level with the original 192-expert pruned checkpoint at native
FP4 precision (0.605 against 0.578), which needs 341 GB on disk. Extra bits lose to extra breadth below about 200 residents and win
above it: at 92 GB of expert memory, 218 residents with the 64 hottest at 3 bits score 0.311,
against 0.398 for 250 at 2 bits and 0.390 for all 384 resident at 2 bits, which needs 139 GB.
Because a miss is re-routed instead of waited on, no token waits on a read and the cache's
bookkeeping costs about 7–14%: the full pool is served on a 128 GB M4 Max at job medians of
30–40 tokens per second with the model's drafter (22–41 per request), about five times the speed
of the exact pruned checkpoint under an existing waiting offload without its drafter, with no
measured loss in sampled code generation (21 of 24 against 18 of 24 correct). We also report two
negative results for learned admission: per-expert thresholds fitted on gate mass, and on a
per-expert miss cost measured through the model, both lose to a single threshold at 144
residents.

## 1. Introduction

A 128 GB Mac holds the non-expert weights of DeepSeek-V4.1-Flash and about 200 of its 384 routed
experts per layer at 2 bits. The usual answers are to prune the rest (fast, lossy everywhere the
pruned set does not cover), or to keep them on SSD and wait for them on a miss (exact, slow: an
existing MLX runtime, oMLX, reports 4.2–5.6 tokens per second for the full model this way on a
128 GB Mac). We ask what a fixed memory budget buys when pruning, quantization and offloading are
one policy instead of three separate decisions.

The policy has four settings per deployment: K residents per layer, an admission threshold tau
on the gate weight of a demanded expert, a hot set H of residents that run at one more bit, and a
tier-0 threshold below which a slot is skipped. A token routes among the experts resident when it
arrives. An expert it asked for above tau is read from SSD in the background and serves later
tokens. Nothing waits.

Contributions:

1. **One frontier for pruning, quantization and offload.** Static pruning is the tau = infinity
   corner of the policy. At equal resident memory the routed cache dominates it at every size
   from 18 to 92 GB, most where memory is small (section 5.1, figure `figures/frontier.svg`).
2. **Breadth before bits, with a measured crossover.** Below about 200 residents per layer a GB is
   worth more as residents than as precision on the hottest experts. Above it the order flips,
   and one extra bit on the 64 hottest residents does more than 32 more residents (section 5.3).
3. **A miss that costs no time, in a real runtime.** Re-routing is done by the router itself:
   evicted slots get a bias of minus infinity, and an expert is switched in only after its bytes
   land. On a 128 GB M4 Max this serves every expert of the model within 7–14% of the speed of a fixed
   set (section 5.5), and the drafter's verify step needs no change because it never waits either.
4. **Two negative results on learned admission** that bound what an admission policy can gain:
   0.064 KL at 144 residents in all, of which neither gate mass nor a measured miss cost recovers
   any (section 6.2).
5. **A selection result.** Ranking experts by total routed mass instead of REAP's per-route score
   lowers KL by 0.184 at 192 experts per layer (section 6.3).

## 2. Related work

- **Expert offloading with a wait.** oMLX (Apache-2.0) runs V4.1 on a 128 GB Mac with experts on
  SSD and waits on every miss (4.2–5.6 tok/s as reported; its drafter takes one of its runs from 6.6
  to 8.5, +29%, because verification also waits). ds4 (MIT) has a slab expert cache that waits. MoE-Infinity, Fate and AdapMoE
  prefetch to hide the wait.
- **A cheaper substitute on a miss.** HOBBIT (arXiv 2411.01433) runs a low-precision copy of a
  missed expert; DynaExq (2511.15015) keeps a hot set at high precision with a low-precision fallback and
  asynchronous promotion; DyMoE (2603.19172) sets mixed precision at run time by importance, by
  depth, with look-ahead prefetch. Cache-prior (2412.00099), SMoE (2508.18983) and MoE-CORE (2610.01950)
  re-route or substitute a miss using router scores; we use the simplest of these rules
  (top-6 among the residents, renormalized).
- **Dynamic resident sets against pruning.** SwapMoE (2308.15030) compares a dynamic resident set
  with static pruning at equal memory on small models. Our comparison is on a 384-expert model over
  the full pool, at 2 bits, with precision in the same policy.
- **Nested precision.** SliceMoE (2512.12990) uses nested bit slices; MatMoE (2611.01704) chooses
  nested code widths per expert from routing scores at run time. Our nested residual is an E8
  lattice code of the base's error.
- **Pruning.** REAP (2510.13999) scores experts per route and deliberately avoids frequency
  weighting; its 50% checkpoint of V4.1 (192 experts per layer) is our exact baseline.
- **Codes.** E8P is the E8-lattice codebook of QuIP#; llama.cpp's IQ2_XXS is an E8-lattice 2-bit
  format and has run V4.1 since 2026-10-25.

What we did not find in this survey (to be re-checked before submission): a miss that costs no
time in a Mac runtime at this scale, a frontier of routed against static at equal memory at V4
scale, and selection by total routed mass.

## 3. Method

### 3.1 Codes

Every routed expert (three projections, 2304 × 5120 and 5120 × 2304) is coded once with E8P at
2.04 bits per weight, 9.0 MB per expert, with importance-weighted fitting. The 192 experts
REAP-50 kept and the 192 it removed were fitted with the same search (fit error 0.090/0.074 and
0.091/0.079 for gate-up/down). For the hot tier, a 1-bit E8 residual of the 2-bit base was
fitted for the 192 survivors: weighted weight error falls from 0.084 to 0.030 (gate, up) and
0.047 to 0.024 (down), 4.6 MB more per expert. (The two error figures are different measures;
to be unified before submission.)

### 3.2 The policy

Per layer: K slots, warm-started with the K experts of most routed mass on calibration data. At
each token the router scores all 384 experts. The top 6 among residents run, renormalized (the
re-route). Each demanded expert outside the residents with gate at least tau is admitted after the
token, evicting the least recently demanded resident, and serves from the next token on (the
simulator assumes a fetch lands within one token; serving measures it). H of the residents, fixed
as the H survivors of most routed mass, run base + residual. Slots with gate below the tier-0
threshold are skipped without renormalizing. Tau = infinity freezes the warm set: that is static
pruning by routed mass, run through the same code path (in the pool of 192 this path reproduced
the earlier static 144 at 2 bits, 0.8123 against 0.812).

### 3.3 Serving without a wait

The runtime is a pinned, unmodified clone of oMLX with patches loaded from our repository.

- Residents are one stacked array of K rows per projection. Installs are positional reads of the
  code file straight into a slot's rows; the code files need no conversion.
- One small kernel takes two top-6's of the router's scores per token: among residents (all other
  biases are minus infinity), which runs, and over the pool, which is the demand the policy reads
  two calls later, so reading it waits for nothing.
- An evicted slot's bias is set so nothing routes to it; its rows are overwritten only after every
  call built before that has finished; its router row becomes the new expert only after the bytes
  are in. A token never waits on a read and never runs a half-written expert.
- Decode runs the lattice kernel for the routed half of the fused MoE, and a fused base + residual
  kernel for hot slots. The drafter's verify block goes through the same hook as a single token.
- The resident count is chosen from RAM at start-up from measured constants.

## 4. Setup

- **Model.** DeepSeek-V4.1-Flash-1017, 40 MoE layers, 384 routed experts, top-6, plus shared
  experts, Engram tables and a 3-stage drafter. The full model at native FP4/FP8 is the teacher.
- **Quality.** Teacher-forced KL against the full model's stored logits (written on a pod), on 22
  eval sessions from 11 domains (agentic Python and Rust, bash, business, business email, code
  diffs, cyber, English chat, English prose, networking, tool use), first 1,024 tokens each,
  domain mean. Calibration (routed mass, warm sets, every fitted code and threshold) used 24
  separate calibration sessions. Settings (tau, K, H, tier 0) were chosen on the eval sessions
  themselves (section 7). Differences are paired block bootstraps over
  positions (95% intervals) or paired session means with standard errors, as stated.
- **Generation.** One request at the model card's sampling (temperature 1.0, top_p 1.0, 4,096
  tokens, thinking off): write a script that finds duplicate files by SHA-256. Correct means it
  runs and lists the three duplicates of a test tree. 24 samples per configuration where stated.
- **Machine.** Apple M4 Max, 128 GB, MLX 0.32.4; 80 GB per process for simulation, 100 GB for
  serving.

## 5. Results

### 5.1 Routed against static at equal memory

Same 2-bit codes for every point, pool of 384, tau 0.3, re-route on a miss.

| Resident expert memory (residents) | Static KL | Routed KL | Routed − static, 95% interval |
|---:|---:|---:|---|
| 17.7 GB (48) | 1.746 | 0.790 | −0.955 [−1.014, −0.918] |
| 35.4 GB (96) | 1.129 | 0.605 | −0.524 [−0.559, −0.482] |
| 53.1 GB (144) | 0.763 | 0.454 | −0.309 [−0.332, −0.283] |
| 73.7 GB (200) | 0.543 | 0.401 | −0.142 [−0.157, −0.129] |
| 92.2 GB (250) | 0.452 | 0.398 | −0.054 [−0.062, −0.046] |
| 92.2 GB (218 + 64 hot) | 0.452 | 0.311 | −0.141 [−0.164, −0.117] |

- A cache of 48 at 28 GiB in all (estimated; 30 GiB measured in serving) is level with the
  static 144 at 60 GiB (0.790 against 0.763).
- A cache of 96 at 44 GiB (46 measured) is level with the exact REAP-50 checkpoint, 192 experts at native FP4
  (0.605 against 0.578).
- The same result holds inside the smaller pool of 192 (−0.056 at 144, −0.231 at 96), so it does
  not depend on the extra experts alone; the wider pool adds 0.296 at 144 residents and tau 0.2
  (0.756 to 0.460).
- Where the gain is: in the pool of 192, English chat and business move most (0.760 to 0.468 and
  0.971 to 0.763 at 144). The static set is chosen on the calibration mix, so the cache gains where
  a session differs from the mix.

### 5.2 How close the policy is to its ceiling

At 144 residents, tau 0.3: 0.454, against 0.424 for the best set per record (an oracle that sees
the whole record) and 0.390 with all 384 resident. In the pool of 192 at 96 residents the oracle gap
was entirely warm-up: 0.050 in the first 256 tokens, 0.002–0.017 after. Tau 0.2 and 0.3 are equal in KL
(0.460 and 0.454) and 0.3 reads a third as much; tau 0.1 is worse (admitting weak slots evicts
useful experts) and 0.45 loses 0.12 in every session. Section 6.2 tests whether a learned
admission rule closes the remaining 0.064.

### 5.3 Breadth against bits

Hot experts at native FP4 (the ceiling for extra bits) or at 3 bits (base + residual), against
2-bit residents at equal expert memory. Paired session differences, mean ± standard error,
without the two tool-use sessions (see section 7):

| Budget | Bits variant | Breadth variant | Difference |
|---|---|---|---|
| 53 GB | 127 residents, 16 FP4 | 144 at 2 bits | −0.010 ± 0.006 |
| 53 GB | 110 residents, 32 FP4 | 144 at 2 bits | +0.011 ± 0.006 |
| 65 GB | 144 residents, 32 FP4 | 177 at 2 bits | −0.015 ± 0.009 |
| 92 GB | 217 residents, 32 FP4 | 250 at 2 bits | −0.036 ± 0.010 |
| 92 GB | 218 residents, 64 at 3 bits | 250 at 2 bits | −0.039 ± 0.009 |

- At 53 GB, 16 FP4 experts are a small gain at best and 32 a loss; at 65 GB a draw; at 92 GB bits
  win. The crossover is near 200
  residents, which is the expert budget of a 128 GB Mac.
- With every expert resident, FP4 on the 32 hottest per layer (8% of experts, 22% of routed
  slots) removes 0.067 of the 0.235 left, in all 20 sessions.
- One more bit does what FP4 does at equal memory (64 at 3 bits against 32 at FP4: −0.004 ±
  0.003), and adds to the base the cache already holds instead of duplicating it.
- 218 + 64 hot at 92 GB is ahead of all 384 resident at 2 bits, which needs 139 GB (−0.020 ±
  0.008 without tool use).

### 5.4 Tier 0

Skipping slots below a gate threshold trades reads, not memory. At 144 residents, 0.05 skips
0.9% of slots for +0.008 [0.000, 0.015]; 0.1 skips 6% for +0.024; 0.2 skips 44%, cuts routed reads
by 43% and lands level with 96 residents. On the 92 GB configuration 0.05 and 0.1 cost 0.002 and
0.006 (intervals include zero). In the earlier nested ladder over 192 bases, 0.1 was free
(+0.001 on 7.7% of slots).

### 5.5 Serving

M4 Max, pool of 384, tau 0.3.

| Configuration | Footprint | Decode tok/s | Correct of 24 |
|---|---:|---|---:|
| Exact REAP-50 (192 FP4), oMLX offload that waits | | 5.6–6.3 | 18 |
| Cache, 144 residents, drafter off | 63 GiB | 22–27 | 17 |
| Cache, 200 residents, drafter off | 87 GiB | 21–28 | 19 |
| Cache, 180 residents, drafter on | 88 GiB | 22–41, median 32 | 21 |

- A fixed set on the fused path decodes seven times as fast as the waiting offload at equal
  memory (30–32 against 3.3–4.5 tok/s at 48 experts). Demand scoring costs 10–14%, and installs on the
  lattice cache a few percent more (26.0–27.9 against 28.6–30.1 tok/s for a fixed 144). More residents cost no speed.
- The drafter gains 9–43% on top (26–28% on scripts, 43% with thinking on, 9% on a 4,341-token
  prompt; oMLX's waiting offload gains 29% from it), with about 2.9 tokens kept per cycle of 3
  drafted.
- Installs in serving match the simulator (2.6 per token at 144 residents, tau 0.3); at 183
  residents with the drafter they fall to 0.2–0.75 per token, 1.5–5.5 MB read per token.
- Installed slots were spot-checked against their files (20 checkpoints over 12,785 installs for
  the FP4 cache, 17 over 2,431 for the lattice cache, and the drafter runs): every checked slot
  matched. A torn read is excluded by the ordering rule, not by a measurement.
- Under memory pressure (another process holding 40 GB of incompressible memory, 20 GB pushed to
  swap) the server's footprint stays at 83 GiB, it answers all 61 requests and decodes 9% slower (33.5
  against 36.7 tok/s). The footprint counts compressed and swapped pages, so whether any of the
  server's pages left RAM is not measured.
- Speed varies between jobs on an otherwise idle machine (the default ran at 39.6, 31.9 and 30.4
  tok/s median in three jobs); we compare within a job only.
- The hot tier in serving is 18–41% slower than the default with two kernel calls and 11% with
  the fused kernel, and is off by default: a hot route
  reads 1.5 times the bytes, and its residuals cost 33 residents of memory.

### 5.6 Generation

- At 24 samples the cache of 144 and the exact REAP-50 checkpoint are level (17 and 18 correct),
  and so are 180 residents with the drafter (21). The claim is "no worse, five times faster",
  not "better" (21 against 18 is Fisher p about 0.5).
- Below 120 residents generation breaks before KL suggests: 0, 2, 4 and 6 of 8 correct at 48, 96,
  120 and 144 residents. The failures at 96 are misspelled identifiers, from re-routed slots (25%
  of demand), not the sampling tail: neither temperature 0.6 nor top_p 0.95 rescues them. At 200
  residents (10% re-routed) none of the 5 failures is a misspelling.

## 6. What did not work

### 6.1 Offloading the residual

The first design kept a 1-bit base of each of REAP-50's 192 experts per layer resident and
offloaded a 1-bit residual on top, so a miss fell back to the base at no wait. Routing the
residuals worked (−0.068 to −0.073 against a fixed hot set of residuals at equal memory), but the
nesting penalty (+0.053 KL for base + residual against a flat 2-bit code) was larger than the
routed cache's gap to plain pruning: at equal memory it scored 0.832 and 0.837 against 0.812 for
144 experts at flat 2 bits (+0.020 [0.003, 0.045] and +0.025 [0.009, 0.051]). A static 1-bit tier was nearly worthless (it gained 0.025 KL at 192 units
and nothing at 288). Offloading whole 2-bit experts with re-route won instead.

### 6.2 Learned admission

Can a per-expert admission threshold, fitted on calibration data, close the 0.064 between the
cache at tau 0.3 and every expert resident?

- **Fit on gate mass.** One threshold per expert, minimizing missed strong gate mass
  (`max(gate − 0.25, 0)`) plus a price per install on stored calibration routes. 9% less missed
  mass at equal installs on held-out routes; in KL +0.022 and +0.026 against tau 0.3 (intervals
  clear of zero).
- **Fit on a measured miss cost.** Two runs under random per-expert thresholds on 52 calibration
  records, scored against all 384 resident, give a cost per layer and expert by ridge regression.
  It predicts held-out KL well (r 0.85 over records, against 0.44 for strong gate mass and 0.16
  for gate mass), and thresholds
  fitted on it miss 62% less of the priced mass. In KL +0.023 [0.015, 0.033].
- Neither failure is noise or a transfer problem: the measured cost fitted on one half of the
  records and priced on the other still favors the fit, and it does so on the eval sessions' own
  routes. The cost is additive over independent random misses but not for a policy chosen by it,
  which withholds the cheap experts together and misses 72% more strong gate mass. Open-loop
  prices do not rank fitted policies; a closed-loop search would cost a model run per candidate
  for at most 0.064.

Also tested on the replay proxy and dropped before any model run: a separate refresh threshold,
eviction by decayed routed mass, and a threshold that rises within a record.

### 6.3 Other results

- **Selection by total routed mass.** Ranking experts by summed routed mass instead of REAP's
  per-route score, at 192 of 384 experts, native precision, 54 eval sessions: 0.532 against
  0.716 (−0.184, session-level interval −0.277 to −0.128, better in 36 of 54 sessions, a count inflated by
  the session mix; English chat is not established as worse, +0.124, interval −0.050 to +0.206). The same rule chose the warm sets above.
- **Re-measuring routing after a cut** (prune twice): on the pod +0.014 at 192 (session-level interval 0.002 to 0.043) and
  +0.027 at 96 (no interval), with tables from 12 times less calibration data than the one-pass
  table; on the Mac, inside REAP-50 at 2 bits, a second pass helped by −0.024 [−0.034, −0.012].
  Not settled.
- **The page cache as the expert cache** is dead on Apple Silicon in its simple form: MLX can
  alias a memory-mapped file, but a GPU kernel that reads 1 MiB of a 2 GiB aliased buffer brings
  every page in.
- **A multi-row verify kernel** saves at most 12% of a draft cycle: distinct experts per route
  are 0.84 at 2 rows, 0.70 at 4, 0.63 at 6.

## 7. Limitations

- **Teacher-forced KL on short windows.** 22 sessions, first 1,024 tokens. A longer, untouched
  confirmation on 54 sessions with the routed forward on a GPU is planned (E7) and not run.
- **The tool-use teacher is off-format.** The eval text carries foreign tool markup, and on it
  the full model is worse than the pruned one on actual-token NLL; tool-use KL stays at 1.8–2.0
  for every variant at 144 residents and above, all 384 resident included. Results are given with and without that domain.
- **Settings chosen on the eval set.** Tau, K, H and the tier-0 threshold were picked on the same
  22 sessions the results are reported on. Codes and thresholds were fitted on calibration data
  only. E7 is the untouched confirmation.
- **Duplicated records.** Four agentic sessions share their first 1,024 tokens, so at this window
  one record counts in two domains. Paired differences are unaffected; domain means weight it
  twice.
- **One generation task.** One request type, 24 samples. The full model does not fit on the Mac,
  so there is no full-model generation reference.
- **Simulated fetch.** The simulator assumes a read lands within one token. Serving matches its
  install counts, but quality in serving is measured only through generation.
- **One model, one machine.** A second MoE (E8) and downstream benchmarks (E6) are not run.
  Footprint constants are this M4 Max's; other chips are unmeasured.

## 8. Conclusion

At a fixed memory budget, static pruning is a cache that never admits. Letting it admit, and
re-routing a miss instead of waiting, dominates pruning at every budget we measured on a
384-expert MoE, makes the whole pool servable on a 128 GB Mac within 7–14% of the speed of a fixed set, and
turns the precision question into a budget question: spend memory on breadth until about 200
residents per layer, then on bits for the hottest experts. What admission policy could still add
is small and was not learned from open-loop prices.

---

## Open before submission

| Item | Needed for | Blocked on |
|---|---|---|
| Title | everything | Jack (section at the top) |
| Model description (parameter counts, shared experts) against the model card | sections 1, 4 | a read of the card |
| E7: 54 untouched sessions, full length, routed forward on GPU | section 7, rubric item 4 | pod time |
| E8: a second MoE | rubric item 8 | download (Jack's go); 30 GB free on the Mac on 2026-11-16 |
| E6: downstream benchmarks at the card's settings | rubric item 8, section 5.6 | design; pod or the Mac |
| A teacher for tool use (on-format eval text, or NLL as the measure) | section 7 | eval set version 2 |
| Dedupe the four agentic sessions | section 7 | eval set version 2 |
| Prior-art search repeated for the three "not found" claims | section 2 | a search pass |
| Figures: frontier (exists), breadth against bits, resident count against KL and correctness | section 5 | plotting only |
