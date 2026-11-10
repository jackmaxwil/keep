# Routed precision: implementation plan to 100% (revision 2)

First written 2026-11-09; revised the same evening after the first routed sweep, the X1 pod run
and the sidecar result. Revision 1 is in git history (`fd3d57e`). Target headline (Jack,
2026-11-09): *Offload the Residual, Not the Expert: Unifying Pruning, Quantization and Offloading in
MoE Inference*. Progress is scored with the fixed rubric in the
[recovery log](2026-11-08-v41-reap-recovery-log.md) ("Headline progress rubric"); the rubric and its
weights do not change with this revision. Score today: **31 / 100**.

Evidence and reasoning for everything below:
[2026-11-09-keep-findings-and-levers.md](2026-11-09-keep-findings-and-levers.md), sections 5 and 6.

## 1. What changed, in four lines

1. **The base model moves.** Ranking experts by total routed mass beats the shipped REAP-50 by
   0.184 KL at 192 experts on the pod's eval. Every Mac number so far sits on REAP-50. All further
   work runs on the new 192 (`fm-192`).
2. **The ladder moves up one bit.** The 1-bit base is not a usable model (KL 1.358 on its own), so a
   cache miss is expensive, and nesting on top of it costs 0.053 KL. The main ladder becomes a
   2-bit base with a 1-bit residual (a third bit). The 1-bit base is cut altogether (Jack,
   2026-11-09): at equal memory 96 experts at 2 bits (1.039) beat 192 at 1 bit (1.358), so it
   loses at small budgets too. The 1-bit code survives only as the residual. The pass already
   running (pass 2) finishes; passes 2b and 3 on the old ladder are cancelled.
3. **Routing the cache is confirmed; "residual beats copy" is not.** A routed cache beats a fixed
   hot set by 0.068 KL at equal memory and 0.28 GB/s of reads. SSD traffic was not the binding
   constraint at that point, which removes the residual's main advantage over a separate copy
   there. The separate-copy sweep (tonight) and the 2 + 1 ladder decide the title.
4. **Three levers are closed:** sidecars (+0.199 KL end to end), pruning twice (no gain over one
   pass at 192, none at 96), and the tool-use NLL effect (an artifact).

## 2. Prior art (unchanged from revision 1)

| Paper | What it does | Overlap with us |
|---|---|---|
| MatMoE (arXiv 2611.01704, ICCAD 2026) | Nested integer codes per expert; bit-width per expert at run time from its routing score | Per-token precision with nested codes. All weights resident; no offload, no pruning. |
| DynaExq (arXiv 2511.15015) | Hot experts resident at high precision, low-precision fallback, asynchronous promotion | Our no-stall cache. Separate copies. GPU memory, not SSD. |
| DyMoE (arXiv 2603.19172) | Run-time mixed precision by importance, depth-adaptive, look-ahead prefetch | Depth regimes, prefetch. |
| HOBBIT (arXiv 2411.01433) | On a miss, run a low-precision copy instead of waiting | Low tier as the miss path. Separate copies. |
| MC-MoE (arXiv 2410.06270) | Static mixed precision plus online per-token expert pruning | Per-token tier 0 with static precision. |
| PMQ, GEMQ (ICML 2026), EdgeMoE | Static per-expert bit allocation | Our step 2c allocator. |

Only abstracts were read for all but MatMoE. W0 reads them in full.

Claims the paper could stand on, with tonight's evidence against each:

| Claim | Status |
|---|---|
| 1. The offloaded unit is the nested residual, not a second copy | Open and at risk. With a 1-bit base the nesting penalty (0.053) exceeds the gap to plain pruning (0.025), and that ladder is cut. Tested on the 2 + 1 ladder, where the base is usable. |
| 2. Tier 0 in the same policy | Supported in part: dropping the 7.7% of slots with gate below 0.1 costs 0.001 KL. No joint policy yet. |
| 3. The regime at 2 bits and below, 384-expert top-6 model, 128 GB laptop | Narrowed: below 2 bits the base is unusable. The regime is 2 to 3 bits. |
| 4. (new) Select experts by total routed mass on routing that resembles the pruned model | Supported with a qualifier: −0.184 KL at 384 → 192 (pod), but −0.052 on positions where the teacher is confident; the rest is fidelity on agentic and tool-call text (findings note, 5.1). −0.177 at 192 → 96 (Mac). No generation check yet. Not yet searched in the literature. |

## 3. Workstreams

Units: one expert at 1 bit per weight = 1 unit = 4.6 MB per layer. A 2-bit expert is 2 units, a
third-bit residual 1 unit, a native FP4 expert 4 units. Budgets are per layer.

### WR. Re-base on the total-mass 192 (new; gates W4 onward; no rubric points)

1. Export `fm-192` on the pod (`prune.py --skip-shard` for the two Engram shards, 152 GB). Running.
2. Download through a CPU pod on the volume, 16 parallel streams. On the Mac, clone the two Engram
   shards from `~/Models/v41-reap50` (`cp -c`, no extra space) and check every file against the
   export's `SHA256SUMS` and REAP-50's.
3. `local_eval.py` against the new checkpoint in BF16/FP4 on the Mac eval set. **Accept** when it
   beats REAP-50 on the same sessions by at least 0.10 KL (the pod saw 0.184 on 54 full sessions),
   and report the difference on teacher-confident positions separately.
3b. A generation check before any codes are fitted to the new set, since everything so far is
   teacher-forced KL: design and size it first (tool-call format validity and code completion on
   held-out prompts, new set against REAP-50). Open question: V4.1 has no local decoder on this
   Mac, so this may have to run on a pod with the reference implementation.
4. Capture routing and importance on the new model (the step-2c capture, four chunks).
5. Codes: flat 2-bit E8P for all 192 (reuse the fits of the experts shared with REAP-50, 76.5%, if
   the importance source is the same; otherwise refit all, about 7 h of GPU), then the 1-bit
   residual of the 2-bit base (`--residual-of`, about 4.5 h).
6. Nested expert sets inside the new 192 by total mass on its own routing: 176, 160, 144, 128, 112.
7. Eval set, version 2: 4,096 tokens per record, and the four sessions that share a first 1,024
   tokens no longer counted as four texts. Sub-1-bit comparisons keep the old set for continuity.

Only after step 3 passes is `~/Models/v41-reap50` removable (341 GB), and pass 2 must finish
first. `quant/e8-1bit-podimp` and `quant/e8-nest-r1` (66 GB) go with it.

### W0. Related work, read in full (gates the claims; no rubric points)

Unchanged. Adds a search for claim 4 (frequency-weighted or mass-based expert pruning criteria,
and calibration on the pruned model's routing).

### W1. Nested codes validated (rubric item 1: done, 10)

Fit complete, 40 of 40 layers; base + residual error 0.102 / 0.084 against 0.090 flat; end to end
+0.053 KL [+0.039, +0.071]. The acceptance bound (all-tier-2 KL at most 0.80) was met at 0.8001.
The alternating refit is no longer planned for the 1 + 1 ladder; it is used on the 2 + 1 ladder
only if its nesting penalty exceeds 0.02 KL.

### W4. Core result (rubric item 4: 0 → 25)

The claim: at equal resident memory, a routed cache of third-bit residuals over a 2-bit base gives
lower KL than any static assignment and than a separate-copy cache, at an SSD read rate the Mac
sustains.

Budgets B = 288, 336 and 384 units (53, 62, 71 GB of routed experts). A configuration keeps N
experts at 2 bits and K residual slots, 2N + K = B.

| Policy | What it is |
|---|---|
| F, flat | B / 2 experts at 2 bits, nothing else. The frontier to beat. |
| S, static-nested | fixed hot set of K residuals by routed mass |
| R, routed-nested (ours) | LRU of K residuals, asynchronous admission at gate ≥ τ |
| D, separate-copy | the hot expert's high tier is its native FP4 copy (4 units, swapped for the 2-bit copy: 2 extra units resident, 6 units moved per promote/demote cycle against 1) |
| O, per-session oracle | hot set = that record's own top K; the ceiling for any cache |

Build: a base-unit field in `routed_report.py` (bases are no longer 1 unit); a native-precision
high tier for `--dynamic BASE|HI`; `--tau-file` (W5).

Runs, on `fm-192`, eval set version 2:

1. Pass A, B = 384: F(192); R, S, O at N = 160, K = 64 and N = 144, K = 96; endpoints (all at
   3 bits, all at 2 bits).
2. Pass B, B = 288 and 336: F(144), F(168); R and S at N = 128, K = 32 and 80; N = 112, K = 64.
3. Pass C: D at the K and τ of the best R in each budget; tier 0 on top of the best R.
4. Carried over from revision 1 on REAP-50 (1-bit base): pass 2 only (oracle, N < 192,
   projection rungs), because it was already an hour in and its oracle result bounds the cache
   policy on any ladder. Passes 2b (separate copies) and 3 (budgets 240 and 336) are cancelled;
   the separate-copy comparison is made on the new ladder in pass C.

Acceptance and scoring inside the item (unchanged in form):

- 10 points: R beats F and S by at least 0.03 KL and beats D at equal memory and equal SSD rate,
  session-level bootstrap 95% interval excluding zero, at one budget, with SSD traffic at or
  below 2.5 GB/s at 30 tok/s.
- 15 points: the same at two of three budgets on eval set version 2.
- 25 points: reproduced on the pod's eval (108 sessions, full length) with the routed forward on
  GPU (W4b).

W4b is redefined: it no longer fits codes for 384 experts (X3 is demoted, see section 5). It ports
the routed forward to `collect.py`, uploads the Mac's codes, and scores the winning configurations
on the pod. Estimate from tonight's throughput (about 11 minutes per stream over 54 sessions):
2–3 h on one H100, under $15.

If R does not beat F on the 2 + 1 ladder either:

1. Admission by prediction (score layer l + 1's gate on layer l's output), measured offline first.
2. Report the routed-cache result against S and D as the contribution, with the selection finding,
   under a title that does not claim to beat pruning. Say so then; do not stretch the claim.

### W5. Unification (rubric item 5: 0 → 10)

Unchanged in design (one benefit-per-byte rule with one λ; `routed_policy.py`,
`routed_frontier.py`). Changes: tier 0 enters with tonight's measurement as its first point; the
sidecar tier is removed; the frontier figure plots both ladders, with F, S, R, D and O.

### W6. Real Apple Silicon streaming (rubric item 6: 0 → 10)

Unchanged in design. Two notes from tonight: the admission rates to replay are low (9.5 to 28 MB
per token at K = 96), so the bar is latency per admission, not sustained throughput; and the
network-volume episode is the same failure in miniature, a single reader faulting pages one at a
time, so the bench measures `pread` from a thread pool against `mmap` faults (X4) on purpose.

### W7. Learned policy and tier-aware codes (rubric item 7: 0 → 10)

- (a) 5 points: per-(layer, expert) thresholds from W5 beat one global τ by at least 0.02 KL.
- (b) 5 points: a jointly fitted nested code (base chosen so that base + residual is best, not the
  base alone) or tier-aware importance, gaining at least 0.02 KL at equal units.

### W8. Generality (rubric item 8: 0 → 10)

Unchanged (Qwen3-30B-A3B, download needs Jack's go). Adds the selection rule as a second thing to
replicate there, since it needs only routing statistics.

### W9. Paper (rubric item 9: 1 → 5)

Unchanged, plus a selection section (claim 4) that can also stand alone as a short note.

### WS. Selection, to its own conclusion (new; supports W9 and a possible REAP-50 v2)

- Tonight: 96 experts in one pass on full-model routing against two passes through `fm-192`
  (`x1b_pod.sh`), to test the rule "re-measure routing when the cut removes a large share of mass".
- X8, domain-balanced mass: calibration tokens are uneven by domain (english-chat 52K, tool-use
  451K) and english-chat is the one domain that got worse (0.448 → 0.572). Needs per-domain
  accumulation in `collect.py` and one calibration pass on a pod; run with W4b. Adopt only if it
  gains at least 0.03 domain-mean KL, because adopting it means re-basing again.
- Publishing a replacement REAP-50 on Hugging Face is Jack's decision.

## 4. Order and expected score

| When (estimate) | Work | Score after |
|---|---|---:|
| Tonight | pass 2 on REAP-50 (last run on the 1-bit base); X1 follow-up; export and download of `fm-192` | 31 |
| Day 1 | WR steps 3–7 (capture, fits); W0 reading; W6 bench built on tonight's traces | 31 |
| Day 2 | W4 passes A–C on `fm-192`; title decision | 41–46 |
| Day 3 | W5 policy and figure; W6 numbers; W7a | 61–66 |
| Day 4 | W4b and X8 on a pod; W7b | 76–81 |
| Days 5–6 | W8 | 91–96 |
| Day 7 | W9 full draft | 100 |

The score moves only on measured results, so a failed acceptance test holds it where it is. The
re-base costs about a day against revision 1's schedule and raises every number it touches.

## 5. Levers: status after tonight

| # | Lever | Status |
|---|---|---|
| X1 | Prune twice at 384 → 192 | Resolved. The ranking rule is the gain (−0.184 KL); the second pass is not (+0.014 at 192, −0.009 n.s. at 96). The second pass used 12× less calibration data than the one-pass table, so "slightly worse" is not separable from sample size. |
| X2 | Pruning as specialization | Closed (artifact). |
| X3 | Keep all 384 bases at 1 bit, offload residuals | Demoted. A 1-bit base scores 1.358 alone and misses fall to it. At 2 bits, 384 bases are 768 units, over any budget here. |
| X4 | OS page cache as the precision cache | Open, in W6. |
| X5 | Stronger base, finer rungs | (b) promoted: the 2 + 1 ladder is now W4's main line. (a) projection rungs are in pass 2. |
| X6 | Budgets by sensitivity | Open, after W4. |
| X7 | Shared base across experts | Open, low priority. |
| X8 | Domain-balanced selection | New, in WS. |
| — | Sidecars (old step 3) | Closed: +0.199 KL [+0.164, +0.219] with 15 layers fitted. |

## 6. Operating rules learned tonight

Pods:
- Never run the model from a network volume: one stream reads 120 MB/s, a random row takes
  2–3 ms. Create the GPU pod with a container disk that holds the checkpoint and copy first
  (16 streams, 476 GB in 7 minutes).
- Smoke-test every launcher on the image before the GPU clock matters; match readiness by
  substring; `PIP_BREAK_SYSTEM_PACKAGES=1` on Ubuntu 24.04 images.
- Estimates from measured throughput: about 390 tok/s for a calibration stream and 11 minutes per
  eval stream over 54 sessions. The X1 estimate (6.5–7.5 h) was 4× too high.
- CPU-only work (exports, transfers) runs on a CPU pod or alongside a GPU job that is already
  running, never on an idle GPU.

Evaluation:
- Report session-level bootstrap intervals; position-level intervals are about half as wide.
- Domain mean and pooled KL differ a lot (0.716 against 0.950 for REAP-50) because the agentic and
  tool-use domains are long and hard. State which one a number is.
- Mac numbers (22 sessions, first 1,024 tokens) and pod numbers (54 sessions, full length) are not
  comparable in absolute terms; only differences within one set are.
- A layer-local metric is not evidence. Twice now (GLM-4.5-Air, V4.1) a sidecar that improved its
  layer objective failed end to end. Score a partial fit end to end before finishing it.
- Settle the expert set before fitting codes to it.
- Write the prediction down before the run (findings note, section 6).

## 7. Resources

- Memory: 80 GB cap; the 8-variant sweep peaked at 64.1 GB.
- Disk: 373 GB freed on 2026-11-09 (REAP-75, the no-effect refit, sidecar files). The new
  checkpoint adds 152 GB; its codes about 100 GB.
- Pod: about $20 spent tonight (H100 about 4 h, CPU pod, volume). Volume `yqazs7aqe6` is 680 GB
  ($48 a month) and holds the full model and the export; delete it after W4b.

## 8. Decisions for Jack

1. Title: decide after the separate-copy sweep and the first 2 + 1 pass. Until then it stands.
2. A replacement REAP-50 on Hugging Face built from the total-mass table (public change).
3. Qwen3-30B-A3B download for W8.
