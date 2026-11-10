# What KEEP has learned, and which of it could go beyond the prior art

Written 2026-11-09 for the paper track (*Offload the Residual, Not the Expert*). Jack asked what the
whole project has measured that may not be in the literature, and whether any win is a bigger lever
than it looked. Three sources: this campaign's own results
([recovery log](2026-11-08-v41-reap-recovery-log.md)), a survey of the older KEEP documents, and the
prior-art search in the [plan](2026-11-09-routed-precision-plan.md). The plan (revision 2) holds
the schedule and the status of each lever; this file holds the evidence and the reasoning. Sections
5 and 6 were added the same evening, after the first results: read them before sections 3 and 4,
several of whose bets they overturn.

Nothing here has been checked against the literature beyond the six papers in the plan. Each lever
gets its own search before it is claimed.

## 1. Findings from this campaign

Differences are domain-mean KL on the Mac eval set (22 sessions, first 1,024 tokens), with a paired
block bootstrap over positions (`routed_report.py --compare`). The interval covers position noise
within the 22 sessions, not which sessions were drawn.

| Finding | Number | 95% interval |
|---|---:|---|
| Choosing 96 experts by total routed mass on the pruned model's routing, against REAP-75's set, both at 2 bits | −0.177 KL | −0.200 to −0.148 |
| A second prune pass (192 → 144 → 96) over one pass | −0.024 | −0.034 to −0.012 |
| Static allocator at 288 units against 144 experts at 2 bits | −0.003 | −0.009 to +0.005 |

- **Selection on the pruned model's routing is the lever; precision allocation was not.** Each of the
  two changes alone (the ranking rule, or measuring on the pruned model) gave about 0.025. Together
  0.177. A second pass is statistically real but small, most of it on tool-use.
- **Quantization statistics survive pruning; routing does not.** Refitting the 2-bit codes with
  importance measured on the pruned model changed nothing (0.753 against 0.747). The importance
  cosine was 0.976; the selection overlap with REAP-75 was 58%.
- **Losses are not additive.** KEEP 2-bit costs +0.17 KL on REAP-50 and +0.02 on a 96-expert model.
- **Two regimes by depth.** The top 96 experts hold 72–83% of routed mass in layers 0–19 and 94–98%
  in layers 20–39. Yet moving budget toward the diffuse layers lost to a uniform budget (1.046
  against 0.955 at 128 experts): concentration is not sensitivity.
- **A static 1-bit tier is nearly worthless** (0.025 KL at 192 units, nothing at 288). The 1-bit
  code's error is 0.31 per projection.
- **Nested lattice codes cost about 12%.** A 1-bit E8 residual on a 1-bit E8 base scores 0.101
  (gate/up) against 0.090 for a flat 2-bit E8P fit.
- **The pruned model beats the full model on tool-use**, by 1.22 nats per token of NLL, on the pod's
  own student as well as locally. Not explained.
- **Sidecars (running):** a rank-256 linear correction removes 36–76% of the layer gap where the
  gap is small (layers 0–3) and 9–15% where it is large (layers 4–6).

## 2. Findings from the older KEEP documents

From a survey of every document under `docs/` and the module docstrings. Rows marked ✓ were
re-read at the cited line for this note; the rest are as the survey reported them.

### Memory and speed come from different weights

- ✓ On V4-Flash, decoding one token reads 15.27 GB: attention 67%, routed experts 11%
  (`2026-08-21-roadmap.md:116-121`). Residents are dequantized to bf16 at bind.
- ✓ On V4.1, each expert kept in every layer costs 0.359 GB of RAM at 2.031 bits per weight "but adds
  no per-token" reads, and fitting the Mac "forces a ~50% prune"
  (`2026-11-05-v41-plan-and-phase0-status.md:15-21`).
- GLM-5.3 inverts this: residents are about 85% of decode traffic (`glm-5.3-flash/2026-10-06-plan.md`).

So on the DeepSeek family, expert precision buys memory and almost no speed. MatMoE argues dynamic
precision from latency; that argument does not carry to this architecture, and the memory argument
is the one this paper makes.

### The Mac thrashes at a 91 GB resident set

- ✓ At 90.99 GB of MLX-active memory the bind alone logged 535,792 swapouts (8.4 GB of anonymous
  buffers), and byte-identical work spanned 16.5× in time across rows that all passed the quiet
  gate (`2026-08-21-roadmap.md:84-89, 208-232`). The document's own conclusion is that the timings
  were "memory weather, not decode".
- Policy since then: leave the wired limit at its default (`docs/troubleshooting.md`).

Anonymous memory at this footprint gets compressed and swapped, which writes to the SSD and stalls
decode. Clean file-backed pages cannot be swapped; they are dropped and read again. That is the
case for lever X4.

### The quantizer

- ✓ A 1-bit code gave a 0.56 block cosine, "not a usable model"; the SwiGLU amplifies projection error
  about 2.8× at block level; E8P beats affine quantization below about 3.1 bits per weight and
  loses above; 2 bits is "at its rate-distortion limit" (`research/wave5-pilot-report.md:12-14, 122,
  215, 305, 339`).
- ✓ Incoherence processing (the random rotation QuIP# is built on) made the importance-aware fit 15%
  worse on the down projection, 9.18e-2 against 7.97e-2, because the rotation flattens a per-column
  importance that spans 180× (`research/wave5-pilot-report.md:144-149`).
- The codebook step returns 0.71 of fractional error per bit per weight and group-size steps
  0.17–0.24, so group size is the rate dial. Uniform group size beats non-uniform at matched rate.
  The belief that the down projection needs extra protection was not supported.
- Only 1 of 11,008 (layer, expert) pairs had no calibration routes.
- An fp32 reduction flips near-tie argmins in about 30% of scale last bits, so exact fits keep
  float64 sums on the CPU.

### Checks that lie

- ✓ A gate/up-swapped artifact loads, gives finite logits and scores cosine 0.9993 against the correct
  one; only a discrimination ratio (4.3e10) caught it (`research/wave5-materializer-report.md:12,
  101`).
- A per-call-sync benchmark harness produced false kills: the E8P kernel read 52.8 GB/s and was
  marked KILL, then 122–126 GB/s chained and 201.9–205.5 GB/s tuned (`2026-11-05-phase0-kill-tests.md`).
- Two "slower than source" verdicts were bugs in the measuring path: a Python-version mismatch that
  made about 50 kernels unreachable (8.36× once fixed), and a loader that re-globbed the build tree
  on every call (4.53× once fixed) (`handoff.md`, `2026-08-21-compounded-speed-verdict.md`).

### Recovery levers already tried on GLM-4.5-Air

- ✓ Router distillation was rejected: best global top-1 change "slightly negative"
  (`glm45-air/rc-pipeline.md:176`).
- ✓ A rank-4 sidecar raised report top-1 from 0.795 to 0.852 and cut KL to 0.314, but ran 1.718× the
  control and failed the speed gate (`glm45-air/rc-pipeline.md:167, 895, 930`). A proxy win that
  failed end to end.
- Raising non-expert precision changed nothing once a loader bug was fixed.

### Streaming and I/O

- A layer-outer, chunk-inner teacher loop is bit-identical to chunk-major and avoided 58.8 h of I/O.
- Prefill I/O favoured VQ: 6 `preadv` calls per layer against 1,536; the gap to the source model was
  compute, not I/O.
- Engram lookups depend only on token ids; the tables are memory-mapped and gathered on the host.
  The V4.1 plan already keeps Engram's 203 GB on the SSD.
- ✓ A reference point from outside: the Rapid-MLX V4.1 build reports teacher-forced KL 0.26 and top-1
  85% against the BF16 source, at 34–49 tok/s on a 256 GB M3 Ultra
  (`2026-11-08-v41-reap75-local-test-plan.md:125-126`).

## 3. What looks like it could go beyond the prior art

In the order I would bet on them on the morning of 2026-11-09. X numbers refer to the plan.
Outcomes so far: item 1 half right (the ranking rule, not the second pass), item 2 demoted, item 6
promoted to the main line. See section 6.

1. **Prune twice, at the first cut (X1).** The one clearly significant win of this campaign has only
   been applied to the second halving. REAP-50's own KL (0.578) is the floor under every number
   here, and it was chosen the old way. If the effect repeats at 384 → 192 it lifts the whole
   frontier, and "measure routing on the pruned model, rank by total mass" is a two-line change to
   REAP that anyone can use. Needs the full model: a pod run.
2. **Keep every expert and page residuals instead of pruning half (X3).** The V4.1 plan prunes 50%
   because each kept expert costs 0.359 GB and the model has to fit. All 384 one-bit bases cost the
   same memory as 192 experts at 2 bits. Whether 384 bases plus a residual cache beats REAP-50 at
   2 bits at equal memory is the title's claim at full scale, and no paper found so far compares
   pruning against residual offload. The local sweep (192 bases against 96 or 144 at 2 bits) is its
   proxy; the weak 1-bit base is the risk.
3. **The OS page cache as the precision cache (X4).** Wire the bases; map the residuals as clean
   file pages; `mincore` before use, `madvise` to fetch. No cache code, and under memory pressure the
   model loses precision instead of swapping. KEEP has already measured the failure this avoids
   (535,792 swapouts, 16.5× timing spread at 91 GB). This is specific to how unified memory and the
   macOS VM behave, and it turns "SSD churn" from writes into reads.
4. **Pruning as specialization (X2): checked the same day, and it is not that.** On the pod's
   full-length outputs (27 eval sessions held locally) the student-192 NLL advantage appears in
   every agent-style domain (tool-use −1.38 nats/token, agentic-python −0.33, agentic-rust −0.32,
   bash −0.16), but only where the teacher is surprised. On positions where the teacher's NLL is at
   most 2 the student is worse in every domain (+0.14 on tool-use). On tool-use, positions with
   teacher NLL above 5 are 30% of the tokens and carry −1.29 of the −1.38. At the worst of them the
   teacher's top prediction is its own tool-call markup (`｜DSML｜` after `<`) or end-of-sentence
   after `>`, while the record continues with foreign markup that the source datasets embed in
   message text. The full model is confidently wrong about text that is off its own format; the
   pruned model is less confident and is penalized less. That is a softer distribution on
   off-format data, not a better model. Two consequences: NLL on these domains is not a quality
   measure, and KL against the teacher there mostly measures agreement on format expectations.
   Lever dropped; the eval caveat stays.
5. **One artifact for every budget.** Because quantization statistics survive pruning and one mass
   ranking yields nested expert sets (pm-144 ⊂ pm-160; 97% overlap between passes), a single set of
   files can serve any memory budget: nested in expert count and in bits. MatMoE nests bits only.
   This is a packaging result that follows from 1–3 rather than an experiment of its own.
6. **Finer rungs and a stronger base (X5).** The 1-bit base is the weak link by three separate
   measurements. Projection-level residuals give 1.33- and 1.67-bit rungs from the existing codes,
   and a 2-bit base with a 1-bit residual moves the whole ladder to where the base is usable.
7. **Sensitivity-based budgets (X6)** and **a shared base across experts (X7)** are cheaper to test
   than to argue about.

Smaller results that are probably worth a paragraph or a short note each: incoherence processing
hurting importance-aware lattice fits; cosine similarity passing a gate/up-swapped model; and the
measurement lessons (per-call sync, one-sided paging gates).

## 4. A correction to my own earlier claim

I first wrote that a nested cache "holds twice as many hot experts" as a separate-copy cache at
equal memory. That is true only if the separate-copy cache keeps the low-precision copy of every hot
expert. One that drops it and reads it back on demotion uses the same memory as the nested design;
what it pays is traffic, three units per promote/demote cycle against one, and a slightly more
accurate high tier in return (0.090 against 0.101). The comparison is therefore run both ways, and
at equal SSD rate as well as equal memory. The title's claim may turn out to be about traffic and
simplicity more than about memory.

## 5. Results of the evening of 2026-11-09

### 5.1 Selection at the first cut (X1, pod)

Students of the full model, 54 eval sessions at full length, KL against the full model's logits.
Domain mean over 11 domains; pooled over positions in brackets.

| Student | Experts | KL | Top-1 |
|---|---:|---:|---:|
| Shipped REAP-50 (`reap192`) | 192 | 0.716 (0.950) | 0.800 |
| Total routed mass, full-model routing, one pass (`fm192`) | 192 | 0.532 (0.650) | 0.814 |
| 384 → 288 → 192, routing re-measured on the 288 (`pt192`) | 192 | 0.546 (0.663) | 0.809 |
| The Mac's 96 inside REAP-50 (`pm96`) | 96 | 1.163 (1.443) | 0.722 |
| 384 → 288 → 192 → 96 (`pt96`) | 96 | 1.154 (1.268) | 0.703 |

| Difference in domain-mean KL | Point | 95%, positions | 95%, sessions | Sessions better |
|---|---:|---|---|---:|
| `fm192` − `reap192` | −0.184 | −0.202 to −0.160 | −0.277 to −0.128 | 36 of 54 |
| `pt192` − `fm192` | +0.014 | +0.006 to +0.020 | +0.002 to +0.043 | 17 of 54 |
| `pt96` − `pm96` | −0.009 | −0.036 to +0.008 | −0.074 to +0.065 | 18 of 54 |

- The ranking rule carries the gain. REAP's per-route score ignores how often an expert is used;
  total routed mass counts it. `fm192` keeps 92.5% of the full model's routed mass and overlaps
  REAP-50 by 76.5%.
- By domain the gain is largest where records are long and hard (agentic-python 1.339 → 0.897,
  tool-use 1.995 → 1.294) and absent or negative where calibration data is thin: english-chat goes
  from 0.448 to 0.572. Calibration tokens by domain run from 43K (business-email) and 52K
  (english-chat) to 451K (tool-use). Total mass inherits that imbalance; REAP's per-route mean does
  not. This is lever X8.
- Re-measuring routing on the 288-expert model did not help at 192. Confound: the two-pass tables
  were ranked from 2 of 24 calibration sessions (257K tokens), the one-pass table from all 24.
- How this fits the Mac result (192 → 96 inside REAP-50: rule alone −0.025, pruned routing alone
  −0.025, both −0.177): a cut that keeps 92.5% of mass barely changes routing, so the full
  model's statistics describe the pruned model. A cut to 96 keeps about 74% and the routing it
  leaves is different. Working rule: re-measure when the cut removes a large share of mass. The
  96-expert follow-up (5.5) tests it.

Where the gain sits (checked the same evening, from the per-position files; `fm192` − `reap192`):

| Positions | Share | Domain-mean KL difference | 95%, sessions |
|---|---:|---:|---|
| All | 100% | −0.184 | −0.279 to −0.131 |
| Teacher NLL below 0.5 | 64% | −0.026 | −0.047 to +0.002 |
| Teacher NLL below 2 | 74% | −0.052 | −0.080 to −0.023 |
| Teacher NLL 2 or more | 26% | −0.465 | −0.637 to −0.361 |

- 59% of the pooled gain comes from the 15% of positions where the teacher's NLL is 5 or more, and
  15% from the 74% where it is below 2. On teacher-confident positions top-1 agreement does not
  move (0.947 against 0.944 pooled). On tool-use the gain on confident positions is −0.004: all of
  it is on positions the teacher did not expect.
- This is not the X2 artifact. On the surprised positions `fm192` agrees with the teacher's top
  token more often (0.42 → 0.51) and its NLL moves toward the teacher's (−2.22 → −0.73 relative to
  it): it reproduces the full model's expectations better, including its tool-call markup. It is
  fidelity on off-format and agentic text, not a general quality gain.
- English-chat is not established as worse: +0.124 over 4,319 positions in 20 short sessions,
  interval −0.050 to +0.206, 8 of 20 sessions better.
- "36 of 54 sessions better" is an artifact of the session mix: 30 sessions are english-chat or
  english-prose. Every tool-use (7), agentic (4), code-diffs (3), bash (2) and cyber (2) session
  improves.
- The four agentic sessions share their first 3,893–3,917 tokens of 16,383 (a common system
  prompt), so agentic-python and agentic-rust overlap by a quarter. With the two merged into one
  domain the difference is −0.159 (−0.251 to −0.122).
- The claim that survives: at 192 experts, total-mass selection is much closer to the full model
  on agentic and tool-call text and modestly closer on ordinary text (−0.05). Whether that is
  better for a user needs a generation check (tool-call validity, code completion), which no
  number here provides.

### 5.2 Routed precision, pass 1 (Mac, inside REAP-50, 1-bit base + 1-bit residual)

Numbers and intervals are in the recovery log. The decomposition at 288 units per layer:

| Configuration | KL | Cost over 192 experts at flat 2 bits (0.747) |
|---|---:|---|
| 144 experts at flat 2 bits | 0.812 | +0.065: 48 experts dropped |
| Routed cache, 192 bases + 96 residuals, τ 0.2 | 0.832 | +0.053 nesting, +0.032 misses that fall to 1 bit |
| Fixed hot set, same codes | 0.905 | +0.053 nesting, +0.105 misses |

- Routing the cache cuts the cost of misses by two thirds against a fixed hot set (0.105 → 0.032)
  with 10% of slots still missing. The remaining miss cost is what a 1-bit fallback costs.
- The nesting penalty alone (0.053) is nearly the whole cost of dropping 48 experts (0.065). With
  a 1-bit base, nested codes cannot beat pruning by much even with a perfect cache.
- SSD traffic was not binding: 0.28 GB/s at τ 0.3 and 0.83 GB/s at τ 0.2, against a 2.5 GB/s
  ceiling. A separate-copy cache moving three times the bytes would still be under it. The
  residual's traffic advantage only matters for small caches (K = 48 at τ 0.2 reads 2.72 GB/s).
- Per-token tiering with everything resident (MatMoE's setting) costs 0.022 KL for 54% fewer
  tier-2 slots and saves no memory. Expert bytes are about 11% of decode traffic on this family, so
  it buys little speed either.
- Per-token tier 0 is free at 7.7% of slots (+0.001 KL).

### 5.3 Sidecars

Closed. Fifteen fitted layers made KL worse by 0.199 [+0.164, +0.219] while every layer's holdout
objective improved. Details in the recovery log.

### 5.4 What a unit buys

From the static points inside REAP-50 (Mac eval), per unit per layer:

| Step | KL change | Per unit |
|---|---:|---:|
| 192 experts at 1 bit → 96 experts at 2 bits (192 units either way) | 1.358 → 1.039 | bits beat breadth at equal memory |
| 96 → 144 experts at 2 bits (+96 units) | −0.227 | −0.0024 |
| 144 → 192 experts at 2 bits (+96 units) | −0.065 | −0.0007 |
| 192 experts, 2 bits → native FP4/BF16 (+384 units or more) | −0.169 | at most −0.0004 |

The second bit is worth far more than the first expert it displaces; the last 48 experts are worth
little; and 0.17 KL sits above 2 bits. If a third bit recovers most of that 0.17 on the experts
that carry most of the routed mass, 96 residual units on the hot experts are worth more than the
96 units that buy experts 145–192. That is the case for the 2 + 1 ladder, and it is a guess until
the residual is fitted and scored.

### 5.5 96 experts, one pass against two (pod follow-up)

Pending when this section was written; results are appended to the recovery log.

## 6. Priors revisited

What I believed this morning, what the evidence says now, and what I do differently.

| Prior | Now | Evidence | Consequence |
|---|---|---|---|
| "Prune twice" is the lever, and it will repeat at 384 → 192 | Half wrong. The ranking rule is the lever at the first cut; re-measuring matters only when routing changes a lot | 5.1 | Re-base on `fm192`; rule stated as "rank by total mass on routing that resembles the target" |
| REAP-50 is a sound floor for everything else | Wrong. It gives up 0.18 KL at the same size | 5.1 | WR in the plan; every later number is re-measured |
| A 1-bit base is weak but usable as a fallback tier | Wrong. It is the dominant cost: 1.358 alone, and each missed slot falls to it | 5.2 | Main ladder becomes 2-bit base + 1-bit residual |
| Nested codes cost "about 12%" and that is acceptable | The 12% in weight error is 0.053 KL end to end, as much as dropping a quarter of the experts | 5.2 | Nested codes must be judged end to end; a joint fit is in W7 |
| The residual wins on memory or on traffic | Memory: only against keep-both caches. Traffic: not binding at K = 96 | 5.2 | The title's claim is at risk; decided by the separate-copy sweep and the 2 + 1 ladder |
| Keeping all 384 experts at 1 bit could beat pruning half (X3) | Very unlikely | 5.2, 5.4 | Demoted; W4b no longer fits 384 experts |
| Sidecars recover a useful share of the layer gap | Wrong end to end | 5.3 | Closed; no layer-local metric accepted as evidence again |
| The eval set is 22 independent texts | 19 at 1,024 tokens; two domains share one prompt | recovery log | Eval set version 2 |
| A network volume is a disk | It is a single-stream 120 MB/s device with 3 ms seeks | recovery log | Copy to local disk first; rule in the plan |
| X1 needs 6.5–7.5 GPU hours | 1.6 h | pod logs | Estimates from measured throughput |

Predictions registered before the runs that test them (2026-11-09, about 20:40 PST):

1. Separate-copy cache, K = 96, swap accounting, against the routed nested cache at 288 units: the
   copy wins on KL by 0.03–0.05 (about 0.78–0.80 against 0.832) and also beats 144 flat experts
   (0.812). Its SSD rate stays under 2.5 GB/s at τ 0.3.
2. Oracle cache at K = 96 lands within 0.015 of the routed LRU at τ 0.2: the policy is near its
   ceiling and the remaining loss is the base.
3. 144 experts all at nested 2 bits (288 units) scores about 0.86, worse than both the routed
   cache (0.832) and flat 144 (0.812).
4. 96 experts in two passes through `fm192` beat one pass on full-model routing by at least 0.05
   domain-mean KL.
5. On the 2 + 1 ladder, 144 experts at 2 bits plus a routed cache of 96 third-bit residuals
   (384 units) beats 192 experts at flat 2 bits by at least 0.03 KL.

Predictions 1 and the old-ladder half of the title question will not be run: Jack cut the 1-bit
base the same evening, and passes 2b and 3 were cancelled. Predictions 2 and 3 are in pass 2,
which finishes. If 5 fails, the paper is about routing a precision cache and about selection, and
the title changes. If 5 holds, the title stands with the qualifier that the base must be a usable
model.

### 6.1 Late revision: the lesson I drew from the 1-bit result was too narrow

Two hours after demoting X3 ("keep every expert, page what is missing") I reversed part of that.
The data refuted a 1-bit copy as the answer to a miss. It did not refute keeping a wide pool of
experts on the SSD. Three measurements point the other way:

- Pruning is the largest loss in the stack: REAP-50 in BF16 is 0.578 KL from the full model, and
  2-bit coding adds 0.169. The residual work addressed the 0.169.
- A cache that follows the session covers far more demand than a fixed set of the same size: 84–90%
  of slots against 75% at 96 of 192, and a routed 48 beat a fixed 96.
- Skipping a low-gate slot costs nothing (+0.001 KL at 7.7% of slots), and static pruning already
  answers every miss by re-routing to the best survivor.

Put together: keep residents at 2 bits, let the resident set follow the session, re-route a miss
among residents, and admit what the router asked for. Static pruning is that system with admission
off. This is conjecture C1 in the plan (revision 3) and it is untested. It was testable tonight
with the existing 2-bit codes and I did not see it, because every design I considered kept a
low-precision copy as the fallback.

Registered before the run (E1, pool = REAP-50's 192, flat 2 bits, same miss rule for both):

6. LRU cache of 144 against the static top 144 (0.812): better by 0.03–0.05 KL.
7. LRU cache of 96 against the static top 96 (1.039): better by at least 0.10.
8. Re-route beats drop on the same cache by at least 0.02.
9. The LRU of 144 at 2 bits beats the 1-bit-base residual cache at the same 288 units (0.832).

## 7. Prior art (moved here from the plan, revision 3)

Found 2026-11-09. Only abstracts were read for all but MatMoE (introduction and method).

| Paper | What it does | Overlap with us |
|---|---|---|
| MatMoE (arXiv 2611.01704, ICCAD 2026) | Nested integer codes per expert; bit-width per expert at run time from its routing score | Per-token precision with nested codes. All weights resident; no offload, no pruning. |
| DynaExq (arXiv 2511.15015) | Hot experts resident at high precision, low-precision fallback, asynchronous promotion | A no-stall cache. Separate copies. GPU memory, not SSD. |
| DyMoE (arXiv 2603.19172) | Run-time mixed precision by importance, depth-adaptive, look-ahead prefetch | Depth regimes, prefetch. |
| HOBBIT (arXiv 2411.01433) | On a miss, run a low-precision copy instead of waiting | A low tier as the miss path. |
| MC-MoE (arXiv 2410.06270) | Static mixed precision plus online per-token expert pruning | Per-token dropping with static precision. |
| PMQ, GEMQ (ICML 2026), EdgeMoE | Static per-expert bit allocation | The step 2c allocator. |

To read in full and to search for (W0): expert caches that re-route or skip on a miss; any
comparison of an expert cache against pruning at equal memory; REAP's comparison of its criterion
against frequency-based ones; calibration on the pruned model's routing.
