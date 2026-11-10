# DeepSeek-V4.1-Flash REAP recovery: progress log

Started 2026-11-08. Branch `jack/hopeful-cori-iszsfq`. This log records what was run, what
it showed, and what was learned, in order. The plan is the five-step recovery list from the
same day (non-uniform budgets, KEEP importance on the pruned model, residual sidecars, router
KD, LoRA KD), applied to the published REAP checkpoints
([REAP-50](https://huggingface.co/jackmaxwil/DeepSeek-V4.1-Flash-REAP-50),
[REAP-75](https://huggingface.co/jackmaxwil/DeepSeek-V4.1-Flash-REAP-75)).

Compute rules for this campaign: the Mac first. A Runpod pod only for a real experiment that
the Mac cannot do in reasonable time, never for setup or idle waiting. Mac memory cap 64 GB for
these runs (Jack, 2026-11-08).

## Starting point

REAP degradation, measured on the pod by router masking inside the full model (437,350
held-out positions; KL over the teacher's top 2,048 logits):

| Experts kept / 384 | KL | top-1 | NLL change |
|---:|---:|---:|---:|
| 256 | 0.511 | 0.842 | +0.025 |
| 192 (REAP-50) | 0.810 | 0.803 | -0.138 |
| 128 | 1.114 | 0.755 | -0.109 |
| 96 (REAP-75) | 1.349 | 0.720 | -0.131 |

- Fit: KL ≈ 1.84 · p^1.17 for pruned fraction p (worst miss 0.03); 1 − top-1 ≈ 0.33 · p^0.69.
  The marginal cost is ~0.005 KL per expert per layer, nearly flat from 33% to 75% pruned.
- Agentic and tool-use text loses the most and loses it early (exponent 0.85–1.1); business,
  code-diffs, cyber, networking and chat lose late (exponent 2–2.5).
- KEEP's own quant loss on V4-Flash (unpruned, 2.031 bpw) was KL 0.149 / top-1 0.899. If the
  two add, REAP-50 + KEEP ≈ 0.96 KL. KEEP's gate (KL ≤ 0.30, top-1 ≥ 0.85) fails on REAP alone
  at every tested level, so REAP loss is the main target.
- Both exported checkpoints were run on the Mac (CPU, reference code, dequantize-then-matmul
  stand-ins) over 5 eval sessions and matched the pod's masked-router numbers: REAP-75 KL 1.378
  vs 1.385, REAP-50 0.809 vs 0.841. Per-position differences of 0.1–0.3 KL are run-to-run
  chaos from rounding; the same session moved 0.17 KL between a solo and a batched run.

## Tooling

- `scripts/runpod/v41_reap/torch_kernel.py`: exact CPU ports of the FP8/FP4 kernels (unit tests
  in `tests/test_runpod_v41_torch_kernel.py`).
- `scripts/runpod/v41_reap/local_parity.py`: one exported checkpoint against the pod's numbers.
- `scripts/runpod/v41_reap/local_eval.py`: any number of router-masked variants of a local
  checkpoint in one layer-streamed pass, scored per position against the pod teacher's top-2,048
  logits. Variants may keep a different count per layer.
- Mac eval set (`~/Models/v41-reap-run/mac-eval-sessions.json`): 2 eval sessions per domain
  (22 sessions, 11 domains), each cut to its first 1,024 tokens, 22,506 positions. Cutting is
  exact because the model is causal; a smoke run matched the pod's per-position values for the
  same positions (REAP-50, eval-00070: 0.480 local vs 0.465 pod). Domains are equally weighted
  here, unlike the pod's position-weighted "all", and only early context (≤1,024 tokens) is
  covered.
- Cost: about 4.5 minutes fixed per pass over REAP-50's weights, plus per-token compute.

## Step 1: non-uniform per-layer budgets

Question: at a fixed total expert budget, does moving experts between layers beat REAP's
uniform count?

Allocation rule tried first (`mass-*`): REAP order within each layer is kept, so every variant
is nested inside REAP-50. Across layers, experts are removed greedily by their share of the
layer's routed output mass (Σ g·‖f‖ over calibration, normalized per layer), lowest first,
with a floor per layer. Variants (`~/Models/v41-reap-run/alloc/`):

| Variant | Mean kept | Per-layer range | Kept mass share (mean) |
|---|---:|---|---:|
| uniform-96 (REAP-75) | 96 | 96 | 0.650 |
| mass-96 | 96 | 24–180 | 0.700 |
| mass-96-f64 | 96 | 64–180 | 0.678 |
| uniform-128 | 128 | 128 | 0.725 |
| mass-128-f64 | 128 | 64–192 | 0.764 |

A frequency-based rule (`freq-*`) was built and rejected before running: it would keep only
10% of the output mass in layers 0–1.

Mass-based allocation cuts layers 20–31 and 36–39 hardest, where a few experts carry most of
the output. It treats every layer as equally sensitive, which is the assumption this run tests.

Results (`~/Models/v41-reap-run/evals/step1`, 22 sessions × 1,024 tokens, one pass of six
variants, 8,211 s, peak 22.5 GB footprint):

| Variant | KL (domain mean) | top-1 | NLL change |
|---|---:|---:|---:|
| REAP-50 (192) | 0.588 | 0.819 | -0.043 |
| uniform-128 | **0.955** | **0.754** | +0.160 |
| mass-128-f64 | 1.046 | 0.718 | +0.394 |
| uniform-96 (REAP-75) | **1.191** | **0.707** | +0.306 |
| mass-96-f64 | 1.344 | 0.673 | +0.584 |
| mass-96 | 1.457 | 0.636 | +0.815 |

Per domain (KL):

| Domain | REAP-50 | uni-128 | mass-128-f64 | uni-96 | mass-96-f64 | mass-96 |
|---|---:|---:|---:|---:|---:|---:|
| agentic (system prompt, see below) | 0.346 | 0.572 | 0.722 | 0.768 | 1.109 | 1.291 |
| bash | 0.265 | 0.507 | 0.599 | 0.671 | 0.888 | 1.053 |
| business | 0.711 | 1.020 | 1.304 | 1.385 | 1.511 | 1.724 |
| business-email | 0.413 | 0.711 | 0.972 | 0.992 | 1.108 | 1.356 |
| code-diffs | 0.263 | 0.461 | 0.625 | 0.647 | 0.813 | 1.049 |
| cyber | 0.566 | 1.049 | 1.414 | 1.518 | 1.953 | 2.060 |
| english-chat | 0.321 | 0.755 | 0.866 | 1.042 | 1.105 | 1.229 |
| english-prose | 0.359 | 0.604 | 0.706 | 0.773 | 0.894 | 0.945 |
| networking | 0.336 | 0.658 | 0.970 | 0.896 | 1.104 | 1.293 |
| tool-use | 2.544 | 3.595 | **2.609** | 3.645 | 3.195 | **2.740** |

Learnings:

- **Mass-based reallocation is worse than uniform**, by 0.09 KL at 128 and 0.15–0.27 at 96, on
  every domain except tool-use. The layers it cuts hardest (20–31, 36–39) are more sensitive than
  their output-mass share says, so "low mass share = cheap to cut" is false across layers. Within
  a layer REAP's ranking holds; across layers, mass share is not a sensitivity measure.
- **Tool-use runs the other way.** Mass allocation cuts tool-use KL by ~1.0 at both budgets
  (3.60 → 2.61 at 128, 3.65 → 2.74 at 96), even though it hurts everything else. Tool-use leans on
  the experts uniform pruning removes from the early layers (mass keeps more there). Tool-use is
  also the worst domain under every variant (KL 2.5 even at REAP-50), so a domain-aware budget
  (or keeping tool-use experts in the early layers) is a candidate, not a global reallocation.
- The REAP-50 → uniform-128 → uniform-96 curve on the Mac (0.59 / 0.95 / 1.19) has the same
  shape as the pod's (0.81 / 1.11 / 1.35). Mac numbers are lower because they cover only the first
  1,024 tokens and weight domains equally.
- **Eval-set flaw found:** all four agentic sessions (agentic-python and agentic-rust) share a
  ~3,900-token system prompt, so at 1,024 tokens they score the same text and returned identical
  numbers. The "agentic" row is the system prompt only, and the domain mean double-counts it. The
  108 eval sessions have only 100 distinct 1,024-token prefixes. A later pass that needs agentic
  coverage must run those sessions past token ~3,900 (about 4× their cost).
- Next for step 1, if it is pursued: measure sensitivity directly (cut one block of layers at a
  time and score) instead of inferring it from mass. Deprioritized behind steps 2–3, because the
  best case here is a small shift around uniform, while tool-use is the outlier worth chasing.

## Step 2: KEEP quantization of the pruned model (running)

- `scripts/runpod/v41_reap/keep_quant.py` fits KEEP's importance-aware E8P codes to REAP-50's
  192 experts per layer from their FP4 weights (the numpy fit with the Metal search, which is
  byte-identical to the shipped V4-Flash path).
- V4.1's down projection has input width 2304, which group size 512 does not divide, so down
  uses group 256: 2.031 bpw for gate/up, 2.0625 for down, 2.042 averaged.
- Weighted relative MSE (pod importance, layer 0): gate/up 0.084, down 0.047; layer 1: 0.089 /
  0.061. About 500 s per layer while the step-1 eval shares the CPU, so ~5.5 h for 40 layers.
- `local_eval.py --expert-override` runs these codes in the eval: `fp4-bf16` (exact FP4 decode,
  the control that isolates removing FP8 activation quantization), a code directory, or a
  per-expert mix of two directories.
- Later layers fit slightly worse: layers 16–17 gate/up 0.092, down 0.082; about 350 s per layer
  once the CPU eval ended.
- 2-bit fit done for all 40 layers (`quant/e8p-podimp`, about 4.5 h): mean weighted relative MSE
  gate/up 0.090, down 0.074, worst layer 17 (0.092). The 1-bit fit started right after.
- The "new importance" half (imatrix measured on the pruned model, 96 calibration records of
  1,024 tokens, `--capture-imatrix`) started when step 1 finished (output
  `~/Models/v41-reap-run/imatrix/reap50-pruned.npz`).
- The first capture (all 96 records in one pass) hit the 64 GB guard after 56 minutes: peak RSS
  64.4 GB, of which only 32 GB was the process's own footprint (the rest is mmapped checkpoint
  pages, which RSS counts). The importance stats are sums, so it was rerun as four passes of 24
  records (`imatrix/chunk{0..3}.npz`) to be added together.
- Capture done: 76,471 tokens, four passes of 15–38 minutes at ~31.5 GB peak RSS, summed into
  `imatrix/reap50-pruned.npz`. 2 of 7,680 surviving experts never routed; 29 routed fewer than
  20 times.
- Pruned-model importance vs the full model's (cosine per expert, mean over routed experts),
  against a noise floor from splitting the pod's 24 sessions into two halves:

  | Space | Pod half vs half (noise) | Pruned vs pod | Worst layers |
  |---|---:|---:|---|
  | gate/up input (hidden) | 0.995 | 0.976 | 15, 21–27 (~0.96) |
  | down input (expert intermediate) | 0.895 | 0.788 | 6–9 (~0.73) |

  Both shifts are larger than the noise, and the down input shifts much more. The comparison is
  confounded: the Mac records are the first 1,024 tokens of each calibration session, while the
  pod summed whole sessions, so data and pruning both differ. The full model cannot run on this
  Mac to separate them, so the test is end to end: requantize with the pruned importance only if
  the 2-bit codes with pod importance show a quant loss worth recovering.
- Eval started (`evals/step2-fp4-bf16`, then `evals/step2-e8p-podimp`): REAP-50 and uniform-96,
  first with the exact FP4 weights in bf16 (control: isolates dropping FP8 activation
  quantization), then with the 2-bit codes.

### Result: KEEP 2-bit on the pruned model (pod importance)

Same 22 sessions × 1,024 tokens. Each pass ~65 min (two variants), peak RSS 52–54 GB.

| Experts | Weights | KL | top-1 | NLL change |
|---|---|---:|---:|---:|
| 192 (REAP-50) | FP4 + FP8 activations (shipped) | 0.588 | 0.819 | -0.043 |
| 192 | FP4 decoded to bf16 (control) | 0.578 | 0.821 | -0.028 |
| 192 | KEEP 2-bit, pod importance | **0.747** | **0.788** | -0.092 |
| 96 (REAP-75) | FP4 + FP8 activations | 1.191 | 0.707 | +0.306 |
| 96 | FP4 decoded to bf16 | 1.192 | 0.708 | +0.306 |
| 96 | KEEP 2-bit, pod importance | 1.216 | 0.689 | +0.313 |

Per domain, REAP-50 (control → 2-bit): agentic 0.33 → 0.42, bash 0.26 → 0.34, business 0.71 →
0.75, business-email 0.42 → 0.48, code-diffs 0.26 → 0.35, cyber 0.55 → 0.78, english-chat 0.33 →
0.46, english-prose 0.36 → 0.41, networking 0.34 → 0.45, tool-use 2.47 → 3.36.

Learnings:

- The control matches the shipped path (0.578 vs 0.588), so the override path is sound and FP8
  activation quantization costs nothing measurable here.
- **KEEP 2-bit adds +0.17 KL / −0.033 top-1 on REAP-50**, close to its +0.149 on unpruned
  V4-Flash. On REAP-50, REAP and KEEP losses roughly add.
- **On REAP-75 it adds only +0.02 KL**, with the same per-expert codes. The losses are far from
  additive once the prune error is large: KL against the teacher is already high, and extra weight
  noise moves the distribution in directions that matter less. Two consequences: the quant loss is
  worth recovering only at mild prune levels, and the step-2b comparisons (all at high KL) will
  favour whatever has less prune error.
- Tool-use takes the biggest quant hit at REAP-50 (+0.89 KL) and even improves slightly at REAP-75
  (3.65 → 3.48). Tool-use is the fragile domain under every change so far.

Next (queued behind the 1-bit fit): requantize at 2 bits with the pruned-model importance
(`quant/e8p-prunedimp`, blended: pruned importance per expert where it routed ≥20 times, the pod's
otherwise, each normalized to mean 1, floored at 1e-4; 29 experts fall back), on the GPU alongside
the step-2b evals on the CPU.

## Step 2b: is REAP the right tool, or a low-bit floor for every expert? (Jack's idea)

Idea: pruning is KEEP's precision ladder with a hard 0-bit rung. Give every expert a low-bit
base instead, and let the runtime raise precision per expert (nested residual stages, hot
residuals in RAM, the rest on SSD).

Memory facts: one expert slot across all 40 layers costs 0.359 GB at 2.031 bpw, so all 384
experts are ~138 GB at 2 bpw (does not fit) and ~70 GB at 1.03 bpw (fits). Decode reads 240
experts per token (6 × 40 layers), so speed depends on bits read, not bits stored.

### Residency replay (done, CPU, no model)

Recorded full-model routing from the 24 calibration sessions (49,152 tokens, first 2,048
tokens each) replayed through a per-layer hot set of 1-bit residual stages (4.6 MB per expert
per layer), resetting at each session:

| Hot set / layer | Residual RAM | Decayed-count miss rate | Misses / token | Gate mass on misses (LRU, 6 sessions) |
|---:|---:|---:|---:|---:|
| 48 | 8.8 GB | 26.5% | 63.5 | 27.8% |
| 96 | 17.7 GB | 13.6% | 32.7 | 14.5% |
| 128 | 23.6 GB | 9.2% | 22.0 | 8.8% |
| 192 | 35.3 GB | 4.4% | 10.5 | 3.7% |
| 256 | 47.1 GB | 2.0% | 4.9 | 1.7% |

- Fetching every miss from the SSD (~0.7 ms per 4.6 MB at 6–7 GB/s) costs ~23 ms/token at 96
  and ~7 ms at 192, against a ~32 ms budget at 31 tok/s. Fetch-on-miss only works at 192+ hot,
  i.e. 70 GB base + 35 GB residuals, which does not leave room on a 128 GB machine.
- The workable form is serve-from-base: a miss runs at the 1-bit base stage with no stall, and
  the residual is promoted in the background. At hot 128 (23.6 GB) about 9% of the routed gate
  mass would run at base precision.
- LRU and decayed counts behave alike; a static REAP-ordered hot set is far worse (58% misses at
  128). REAP ranks by mean contribution per routed token, not by frequency.
- Side finding: under the full model's routing, REAP-50's 192 survivors receive only ~60% of
  routed slots (the static-192 miss rate is 40%). REAP-50 reroutes 40% of all expert slots.

### Static frontier test (queued)

Inside REAP-50, REAP-75's 96 experts per layer are "hot" and the other 96 "cold":

| Variant | Memory (relative) | Question |
|---|---:|---|
| hot 96 @ 2.03, cold dropped (REAP-75 + KEEP) | ~195 | prune baseline |
| all 192 @ 1.03 | ~198 | REAP vs a low-bit floor at equal memory |
| hot 96 @ 2.03, cold 96 @ 1.03 | ~294 | the cost of a cold expert at 1 bit vs 0 |

KEEP's 1-bit tier is the E8 codebook with 8-bit indices (`--code-bits 8`): 1.031 bpw gate/up,
1.0625 down. Smoke fit on layer 0: weighted relative MSE 0.32 against 0.094 at 2 bits, so the
1-bit floor is coarse at the weight level; the KL test decides.

The 1-bit fit finished in 3.3 h (`quant/e8-1bit-podimp`, much faster than the smoke estimate once
the CPU was free): mean weighted relative MSE gate/up 0.311, down 0.253, about 3.4× the 2-bit
error. Evals running: `evals/step2b-1bit` (REAP-50 and uniform-96, all experts at 1 bit), then
`evals/step2b-mixed` (REAP-75's 96 at 2 bits, the other 96 at 1 bit). The 2-bit row of the table
is `evals/step2-e8p-podimp` uniform-96 (KL 1.216).

## Step 2c: precision as a learned lever (static allocation first)

Framing (Jack, 2026-11-08): precision should be a routed, per-token dimension of the MoE, with
nested codes, so that pruning (the 0-bit rung), quantization and offloading become one
optimization: minimize E[KL] + λ·E[bits read per token] subject to bits stored ≤ RAM. Static
decisions are how many rungs each expert stores; dynamic decisions are which rung each routed
expert runs at for a token.

Evidence from this campaign that per-axis budgets are the wrong shape:

- Losses do not add: the same 2-bit codes cost +0.17 KL on REAP-50 and +0.02 on REAP-75.
- Proxies fail: moving experts between layers by mass share lost to uniform (step 1).
- Allocation is input-dependent: the reallocation that hurt every domain helped tool-use by ~1 KL.
- Decode reads 240 experts per token, so bits read (bandwidth) and bits stored (RAM) are separate
  levers.

Closest prior work found so far (not a full literature search): HOBBIT (arXiv 2411.01433) serves
cache-miss experts from low-precision copies with layer-level prefetch (up to 9.93× decode on
consumer GPUs); Mixtral offloading, Pre-gated MoE, Fate and ProMoE predict next-layer experts;
Matryoshka-style quantization trains nested codes; HAWQ / Bayesian Bits learn per-layer
bit-widths. The combination tested here, where the router itself chooses (expert, rung) under a
bits penalty with nested codes, has not been found in that set.

Test ladder:

1. Static allocation without training (Mac). `scripts/runpod/v41_reap/precision_alloc.py`:
   within each layer, rungs 0/1/2 bits under a fixed unit budget (1-bit expert = 1 unit), greedy
   by estimated error reduction per unit, cost = expert's routed output-mass share × relative
   output error at that rung (layer-mean KEEP errors; dropped = 1). Uniform budget across layers.
2. Check the estimator against measured KL (step 2b points).
3. Router over (expert, rung) trained by KD with a bits penalty (pod; merges with step 4).
4. Trainable nested codebooks (pod; merges with step 5).

Step 1 allocations (`~/Models/v41-reap-run/alloc/prec-{192,288}`):

| Budget / layer | Memory equivalent | Kept (mean) | at 2 bits | at 1 bit | Overlap of 2-bit set with REAP-75 |
|---:|---|---:|---:|---:|---:|
| 192 | REAP-75 at 2 bits | 101 | 91 | 10 | |
| 288 | REAP-75 at 2 bits + rest at 1 bit | 147 | 141 | 6 | 54% |

The estimator does not favour a 1-bit floor. Routed mass is concentrated enough that upgrading a
heavy expert to 2 bits buys more estimated error than keeping a light one at 1 bit, so it drops
the light tail and keeps almost everything else at 2 bits. Its 2-bit set also differs from REAP's
(54% overlap), because it ranks by total routed mass (frequency × contribution) where REAP ranks by
contribution per routed token. Step 2b's "all 192 at 1 bit" eval tests the floor directly; the
step-2c evals (`evals/step2c-prec-{192,288}`) are queued behind it. If the floor beats these,
mass × error is the same kind of failed proxy as in step 1 (it ignores rerouting and rare but
critical experts), and allocation has to be learned against KL.

### SSD churn under admission thresholds (residency replay, CPU)

Same 24 calibration sessions (49,152 tokens of full-model routing), decayed-count hot set per
layer, 4.6 MB per residual stage. A miss whose gate weight is below the threshold is served at the
base rung and never promoted; misses at or above it are read from the SSD. Gate weights sum to 1.5
per token-layer (route_scale); median 0.22. Output `~/Models/v41-reap-run/residency-admission.txt`.

| Hot / layer | Threshold | Promotions / token | SSD MB / token | GB/s at 30 tok/s | Gate mass never promoted |
|---:|---:|---:|---:|---:|---:|
| 96 | none | 33.0 | 152 | 4.55 | 0 |
| 96 | 0.2 | 15.2 | 70 | 2.09 | 5.7% |
| 128 | none | 22.6 | 104 | 3.12 | 0 |
| 128 | 0.2 | 10.4 | 48 | 1.44 | 4.4% |
| 128 | 0.3 | 3.4 | 16 | 0.47 | 16.5% |
| 192 | none | 11.8 | 54 | 1.63 | 0 |
| 192 | 0.2 | 6.2 | 29 | 0.86 | 3.2% |
| 192 | 0.3 | 3.0 | 14 | 0.42 | 15.9% |

- Promoting every miss is not sustainable below 192 hot: 3–4.5 GB/s of continuous reads against
  ~5.4 GB/s sequential, before any prefill traffic.
- A gate-weight threshold of 0.2 halves the traffic at every hot size for 3–6% of gate mass left
  at base precision. At hot 128 (23.6 GB of residuals) that is 1.4 GB/s, about a quarter of the
  SSD. 0.3 cuts traffic another 3× but leaves ~16% of the mass unpromoted.
- The gate weight is a usable admission signal, and it is the same quantity a learned router
  would weigh against a bits penalty: low-weight picks contribute little, so they are the cheap
  ones to run at base. Whether 4–6% of gate mass at 1 bit is affordable is what step 2b measures
  (all 192 at 1 bit is the 100% case).
- Not yet replayed: next-layer prefetch (needs hidden states, not just routes) and second-hit
  admission.

### Step 2b results: the static frontier

Same 22 sessions × 1,024 tokens. Units are per-layer memory in 1-bit-expert equivalents
(2-bit expert = 2, FP4 ≈ 4.25).

| Configuration (inside REAP-50) | Units / layer | KL | top-1 | NLL change |
|---|---:|---:|---:|---:|
| REAP-75's 96 at 2 bits, rest dropped | 192 | **1.216** | **0.689** | +0.313 |
| all 192 at 1 bit | 192 | 1.358 | 0.637 | +0.425 |
| REAP-75's 96 at 1 bit, rest dropped | 96 | 1.826 | 0.565 | +0.943 |
| REAP-75's 96 at 2 bits, other 96 at 1 bit | 288 | 0.954 | 0.738 | +0.010 |
| all 192 at 2 bits | 384 | 0.747 | 0.788 | -0.092 |
| all 192 FP4 (bf16 decode) | ~816 | 0.578 | 0.821 | -0.028 |

Per domain (KL), all-192 at 1 bit / mixed 288: agentic 1.01 / 0.55, bash 0.82 / 0.45, business
1.80 / 1.06, business-email 1.16 / 0.75, code-diffs 0.93 / 0.47, cyber 1.57 / 1.09, english-chat
1.59 / 0.87, english-prose 0.90 / 0.56, networking 0.93 / 0.60, tool-use 3.22 / 3.56.

Learnings:

- **At equal memory, pruning plus 2 bits beats a 1-bit floor**: 1.216 vs 1.358 KL. The static
  version of "no REAP, every expert at low bits" loses. The step-2c estimator predicted this
  direction (drop the light tail, keep the rest at 2 bits).
- **KEEP's 1-bit tier is expensive**: all of REAP-50 at 1 bit costs +0.78 KL over FP4, against
  +0.17 at 2 bits. The 1-bit E8 codebook (weight error 0.31) is too coarse to be a floor that runs
  a large share of the routed mass.
- **But a 1-bit rung beats dropping**: adding REAP-50's other 96 experts at 1 bit to REAP-75 at 2
  bits cuts KL from 1.216 to 0.954. Upgrading those same 96 from 1 to 2 bits cuts it further to
  0.747. Per unit of memory the two steps are worth about the same (-0.26 and -0.21 KL per 96
  units), and upgrading REAP-75's 96 from 1 to 2 bits is worth more (-0.61). Marginal value per
  unit: hot 1→2 > cold 0→1 ≈ cold 1→2. That is the shape a learned allocator should find, and it
  is consistent with the mass × error estimator.
- Frontier so far (units, KL): (96, 1.83), (192, 1.22), (288, 0.95), (384, 0.75), (~816, 0.58).
- For the serve-from-base runtime: running a miss at the 1-bit base is not free. The all-1-bit
  pass is the 100%-of-mass case (+0.78 KL); at the 4% never-promoted mass of threshold 0.2 the
  cost is unmeasured and should not be assumed linear. A better low rung (1.5 bits, or a 1-bit
  base with a small learned correction) would change this picture more than any allocation rule.
- **Tool-use anomaly**: tool-use KL is 2.5–3.6 under every variant and does not order with
  memory (1-bit uniform-96 at 3.06 beats 2-bit uniform-96 at 3.48), and its NLL change on REAP-50
  is -1.1 to -1.5 (the pruned model predicts the actual tokens much better than the teacher). That
  pattern looks like a measurement problem in those two sessions' teacher logits (alignment or
  template), not a model effect. To check before tool-use numbers drive any decision.

### Tool-use anomaly: checked

- Teacher logits are aligned with the tokens (argmax matches the next token 56–83% of the time,
  the current token 0%). The pod's own REAP-50 run shows the same gap as the Mac, so it is not a
  local bug.
- On the pod data the pruned model beats the full model on actual-token NLL in the tool and code
  domains: tool-use -1.22 nats, agentic -0.33, english-chat -0.27, bash -0.16; it loses on prose,
  business and cyber (+0.3 to +0.4). On tool-use session eval-00018 the full model's NLL is 3.9–5.2
  nats per token in every position band, against 1.4–4.3 for REAP-50.
- The full model has catastrophic positions: 1.4% of all eval positions have teacher NLL > 15
  nats (3.6% in agentic), on trivially predictable tokens (for example `"title": "` then
  `Content`, teacher 21.8 nats, REAP-50 0.0). REAP-50 has them too, but fewer (2,199 vs 2,895).
- Those positions carry only ~4% of the measured KL; dropping them moves the domain means by
  0.015–0.02 (REAP-50 0.578 → 0.563, 2-bit 0.747 → 0.725, 1-bit 1.358 → 1.335). The frontier
  conclusions do not change. Tool-use KL stays ~2.4 without them, so the tool-use gap is broad,
  not just spikes.
- Reading: on DSML tool-call text the full model, run through the reference torch path with the
  FP4/FP8 kernels, is far worse than a frontier model should be, and removing experts improves it.
  Either some experts misbehave in this numeric path on that text, or the full model really is
  weak there. Until that is settled, tool-use KL against this teacher is not a quality measure
  (the teacher is the worse model on that domain), and NLL on actual tokens should be reported
  next to it. Isolating it needs the full model (pod or base download), so it is noted, not
  chased.

### Step 2c results: static per-expert allocation

| Configuration | Units / layer | KL | top-1 | NLL change |
|---|---:|---:|---:|---:|
| REAP-75's 96 at 2 bits (prune + KEEP) | 192 | 1.216 | 0.689 | +0.313 |
| all 192 at 1 bit | 192 | 1.358 | 0.637 | +0.425 |
| **prec-192** (≈101 kept: 91 at 2 bits, 10 at 1 bit) | 192 | **1.014** | **0.732** | +0.158 |
| REAP-75's 96 at 2 bits + other 96 at 1 bit | 288 | 0.954 | 0.738 | +0.010 |
| **prec-288** (≈147 kept: 141 at 2 bits, 6 at 1 bit) | 288 | **0.809** | **0.773** | -0.039 |
| all 192 at 2 bits | 384 | 0.747 | 0.788 | -0.092 |

Per domain (KL), REAP-75@2 → prec-192: agentic 0.73 → 0.55, bash 0.71 → 0.52, business 1.51 →
1.38, business-email 1.07 → 0.78, code-diffs 0.72 → 0.50, cyber 1.53 → 1.09, english-chat 1.19 →
1.19, english-prose 0.82 → 0.67, networking 0.90 → 0.68, tool-use 3.48 → 3.25.

Learnings:

- **At REAP-75's memory, the allocator cuts KL by 0.20 (−17%) and lifts top-1 by 4.3 points**,
  better or equal on every domain. At 288 units it beats the REAP-ordered mix by 0.145 and comes
  within 0.06 of all-192-at-2-bits with 75% of the memory.
- The estimator is a crude proxy (layer-mean errors, mass × error, no training), so this is a
  floor on what a learned allocation can do, not a ceiling.
- Open question before crediting "precision allocation": prec-192 is mostly a different choice
  of which ~96 experts get 2 bits (58% overlap with REAP-75), plus a small 1-bit tail. Two things
  changed at once: the ranking criterion (total routed mass instead of REAP's contribution per
  routed token) and the routing it is measured on (the pruned model's own instead of the full
  model's). Ablation running (`evals/step2c-selection`, all 96 at 2 bits): full-model total mass,
  pruned-model REAP score, pruned-model total mass.

### Step 2c selection ablation

All variants keep 96 experts per layer at KEEP 2 bits (`e8p-podimp`, 192 units), so only the
choice of experts differs. Domain-mean KL over the same held-out eval set:

| 96 experts chosen by | Overlap with REAP-75 | KL | top-1 |
|---|---:|---:|---:|
| REAP score, full-model routing (shipped REAP-75) | 1.000 | 1.216 | 0.689 |
| total routed mass, full-model routing | 0.664 | 1.191 | 0.707 |
| REAP score, pruned-model routing | 0.784 | 1.188 | 0.701 |
| **total routed mass, pruned-model routing** | 0.580 | **1.039** | **0.733** |
| prec-192 (the same ranking, plus a 1-bit tail) | — | 1.014 | 0.732 |

Learnings:

- **Selection is most of the step-2c gain.** Of prec-192's 0.202 KL cut, 0.177 comes from
  choosing the 96 experts by total routed mass measured on REAP-50's own routing; the 1-bit tail
  adds 0.025 (plus 0.023 on tool-use, the domain where it helps most).
- **The two changes only pay together.** Each alone moves KL by about 0.025; combined they move
  it by 0.177. Ranking by total mass on the full model's routing does not see how routing
  redistributes once half the experts are gone, and REAP's per-route score on the pruned model
  still favors experts that are strong when picked but rarely picked.
- So a second prune pass measured on the first pruned model (a cheap form of iterative pruning)
  is worth more than KEEP's precision trade at this budget. The 1-bit rung is still a real but
  small positive at 192 units, and grows at 288 (prec-288 0.809 vs mixed 0.954, of which the
  ranking accounts for an unmeasured share).
- The mass capture came from calibration records (first 1,024 tokens of the 24 calibration
  sessions), not the eval set, so this is not leakage.

### Step 2 closed: pruned-model importance for the KEEP fit

The 2-bit refit with importance measured on REAP-50 itself (`quant/e8p-prunedimp`) was evaluated
against the pod-importance fit (`e8p-podimp`), same eval set:

| Variant | Importance | KL | top-1 | NLL change |
|---|---|---:|---:|---:|
| REAP-50, all 192 at 2 bits | pod (full model) | 0.747 | 0.788 | -0.092 |
| REAP-50, all 192 at 2 bits | pruned model | 0.753 | 0.790 | -0.106 |
| prunedmass-96 at 2 bits | pod (full model) | 1.039 | 0.733 | +0.154 |
| prunedmass-96 at 2 bits | pruned model | 1.027 | 0.734 | +0.159 |

Learnings:

- **No effect.** The differences (+0.006, -0.012) are within run-to-run noise, and the
  prunedmass-96 gain sits almost entirely in tool-use (3.42 to 3.28), the domain whose KL is not
  a quality measure against this teacher. Without tool-use the two fits tie.
- So the input importance KEEP weights its fit by barely moves after pruning (hidden-space cosine
  0.976 in step 2 already said so), while the routing mass used for selection moves a lot (step 2c
  ablation). Recomputing statistics on the pruned model matters for which experts to keep, not
  for how to quantize them.
- Peak RSS 56.6 GB, 48 minutes for two variants.

### Step 2d: iterative pruning (192 → 144 → 96)

First pass: the top 144 experts per layer by total routed mass on REAP-50's routing
(`alloc/iter/pm-144.json`). Routing recaptured on REAP-50 masked to those 144 (same 96
calibration records in 4 chunks, 13–15 minutes each, about 31 GB peak RSS,
`imatrix/pm144-chunk*.npz`), then the top 96 of the 144 by the new mass (`alloc/iter/iter-96.json`).
The second pass changes little: iter-96 overlaps the one-pass 96 by 97.3% (minimum 94.8% in any
layer, about 2.6 experts per layer). Eval at KEEP 2 bits (`e8p-podimp`), `evals/step2d-iterative`:

| Configuration | Units / layer | KL | KL without tool-use | top-1 |
|---|---:|---:|---:|---:|
| one-pass mass, 96 at 2 bits | 192 | 1.039 | 0.801 | 0.733 |
| **two-pass mass, 96 at 2 bits** | 192 | 1.016 | 0.795 | 0.731 |
| prec-192 (one-pass ranking + 1-bit tail) | 192 | 1.014 | — | 0.732 |
| REAP-75's 96 at 2 bits + other 96 at 1 bit | 288 | 0.954 | 0.693 | 0.738 |
| **one-pass mass, 144 at 2 bits** | 288 | 0.812 | 0.566 | 0.770 |
| prec-288 (≈141 at 2 bits + 6 at 1 bit) | 288 | 0.809 | 0.564 | 0.773 |

Learnings:

- **Iterative pruning saturates after one pass.** The second pass gains 0.023 KL, below the
  0.03 bar set before the run, and 0.006 of it outside tool-use. One re-measurement on the pruned
  model captures nearly all of the routing redistribution.
- **At 288 units the 1-bit rung adds nothing either:** 144 experts at 2 bits score 0.812 against
  prec-288's 0.809. prec-288's lead over the REAP-ordered mix is all selection.
- Net for step 2: under the current codes, the whole gain over REAP-75 + KEEP comes from
  re-ranking experts by total routed mass on the pruned model's routing. The static 1-bit rung
  helps only marginally (0.025 at 192 units, nothing at 288). A low rung that pays needs to be
  better than this 1-bit E8 fit (errors 0.31/0.25), or dynamic, or both.
- Step 2 is done; step 3 (residual sidecars) is next.

## Precision as a routed dimension (the paper track)

Working title: *Precision as a Routed Dimension: unifying pruning, quantization and offloading in
MoE inference*. Jack, 2026-11-09: build, run, test, commit; this is the headline to reach.

### Claim to test

Static compression picks one precision per expert (or drops it) before inference. The router
already decides, per token, how much each expert matters (its gate weight). Make precision a
routed decision too: every (token, expert) slot runs at one of three tiers chosen at inference
time, from codes nested so that each tier is a prefix of the next:

| Tier | Weights | Lives in | Cost per use |
|---|---|---|---|
| 0 (drop) | none | nowhere | 0 |
| 1 (base) | 1-bit E8 code | unified memory, always resident | 4.6 MB read |
| 2 (base + residual) | 1-bit base + 1-bit E8 residual of it | residual in an LRU cache, misses on SSD | 9.2 MB read |

Pruning is tier 0 (per expert or per token), quantization is the tier-1/tier-2 choice, and
offloading is where the tier-2 residual lives. One policy (gate thresholds `tau_drop`, `tau`, and
the cache size K) sets all three, and three costs fall out of it: resident memory (bases + K
residuals per layer), SSD traffic (residual admissions per token), and unified-memory bytes read
per token (the decode speed bound on Apple Silicon, 546 GB/s on the M4 Max).

The cache never stalls decode: a slot gets tier 2 only if its residual is already resident; a slot
with gate >= tau admits its residual by an async SSD read that serves later tokens. This is the
HOBBIT-style "use a cheaper copy on a miss" (arXiv 2411.01433), with the cheaper copy being the
prefix of the expensive one, so a hit costs only the residual's bytes and memory holds no
duplicate.

### Evidence so far

- Nested codes are almost free: on layer 5 (2 experts), base + residual scores 0.103 (gate/up) and
  0.083 (down) weighted relative MSE, against 0.090 / 0.074 for a flat 2-bit E8P fit (all-layer
  means). The base alone is 0.31 / 0.25.
- Static per-expert allocation (step 2c) beat REAP-75 + KEEP by 0.20 KL at equal memory, but the
  ablation (2c, 2d) traced almost all of it to selection; a static 1-bit tier added 0.025. The
  routed version has to beat that static frontier to carry the paper.

### Tooling

- `keep_quant.py --residual-of DIR`: the nested residual tier (`quant/e8-nest-r1`, all 192 experts
  of REAP-50, running).
- `local_eval.py --dynamic BASE+RESIDUAL --tau NAME=t --tau-drop NAME=t --cache-k NAME=K
  --cache-warm NPZ`: per-slot tiers in the forward, with the LRU simulated in token order per
  record (`residual_cache`, unit-tested). Writes `tier-counts.npy` per variant: per layer and
  expert, slots and gate mass at each tier, and admissions.

### The experiments that decide it

1. Endpoints: all tier 2 (tau 0), all tier 1 (tau 1e9). Must land near flat 2-bit (0.747) and
   flat 1-bit (1.358).
2. Routed tiers, everything resident (no SSD): tau sweep. Bytes read per token against KL,
   compared with static allocations at equal bytes.
3. Routed tiers with offload: K in {48, 96} residuals per layer, tau in {0.2, 0.3}, warm-started
   with the pruned-model hot set. Resident units = 192 + K. The bar: beat prec-288 (0.809 at 288
   units, nothing streamed) at 288 units with an SSD rate the M4 Max sustains (about 5 GB/s
   sequential; well under that at 30 tok/s).
4. Per-token tier 0 (tau_drop): does skipping low-gate slots cost less KL than static pruning
   saves?

### Observation: two routing regimes by depth

Share of REAP-50's routed mass (pruned-model capture) held by each layer's top 96 experts:
layers 0–19 keep 0.72–0.83, layers 20–39 keep 0.94–0.98 (layer 32 0.86 is the one dip). Early
layers route diffusely, late layers concentrate. The step-3 sidecar fit sees the same split: in
layers 4–6 the 96-expert 2-bit student misses 25–31% of the teacher MoE's output energy on held-out
tokens, and a rank-256 linear sidecar removes only 9–15% of it, while in layers 0–3 the gap is
3–11% and the sidecar removes 36–76%. A fixed per-layer budget cannot follow this; per-token tiers
can, since a diffuse layer simply routes more of its slots to the base tier.

### Headline progress rubric

Every major status update reports progress to the headline with this fixed rubric (weights sum
to 100; partial credit only for measured progress).

| # | Item | Weight | Score | State (2026-11-09) |
|---|---|---:|---:|---|
| 1 | Nested codes, all layers, validated | 10 | 10 | fit 40/40; error 0.102 gate/up, 0.084 down; end to end +0.053 KL against flat 2-bit |
| 2 | Per-token routed tiers, implemented and tested | 10 | 10 | `--dynamic/--tau/--tau-drop`, smoke-tested |
| 3 | Offload cache simulation and cost model | 10 | 10 | `residual_cache` (unit-tested), `routed_report.py` |
| 4 | Core result: routed beats the static frontier at equal resident memory, SSD rate sustainable | 25 | 0 | pass 1: R beats S by 0.068, but loses to 144 flat 2-bit experts by 0.025; D pending (pass 2b) |
| 5 | Unification: per-token tier 0 + expert pruning in one policy, frontier figure | 10 | 0 | tier 0 measured free at 7.7% of slots; no joint policy or figure yet |
| 6 | Real Apple Silicon streaming measured, not simulated | 10 | 0 | |
| 7 | Learned tier policy or trained nested codes | 10 | 0 | pod work, after 4 |
| 8 | Generality: a second MoE plus downstream benchmarks | 10 | 0 | |
| 9 | Paper draft | 5 | 1 | framing and experiment plan in this log |
| | **Total** | 100 | **31** | |

### 2026-11-09: prior art, plan, restart

- A novelty search found MatMoE (arXiv 2611.01704, 1 Nov 2026: nested codes with per-expert
  bit-width chosen from routing scores at run time), DynaExq (2511.15015: hot set at high precision
  with async promotion), DyMoE (2603.19172), HOBBIT and MC-MoE. Per-token precision with nested
  codes is no longer new by itself. What is left to claim: the nested residual as the offloaded
  unit, tier 0 in the same policy, and the regime at 2 bits and below. Details and the full plan to
  100%: [2026-11-09-routed-precision-plan.md](2026-11-09-routed-precision-plan.md).
- The session restart killed both fits (sidecar at layer 9, nested at layer 8). The nested fit
  resumed; the sidecar fit is sequential and restarted from layer 0. Jobs now run detached from
  `~/Models/v41-reap-run/jobs/` so a session exit cannot stop them.
- 1-bit codes are stored as uint8 from now on (`keep_quant.py`); the existing 48 layer files were
  converted in place and verified value-identical, freeing 38 GB.
- The first routed sweep swaps the τ = 0.41 variant for `s96`, a static hot set of 96 residuals with
  no admissions: the same codes and memory as the routed cache, so the comparison isolates routing.

- Headline changed (Jack, 2026-11-09): *Offload the Residual, Not the Expert: Unifying Pruning, Quantization and Offloading in MoE Inference*. The rubric and its weights are unchanged; item 4's
  comparison against a separate-copy cache is now the title's own claim.
- Survey of the older KEEP documents, this campaign's findings with bootstrap intervals, and the
  candidate levers X1–X7: [2026-11-09-keep-findings-and-levers.md](2026-11-09-keep-findings-and-levers.md).
  Bootstrap (positions, paired): selection on pruned routing −0.177 KL [−0.200, −0.148]; second
  prune pass −0.024 [−0.034, −0.012]; static allocator at 288 against 144 at 2 bits −0.003
  [−0.009, +0.005].
- Harness additions for W4: separate-copy high tier (`--dynamic BASE|HI`), oracle cache policy,
  projection-level rungs (`--hi-proj`), admission traces (`--trace`), expert masks under
  `--dynamic`, and `routed_report.py --compare`. Passes 2, 2b and 3 are queued behind pass 1
  (`~/Models/v41-reap-run/jobs/queue2.sh`, 24 variants, about 11 h).

### 2026-11-09: X1 on a pod (prune twice at the first cut), and X2 closed

- X2 (the pruned model's better tool-use NLL) is an artifact, settled from the pod's stored
  outputs: the advantage exists only on positions where the teacher is surprised by off-format text
  inside the records. Details in the findings note.
- X1 started. Design: mask the full model's router to the top 288 experts by total routed mass
  (`x1/fm-288.json`, 97.5% of mass), run calibration on that masked model (`collect.py --mask`, 2 of
  24 sessions, about 256K tokens), keep the top 192 by the new mass (`pt-192`), repeat once to 96
  (`pt-96`), then score five students against the teacher on 54 eval sessions: shipped REAP-50,
  one-pass total mass at 192 (`fm-192`, the criterion alone; it overlaps REAP-50 by 76.5%),
  `pt-192`, the Mac's 96 inside REAP-50 (`pm-96`), and `pt-96`. Script: `x1_pod.sh`.
- Runpod, US-NE-1 (H100 SXM Low there, RTX PRO 6000 as fallback, CPU pods available): network volume
  `yqazs7aqe6` (600 GB) and CPU pod `hd0czk7klgonpi` (4 vCPU, $0.12/h) for the 510 GB fetch, the
  wheels and the inputs, so the GPU pod starts on GPU work. Estimated GPU time 6.5–7.5 h at
  $3.99/h. The volume stays for W4b.

### 2026-11-09: step-3 sidecars closed (worse end to end), and an eval-set caveat

- The 4,096-token sidecar fit (rank 256, 62 records balanced over 13 domains, about 150K tokens)
  stopped at layer 15 of 40: peak RSS 81.8 GB passed the 80 GB guard right after layer 14. Layers
  1 and 14 are the two Engram layers; the pages their lookups touch stayed mapped. `sidecar_fit.py`
  now unmaps each table after its layer and resumes from saved layers (resume not yet exercised).
- Before resuming, the 15 fitted layers (0–14) were scored alone (`evals/step3-sidecar-l15`;
  `local_eval.py --sidecar` now accepts a partial fit). Against the same 96 experts at 2 bits
  without sidecars: KL 1.238 against 1.039, top-1 0.705 against 0.733; paired bootstrap
  **+0.199 [+0.164, +0.219]**. The median per-position change is 0.0006. The harm is in
  english-prose (+0.71, +1.15), english-chat (+0.34, +0.59) and business (+0.37, +0.48); the code,
  agentic and tool-use sessions move by −0.11 to +0.12.
- Layer-local holdout numbers said the opposite (holdout residual energy down 8–19% in the
  large-gap layers, 79–93% in layers 8, 10 and 14). Cause not established. Energy-weighted error
  on the fit's own records is not KL; an application mismatch between fit and eval is not excluded,
  though it would be expected to hurt every domain. Same shape as the GLM-4.5-Air sidecar: a proxy
  win that failed end to end. The fit is not resumed and the lever is closed.
- Eval-set caveat found on the way: four of the 22 Mac eval sessions (`eval-00000`, `-00001`,
  `-00004`, `-00005`; both agentic-python and both agentic-rust) have identical first 1,024
  tokens, a shared system prompt. At 1,024 tokens the Mac set is 19 distinct texts and those two
  domains measure the same one. Paired differences between variants are unaffected; domain means
  at 1,024 tokens overweight that prompt. The 4,096-token confirmation pass (W4 pass 4) is where
  those sessions diverge.

### 2026-11-09: X1 running on an H100, after 70 lost minutes

- GPU pod `q8vcl6ob0hf0lk` (H100 SXM, US-NE-1, $3.99/h) created 01:01 UTC once the fetch verified;
  the CPU pod was terminated. Calibration on the 288-expert masked model started 02:31 UTC. The 90
  minutes between cost about $6 and did no GPU work:
  - 15 minutes: my readiness probe compared the proxy's reply to a prefix, and the reply starts
    with a terminal escape sequence, so it waited out all its tries on a pod that was already up.
  - One failed launch: the Ubuntu 24.04 image marks the system Python as externally managed
    (`PIP_BREAK_SYSTEM_PACKAGES=1` in `x1_pod.sh`).
  - 50 minutes: the smoke test sat at 0% GPU. The network volume reads one stream at about
    120 MB/s and a random 256-byte row in 2–3 ms; the loader reads shards through one mmap and
    the Engram lookup faults in rows one page at a time. Sixteen parallel streams read 3.2 GB/s
    across files and 1.1 GB/s inside one file.
- Fix: container disk raised from 50 to 560 GB (restarts the container), and `x1_pod.sh` copies
  the checkpoint to the container's own disk with 16 parallel `cp` and checks the byte total
  (`LOCAL_HF=/root/v41-hf`; `pod.sh` and `x1_pod.sh` take `HF` from the environment). The copy took
  7 minutes for 476 GB. Smoke then passed: 16,384 tokens in 447 s, 48.1 GB peak GPU memory.
- For W4b and any later pod on this volume: budget the 7-minute copy, create the pod with a
  560 GB container disk from the start, and probe readiness by substring.

### 2026-11-09: routed sweep, pass 1 (192 bases, 22 sessions, first 1,024 tokens)

Nested fit finished: 40 of 40 layers, total error 0.102 (gate, up) and 0.084 (down) against 0.090
for the flat 2-bit fit. Sweep: 12,415 s forward, peak RSS 64.1 GB. Units are 1-bit experts per
layer; SSD rate is at 30 tok/s.

| Variant | Units | KL | Top-1 | Tier-2 slots | SSD GB/s |
|---|---:|---:|---:|---:|---:|
| all2: every residual resident, all slots tier 2 | 384 | 0.800 | 0.779 | 100% | 0 |
| t22 (M): all resident, tier 2 at gate ≥ 0.22 | 384 | 0.822 | 0.767 | 46% | 0 |
| d10: t22 plus tier 0 below gate 0.1 | 384 | 0.823 | 0.763 | 46% (7.7% dropped) | 0 |
| c96t2 (R): LRU of 96 residuals, τ 0.2 | 288 | 0.832 | 0.772 | 90% | 0.83 |
| c96t3 (R): LRU of 96, τ 0.3 | 288 | 0.837 | 0.771 | 84% | 0.28 |
| s96 (S): static hot set of 96 | 288 | 0.905 | 0.753 | 75% | 0 |
| c48t2 (R): LRU of 48, τ 0.2 | 240 | 0.876 | 0.758 | 76% | 2.72 |
| all1: base only | 192 | 1.358 | 0.637 | 0% | 0 |

Static references from step 2, same sessions: 192 experts at flat 2 bits 0.747 (384 units);
144 experts at flat 2 bits 0.812 (288 units); 96 at flat 2 bits 1.039 (192 units).

Paired block bootstrap (`routed_report.py --compare`):

| A against B | KL(A) − KL(B) | 95% interval |
|---|---:|---|
| R c96t3 against S s96, both 288 units | −0.068 | −0.077 to −0.060 |
| R c96t2 against S s96 | −0.073 | −0.084 to −0.065 |
| R c48t2 (240 units) against S s96 (288 units) | −0.029 | −0.042 to −0.022 |
| R c96t3 against 144 flat 2-bit experts, both 288 units | **+0.025** | +0.009 to +0.051 |
| R c96t2 against 144 flat 2-bit experts | +0.020 | +0.003 to +0.045 |
| all2 against 192 flat 2-bit experts (the nesting penalty) | +0.053 | +0.039 to +0.071 |
| R c96t2 (288 units) against all2 (384 units) | +0.032 | +0.023 to +0.037 |
| t22 against all2 | +0.022 | +0.015 to +0.031 |
| d10 against t22 (tier 0 on 7.7% of slots) | +0.001 | −0.002 to +0.004 |

What it says:

- Routing the cache works. With the same codes and the same memory, an LRU with a gate threshold
  beats a fixed hot set by 0.07 KL, and a cache of 48 beats a fixed set of 96. At τ 0.3 the cost
  is 9.5 MB per token, 0.28 GB/s, a ninth of the 2.5 GB/s ceiling.
- It does not yet beat plain pruning. At 288 units, 144 experts at flat 2 bits score 0.812 and the
  routed cache 0.832–0.837. The nesting penalty (+0.053) is larger than that gap: with a flat-quality
  high tier the routed cache would be ahead. This is the argument for a separate-copy high tier,
  which pass 2b measures, and for the 2-bit-base ladder (X5).
- Tier 0 is free: dropping the 7.7% of slots with gate below 0.1 changes KL by 0.001 and cuts the
  expert bytes read per token by 5%.
- Item 4's 10 points need R to beat D as well, so the score waits for pass 2b. Missing static
  point: 120 experts at flat 2 bits (240 units), to compare with c48 directly.

### 2026-11-09: X1 result, re-base decision, plan revision 2

- X1 finished in 1.6 h of GPU work (smoke 02:23 UTC, scores 03:59 UTC). Domain-mean KL on 54 eval
  sessions: shipped REAP-50 0.716, one-pass total mass 0.532, prune-twice 0.546; at 96 experts the
  Mac's pick 1.163 and prune-twice 1.154. Total mass against REAP-50: −0.184, session-level 95%
  interval −0.277 to −0.128, better in 36 of 54 sessions; english-chat is worse (0.448 → 0.572).
  Tables, intervals and the reading are in the
  [findings note](2026-11-09-keep-findings-and-levers.md), section 5.1. Outputs:
  `~/Models/v41-reap-run/x1/pod/` (scores, tables, per-position eval files, 2.4 GB).
- Follow-up on the same pod while it was hot (`x1b_pod.sh`): 96 experts in one pass on full-model
  routing (`fm96`, keeps 74.0% of mass) against two passes through `fm192` (`pf96`, 81.2% of the
  192's mass).
- Jack approved re-basing on the total-mass 192 and asked for a disk cleanup. Export running on
  the pod's CPU alongside the follow-up: `prune.py --skip-shard` (new, tested) leaves out the two
  Engram shards, which are byte-identical to REAP-50's, so the export is 152 GB instead of 341.
  Network volume grown 600 → 680 GB to hold it (volumes cannot shrink; $48 a month until deleted).
- Deleted on the Mac with Jack's confirmation, 373 GB: `~/Models/v41-reap75` (on Hugging Face),
  `quant/e8p-prunedimp`, `sidecar/codes-prunedmass-96`, `sidecar/mac-4k-r256`, `sidecar/smoke`.
- Plan revised ([revision 2](2026-11-09-routed-precision-plan.md)): re-base first (WR), main
  ladder 2-bit base + 1-bit residual, W4b redefined, sidecars and prune-twice closed, X3 demoted,
  X8 (domain-balanced selection) added, operating rules for pods and evals. Five predictions are
  registered in the findings note, section 6, before the runs that test them.

### 2026-11-09, late: full volume, cancelled sweeps, plan revision 3

- Both pod jobs (the 96-expert follow-up and the export) died silently between 04:35 and 04:43
  UTC. Cause: the network volume filled. I sized it in GB against `du` figures in GiB (476 GiB of
  model is 511 GB; the 152 GiB export is 163 GB), so 680 GB was 6 GB short. A full volume here
  writes zero-length files and returns success, so neither job logged an error, and my status
  check (`pgrep -f prune.py` inside an ssh command that itself contains that string) kept reporting
  the export as running. About 15 minutes of idle H100.
- Recovery: partial export deleted, volume grown to 720 GB ($50 a month), follow-up eval resumed
  alone at 04:51 UTC from its 8 finished sessions (files written before the volume filled were
  checked: complete). The export moves to a CPU pod.
- Jack cut the 1-bit base altogether. Passes 2b and 3 cancelled; pass 2 finishes.
- Where the selection gain sits, and the checks behind it: findings note, section 5.1.
- Plan [revision 3](2026-11-09-routed-precision-plan.md): thesis "pruning is a frozen cache",
  target architecture, a critique of the approach, ten falsifiable conjectures, eleven ordered
  experiments, and a proposed working title conditional on C1. The first experiment (E1) needs no
  new fits and runs after pass 2.
