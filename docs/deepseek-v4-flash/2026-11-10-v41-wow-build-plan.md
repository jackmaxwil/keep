# V4.1 on a 128 GB Mac, generating: build plan

Written 2026-11-09/10 from three read-only surveys (repo inventory, sizing, prior art), three rival
plans drafted from different premises, and a red-team of each against the code. Everything marked
"estimate" is a planner's estimate and has not been measured. Days are working days.

**The bar.** A stranger with a 128 GB Mac runs one command and watches the full
DeepSeek-V4.1-Flash, with all 384 experts per layer reachable, do an agentic coding task at a speed
they would use. Nothing generates V4.1 in this repo today.

**The claim to build toward.** The full model already runs on 128 GB (oMLX), at 4.2–5.6 tok/s,
because every cache miss waits for the SSD. The fully resident build (Rapid-MLX) does 28 tok/s
without the drafter and needs 256 GB. Our claim is the trade in between: a miss that costs no
time, at a measured and bounded loss of exactness. It is a speed and evidence claim, not a new
idea; section 7 says which ideas are already shipped.

## 1. The three questions

### 1.1 How much of a V4.1 decoder already exists here?

Source: repo inventory, 2026-11-09, read file by file against the reference
(`~/Models/v41-reap50/inference/model.py`).

- No V4.1 decoder. A complete MLX decoder for the older V4-Flash exists
  (`src/ramp/models/deepseek_v4_flash_adapter.py`, `src/keep/quality/dsv4_mtp_runtime.py`):
  prefill, KV cache, greedy decode, drafter generation with rollback. Correct on toys; never shown
  fast (its one decode figure, 0.24 tok/s, was paging on an M5 Max).
- Reusable for V4.1: router and SwiGLU (`adapter:913-1078`), RoPE, sinkhorn math and its fused
  Metal kernel (`src/ramp/kernels/hc_sinkhorn.py`), expert compute from 2-bit codes
  (`src/ramp/kernels/e8p_m1.py`, 202–206 GB/s on this Mac at V4-Flash shapes), FP8 residents as
  mxfp8 (`src/keep/convert/fp8_block.py`), native FP4 experts (`mx.gather_qmm`), `preadv` I/O
  (`src/keep/quality/dsv4_teacher_runner.py:640-760`), and a numeric oracle: exact CPU ports of all
  six reference kernels (`scripts/runpod/v41_reap/torch_kernel.py`).
- Missing: Engram in MLX entirely; V4.1 attention (compressed KV shared across layers, shared index
  keys, two-level candidate blocks); hyper-connection wiring (mix weights used one sublayer late);
  a cache layout with rollback; drafter changes; a loader; a sampler; the chat format; any server.
- The reference cannot run on a Mac as shipped (its kernels are CUDA-only) and decodes one token
  per call after the prompt, so chunked prefill and drafter verification have no reference to copy.
- Outside this repo the whole decoder exists twice, under permissive licences (section 7).

### 1.2 Where does decode time go?

Source: sizing survey, 2026-11-09, from the safetensors headers and the reference; no V4.1 decode
has been timed on any Mac in this repo.

| Read per decoded token, full model | GB |
|---|---:|
| Attention | 5.15 |
| Shared experts | 1.42 |
| Output head | 1.32 |
| Router, hyper-connections, norms, Engram projection | 0.63 |
| **Non-expert subtotal** | **8.52** |
| 240 routed experts at native FP4 / at 2.03 bits | 4.51 / 2.16 |
| **Total** | **13.0 / 10.7** |

- Ceiling at this Mac's measured read rate (409 GB/s on 2026-10-06, 474–477 on 2026-11-05):
  31–45 tok/s. Routed experts are 35% of bytes at native precision, not the 11% seen on V4-Flash
  in August; that lead does not transfer.
- With the drafter, one verify pass serves several tokens, so non-expert reads are shared and
  experts become the larger share. Break-even is about 2.2 accepted tokens per cycle (estimate);
  Rapid-MLX reports about 3.1.
- oMLX's 4–6 tok/s is consistent with waiting on about 74 expert loads per token (estimate from
  its hit rates). Removing the waits alone caps near 20 tok/s; the rest depends on staying on its
  fused kernels.
- Unknown until measured here: the time split by component, sinkhorn cost (80 calls per token),
  the indexer scan at long context.

### 1.3 What is resident besides routed experts?

Source: sizing survey, 2026-11-09.

| Item | GB |
|---|---:|
| All non-expert weights as stored (FP8) | 9.85 |
| The same expanded to bf16 | 16.7 |
| Drafter as stored | 7.9 |
| Context state at 64K tokens (3,200 bytes per token) | 0.21 |
| One routed expert: native FP4 / our 2.03-bit code / oMLX 3-bit | 0.0188 / 0.0090 / about 0.0176 |
| 192 experts per layer at 2.03 bits | 69 |
| 384 experts per layer at 2.03 bits | 138 |

- Memory goes almost entirely to routed experts. The Engram page cache is the open term: up to
  0.77 MB per new token, so as much as 50 GB at 64K if n-grams never repeat (upper bound).
- The "91 GB thrash" was measured on an M5 Max 128 GB on 2026-08-21, not on this M4 Max. Nobody
  has measured the ceiling here.
- At the 80 GB process cap about 160 experts per layer fit; about 250 need roughly 108 GB with
  residents and the drafter, which one planner puts at Metal's limit on this machine.
  The top 192 / 250 / 288 experts by routed mass cover 92.5% / 95.2% / 97.5% of routed mass.

## 2. Chosen architecture

**Build on oMLX as a pinned fork; plug in our cache policy, selection tables and (if it passes its
kill test) our 2-bit kernels.**

- oMLX (`~/Development/_refs/omlx`, commit `4213c328`, Apache-2.0 with MIT parts) already has the
  V4.1 decoder, the Engram reader, the drafter, an expert store with `pread` into slots, a server
  with tool-call parsing, and it reads the FP4/FP8 checkpoint already on this Mac.
- The change is a **compact router**: each layer keeps C experts in slot stacks and the router's
  weight and bias are sliced to those C rows. oMLX's fused kernels then pick the top 6 among
  residents and renormalise. That is the re-route, with no wait, no read and no extra sync. Demand
  for absent experts is read once per token from the full router and drives background admission.
- New code (estimate, about 300 lines plus four edits): `routed_cache.py` beside
  `omlx/patches/deepseek_v41/moe_offload.py`; later `e8p_moe.py` calling
  `src/ramp/kernels/e8p_m1.py`.
- Stock oMLX in its exact, waiting mode is both the day-1 measurement tool and the baseline.

Why the rivals lost:

- **Extend our own V4-Flash runtime.** Its own planner called it "a rewrite, and a fourth
  implementation": the adapter differs from V4.1 in attention, hyper-connection wiring, caches
  and drafter, and has no Engram. Twelve days to a first MLX token and about 47 to a demo, spent
  mostly on what two projects already ship, on a runtime with no clean speed measurement. Its
  E8P kernels and oracle survive as plug-ins.
- **Port the reference directly.** A torch decoder on CPU gives a first token in 3.5 days at
  20–80 s per token, and none of it carries into a fast runtime. Its one unique product, a
  generation oracle, is covered by stock oMLX from day 1 at about 5 tok/s. Kept from it: parity
  gates must be statistical (rounding alone moves KL by 0.1–0.3 per position between runs).
- **Rapid-MLX as the base.** The cleaner kernel host (9.4k lines against about 60k) and the faster
  resident path, but it has no expert store, refuses to start under 224 GiB, and needs a download
  of about 320 GB. It is the fallback if oMLX's fused path proves slow.

Red-team of the chosen plan:

- **Speed is unverified until day 4.** oMLX publishes no fully resident V4.1 number. The
  planner's estimate is 15–20 tok/s without the drafter.
- **Their codebase moves** (40 commits to the V4.1 directory since October; exact `mlx` pin;
  compiled kernels). Pin the commit and do not rebase before the demo.
- **Little of the system is ours.** What is: the zero-latency miss on a Mac, the policies in
  section 5, a denser code, and the measurements.
- **oMLX chose exactness on purpose**, and cites a result that pinning the hot half and zeroing the
  rest costs about 91% accuracy on a maths benchmark (source could not be verified; zeroing is not
  re-routing). Teacher-forced KL will not settle this. Generation has to.

## 3. Kill tests before any building (days 0–1)

| # | Test | Kills |
|---|---|---|
| K1 | Stock oMLX on this Mac, local pruned checkpoint: tok/s, hit rate, resident memory, and its no-miss floor. **Passed 2026-11-10:** the loader takes the 192-expert checkpoint; with oMLX's own offload at 48 of 192 experts it decodes at 3.3–4.5 tok/s (hit rate 78%, about 0.9 GB read per token) and is not on the fused path; with a fixed 48 or 64 experts and a sliced router it decodes at 30–32 tok/s in the same memory. | The base, if the no-miss floor is under about 10 tok/s: switch to Rapid-MLX's decode kernels. |
| K2 | `bench_e8p_m1` at 5120 × 2304 against oMLX's 2-bit affine kernel. **Passed 2026-11-10:** per routed expert the lattice kernel takes 0.037 ms, oMLX's affine 2-bit 0.047 and its FP4 0.042; the lattice expert is 9.0 MB against 11.1. | Our lattice kernel in the runtime: ship affine 2-bit instead. |
| K3 | Static 250 and 288 of 384 against the full model, teacher-forced, on the pod. **Done 2026-11-10:** KL 0.323 at 250 and 0.235 at 288 (0.532 at 192, 1.123 at 96), native precision, no cache. | Nothing; it sets how much the cache has to recover. At the 80 GB cap (about 160 experts) a fixed set costs roughly 0.6–0.7 by interpolation, so the cap matters more than the code density. |
| K4 | E1: expert cache with re-route against static sets, teacher-forced. **Passed 2026-11-10:** the cache beats the static set by 0.056 KL at 144 residents and 0.231 at 96 (pool of 192, 2 bits); a cache of 96 equals a static 144; the oracle gap (0.020) is all in the first 256 tokens. | The quality case for the cache, if it is not better than a static set. |
| K5 | 20 lines: an MLX array over a 1 GB memory-mapped file through the E8P kernel; check footprint and `mincore`. **Failed 2026-11-11** (`scripts/v41_build/k5_pagecache.py`, mlx 0.32.4): `mx.asarray(np.memmap, copy=False)` does alias the mapping (same address, no page touched), but a GPU kernel that reads 1 MiB of a 2 GiB aliased buffer brings all 131,072 pages into memory (0.79 s). One buffer over all experts would page in the whole pool. | Idea 3 (page cache), if it copies. Dead in the one-buffer form; one buffer per expert would need the routed experts known on the host before dispatch, which costs a sync per layer. |
| K6 | Offline, from stored routes: how well token ids predict experts; how well a prompt's demand predicts the rest of the session; recall of the drafter's 256-d head. | Ideas 2, 6 and 7 respectively. |

## 4. Critical path

**To the first generated token on this Mac: day 1.** Stock oMLX on `~/Models/v41-reap50` (already
on disk, no download). That token comes from the pruned 192-expert model in exact mode. It is a
measurement, not the result.

**To the first no-wait token: day 4** (estimate; I would plan 4–6).

| Day | Work | Done when |
|---|---|---|
| 1 | K1, K2, K5; parity of oMLX against `torch_kernel.py` on the local model | Top-1 match on 200 positions and KL under 0.01 against the oracle |
| 2–4 | Compact router with a static resident set | Bit-identical to stock with C = all; all 40 layers stay on the fused path; at least 12 tok/s without the drafter |

**To the recorded demo: about 19 days with no slack; plan 19–26.**

| Days | Work |
|---|---|
| 5–7 | Background admission, warm tables from our selection, drafter on |
| in parallel | Pod or download: the other 192 experts and the full router rows (decision 2) |
| 8–11 | Lattice-code expert path and the pool format (only if K2 passed) |
| 12–13 | The 384 pool end to end |
| 14–16 | Generation benchmarks, baseline (stock oMLX at equal memory), two ablations |
| 17–19 | Wrapper script, model card, recording |

## 5. The seven ideas, placed

None is on the path to the first token. Ideas 4 and 5 share one mechanism. Two planners placed
them independently; where they disagreed the more sceptical reading is kept.

| # | Idea | Attaches at (oMLX fork) | Cheapest kill | When |
|---|---|---|---|---|
| 1 | Drafter with a no-wait cache | `mtp.py` and the verify path in `decode_fused.py`; admission only between draft-verify cycles so one cycle sees one resident set. Needs a multi-row lattice kernel. | Day 7: tokens per cycle over cycle time under 1.5× single-token decode. | Demo |
| 2 | Drafted tokens as lookahead | Admission queue in `routed_cache.py`; needs a token-id → expert prior. | K6: recall of the next four tokens' experts against plain decayed counts. | After |
| 3 | OS page cache as the cache | A native module with a no-copy Metal buffer over a mapping. | K5, as a standalone Metal test: dead if Metal wires the pages or a fault stalls the GPU. | After |
| 4 | Exactness inside tool calls | `output_parser.py` already tracks the tool-call state; the exact path is the existing load-and-wait. Drafter depth 1 inside a call. | 50 tool-call prompts at C = 160: cut if re-routed validity equals exact. | Demo, conditional (1–2 d) |
| 5 | Exact-to-fast dial | The routing weight that missed, scored in `routed_cache.py`; same exact path as 4. | Offline now: the harness already takes a threshold, so plot KL against the share of tokens that wait. A flat curve kills the runtime dial. | Curve before the demo; runtime after |
| 6 | Prompt as profile | Prefill branch of the MoE call: accumulate the full router's weight per expert, bulk-load before decode. | K6: mass covered over the next 64 generated tokens, against the static set plus counts. | Demo if it passes (1 d) |
| 7 | Shortlist head | `head.py` plus the drafter's 256-d head. | Day-4 profile: cut if the head is under 5% of token time. | After |

What the code says about them:

- **Idea 1 is the strongest, with a measurement behind it.** oMLX has allowed the drafter with
  offload since 2026-09-23, and it gains only 29% (6.6 → 8.5 tok/s at 81–89% acceptance), because a
  four-token verify asks for four times the experts and waits for them (333 ms against 151 ms).
  That wait is exactly what a no-wait cache removes.
- **Idea 3 is harder than it looked.** Neither runtime computes on mapped pages; both copy into
  allocated buffers, and Rapid-MLX's loader warns that a cold page-in inside a GPU command trips
  Metal's watchdog. A safe partial version: use `mincore` only to rank absent experts that are
  already in the page cache, so they are admitted first.
- **Ideas 2 and 7 are weak as stated.** A verify pass already computes the drafted tokens, so
  Engram lookahead saves under 1% of wall time, and oMLX's own router prefetch ran 2–15% slower.
  The drafter's 256-d head is a bias on the previous token and never sees the hidden state, so a
  shortlist would need a newly trained projection.

## 6. The inventory, placed

Exists = in this repo. oMLX = reuse from the fork (Apache-2.0/MIT). Build = new work. Cut = not for
the demo, with the reason.

| # | Item | Placement |
|---|---|---|
| 1 | Block port | oMLX `language.py` |
| 2 | Engram | oMLX `engram.py`, `storage.py` |
| 3 | FP8/FP4 loaders | oMLX `loading.py`, `convert.py` |
| 4 | Incremental decode, KV, prefill | oMLX `decode_fused.py`, `cache.py` |
| 5 | Tokenizer, chat, tool calls, sampling | oMLX `encoding.py`, `tool_parser.py` |
| 6 | Drafter | oMLX `dspark.py`, `mtp.py`; build the verify check on the compact router |
| 7 | Parity harness | Build on `torch_kernel.py`, `local_parity.py`; statistical gates |
| 8 | Speed: experts from 2-bit codes, attention kernels | Build `e8p_moe.py` on `e8p_m1.py` if K2 passes; attention kernels from oMLX |
| 9 | On-disk pool format | Build: stacked safetensors readable by oMLX's row reads |
| 10 | Resident-set manager | Build on oMLX `_ExpertSlots` |
| 11 | Admission path | Build: background, applied between tokens |
| 12 | Router changes | Build: compact router. Gate floor cut until measured in the runtime |
| 13 | Prefill behaviour | Comes with the compact router |
| 14 | Memory-pressure response | oMLX's memory guard; shrink C |
| 15 | I/O sharing with Engram | oMLX: the expert pool is already separate |
| 16 | Warm start, persistence | Build: save and load slot ids |
| 17 | Hot tier above 2 bits | Cut: unproven, and one tier is enough for the demo |
| 18 | 2-bit codes for all 384 | Build: fit on the Mac after fetching the other 192 experts, or on a pod (decision 2) |
| 19 | Importance statistics, router rows | Exists for all 384 (statistics); router rows from the pod volume |
| 20 | Non-expert weight format | oMLX: source FP8 as stored (9.85 GB) |
| 21 | Selection and warm tables | Exists (`rerank.py`, total mass); regenerate for the pool |
| 22 | Packaging, download | Build after the demo: codes pool on Hugging Face (public: decision 4) |
| 23 | Licence review | Half a day: MIT model, Apache-2.0 fork notice |
| 24 | Eval set version 2 | Build (paper track, not the demo) |
| 25 | Generation benchmarks | Build: tool-call validity, code tasks that execute, one agentic task |
| 26 | Baselines | Stock oMLX at equal memory, run here; Rapid-MLX numbers quoted (216 GB does not fit) |
| 27 | Hardware measurements | Build: decode ms per token by component, memory, SSD rate, swap count |
| 28 | Smaller-RAM emulation | Cut until after the demo; the slot count gives it later |
| 29 | Ablations | Two: static set against cache, wait against re-route |
| 30 | Second model | Cut from the demo; stays in the paper plan |
| 31 | Reproducibility | Pinned fork commit, checksums, one script |
| 32 | CLI and server | oMLX `omlx serve` |
| 33 | One-command install | Build: wrapper script |
| 34 | Support matrix, CI | Cut to "128 GB M-series" |
| 35 | Recorded demo | Build |
| 36 | Cards, paper | Model card for the demo; paper separately |
| 37 | Gate: cache beats static | K4, passed 2026-11-10 |
| 38 | Gate: cache over 384 recovers the pruning loss | Teacher-forced in our harness once item 18 exists; then in generation |
| 39 | Gate: speed bound | K1, then the day-4 measurement |
| 40 | Gate: compute from 2-bit codes fast enough | K2 |

## 7. Prior art

| Project or paper | What it does | Reuse | Licence |
|---|---|---|---|
| DeepSeek V4.1-Flash | Model, torch reference, Engram, drafter | Reference and oracle | MIT, weights included |
| oMLX | V4.1 on MLX; expert store on SSD, waits on a miss; drafter with offload; 4.2–5.6 tok/s on 128 GB | The base | Apache-2.0, MIT parts |
| Rapid-MLX | V4.1 fully resident: 216 GB, KL 0.26, 28 tok/s plain, 33–49 with the drafter; Engram prefetch | Fallback kernels | Apache-2.0 |
| ds4 | C/Metal engine with a slab expert cache; waits on a miss | Design reading | MIT |
| llama.cpp | V4.1 merged 2026-10-25; IQ2_XXS is an E8-lattice 2-bit format | — | MIT |
| Cache-prior (2412.00099), SMoE (2508.18983), MoE-CORE (2610.01950) | Re-route or substitute a miss using router scores | The rule | papers |
| SwapMoE (2308.15030) | Dynamic resident set against static pruning at equal memory, small models | The comparison | paper |
| SliceMoE (2512.12990), MatMoE (2611.01704) | Nested bit slices; nested codes chosen by routing score | — | papers |
| HOBBIT, DynaExq, DyMoE, Fate, AdapMoE, MoE-Infinity | Low-precision fallback, asynchronous promotion, prefetch | Baselines | papers |
| REAP (2510.13999) | Per-route pruning score; deliberately avoids frequency weighting | Already implemented here | Apache-2.0 |

Which of our ideas someone has already shipped:

- **Shipped:** the full model on 128 GB with experts on SSD (oMLX, ds4); drafter with offload
  (oMLX); 2-bit V4.1 builds (three of them); Engram on SSD with prefetch (Rapid-MLX).
- **Published, not shipped on a Mac:** re-routing a miss; nested precision caches; a dynamic set
  against static pruning.
- **Not found:** a miss that costs no time in a Mac runtime at this scale; exactness switched by
  grammar state; the exact-to-fast curve; a residency check on the page cache with re-route;
  drafted tokens as lookahead; quality against pruning at V4 scale; selection by total routed mass.

## 8. Decisions for Jack

1. **Memory cap. Decided 2026-11-09: stay at 80 GB for now** (the Mac is under load from other
   sessions; it can be raised later, and pods are the route for anything that needs more). The
   build therefore targets C = 160 experts per layer on the Mac. Anything that needs about 250
   resident (roughly 108 GB) is measured teacher-forced on a pod until the cap changes.
2. **Source of the other 192 experts.** (a) Fetch their FP4 weights from the pod volume (about
   144 GiB) and fit our codes on the Mac overnight; (b) fit on a pod (spend; the fit has only run
   on the Mac GPU); (c) download oMLX's 3-bit build (about 331 GiB), which gives a 384-expert
   token soonest. Free disk is about 430 GiB, so not all of them.
3. **Fork.** Whether a pinned oMLX fork is acceptable as the shipped form, and whether it is public.
4. **Claim.** A model that is not exact is the product. The demo should not be published before
   the generation check.
5. **Housekeeping.** A planner's Python call created `~/Development/_refs/omlx/.venv` (1.8 GB),
   `uv.lock` and an egg-info directory in the oMLX clone. Remove or keep.
6. **Paper title.** Still open; this plan does not depend on it.

## 9. Runbook for K1 (first generated token, stock oMLX)

Checked 2026-11-09 by reading the clone and running only `omlx --help`: the environment at
`~/Development/_refs/omlx/.venv` imports (`mlx` 0.32.4, `omlx` 0.7.4.dev1). Not yet done: the
native kernels are not built, and no model has been loaded.

Run when no other heavy job is on the Mac (E1 holds it tonight):

1. `make kernels` in the clone (oMLX's fused paths need them; a plain install skips them).
2. A model directory containing a link to `~/Models/v41-reap50`. oMLX's V4.1 loader accepts the
   original checkpoint format; whether it accepts a 192-expert count is the first thing this run
   tests.
3. Model settings: `moe_expert_offload_enabled`, `moe_expert_offload_resident_fraction` 0.375
   (72 of 192 experts at native FP4, about 54 GB), `deepseek_v41_engram_ssd_offload`; memory guard
   below 80 GB. oMLX's own figures show peak footprint about 12 GiB above its Metal-active size, so
   start lower (0.25) if the first load comes close to the cap.
4. `omlx serve --model-dir …`, then two 64-token chat completions on one prompt (cold, then warm)
   and the same prompt replayed for the no-miss floor.
5. Record: decode tok/s, prefill tok/s, hit rate (oMLX logs it per request), peak footprint, and
   whether every layer stayed on the fused path.

Reference points from oMLX's own document: on a smaller MoE model it measures 122.5 tok/s fully
resident, 59.4 at 50% residency and 40.1 at 25%, so waiting on misses costs it a factor of two to
three there; on V4.1 it says decode "is bound by miss latency at one to two misses per layer per
token".

Not runnable yet: the offline checks for ideas 2 and 6 (K6) need per-token router demand, which
the harness does not dump; the stored routes are a 2,048-token sample without token ids.

### K1 result (2026-11-10)

Run on the M4 Max from the pinned clone, native FP4 experts, Engram on SSD, greedy decoding, one
request at a time. Scripts: `scripts/v41_build/omlx_counters.py` (reads oMLX's expert-slot
counters, which its server does not log for V4.1) and `scripts/v41_build/omlx_compact.py` (a
fixed resident set through a linked model directory and three read-side patches; the clone is not
modified).

| Configuration | Resident experts | Decode tok/s | Prefill tok/s | Memory after load (peak) |
|---|---:|---:|---:|---:|
| oMLX expert offload, waits on a miss | 48 of 192 | 3.3–4.5 | 6 | 45.4 GiB (54) |
| fixed set, sliced router | 48 | 30–32 | 150 | 45.4 GiB (55) |
| fixed set, sliced router | 64 | 30–31 | 145 | 56.6 GiB (67) |

- oMLX's offload path: 78% hit rate on a 256-token completion, about 47 misses per token at
  18.8 MB each, so roughly 0.9 GB read per token; an exact replay of a 45-token exchange still
  misses 39% of lookups at 48 slots. Its offloaded experts are slot wrappers that the fused decode
  kernels do not accept (`decode_moe.supported`), so this path is slow even on a hit.
- The fixed set runs on the fused kernels because the experts are plain stacked arrays and
  `n_routed_experts` is the resident count. This is the compact router's static case, and it is
  seven times the speed of waiting on a miss at equal memory. The estimate of 15–20 tok/s was low.
- Consequence for the cache: keep the experts as one stacked array of C rows per projection and
  overwrite a row in place on admission, with the router row rewritten in the same step. Do not
  build on oMLX's slot wrapper.
- Memory: about 11.8 GiB for everything but routed experts, 0.70 GiB per expert at FP4, and about
  10 GiB of extra peak during load. Under the 80 GB cap that is about 72 experts at FP4 and about
  155 at 2 bits.
- Quality at these sizes is poor, as the teacher-forced numbers predict: 48 experts give clipped
  but on-topic text, and 64 fall into a loop within a 1,024-token greedy completion. Speed is
  settled; quality needs the cache and 2-bit experts.
- Side effect on the machine: the server wrote `~/.omlx/bin/omlx-cluster-python`, a link to the
  clone's Python.

### No-wait cache, first version (2026-11-10)

`scripts/v41_build/omlx_routed.py`, on the fixed-set base above, clone unmodified. 64 resident
experts per layer at native FP4, drawn from the pruned pool of 192; admission at gate weight 0.2,
4 spare slots per layer, least recently demanded evicted.

How a miss costs no time:

- Every layer also scores the whole pool's router rows on its input. The demanded experts ride in
  the token's own graph, so reading them two calls later waits for nothing.
- The router is the switch. An evicted slot's bias is set so nothing routes to it; reader threads
  overwrite its rows in place only after every call built before that has finished; its router
  row becomes the new expert only after the bytes are in. A token never waits on a read and never
  runs a half-written expert. An admitted expert serves from about three tokens after the demand.
- Stored FP4 bytes are the array's bytes, so an install is six positional reads straight into
  the slot, 18.8 MB, with no copy.

| Run | Decode tok/s | Notes |
|---|---:|---|
| fixed 64, no demand tracking | 30–31 | from K1 |
| cache code, admissions off | 26.6–28.2 | output identical to fixed 64 over 1,024 tokens; demand scoring costs 10–14% |
| cache on | 22.0–26.1 | 8.3 installs per token (156 MB), 75% of demanded slots resident |
| oMLX offload, exact, 48 slots | 4.5 | reference text below |

- Checks: with admissions off the output equals the fixed set's, token for token. With them on,
  every routed slot in the first and last layer matched its expert's stored bytes at 20
  checkpoints across 12,785 installs. A torn read in flight is excluded by the ordering rule, not
  by a measurement.
- Quality is not there at this size. On a 1,024-token coding prompt the exact pool of 192 writes a
  correct script; fixed 64 and cache 64 both leave the reference within a few words and end in a
  loop. The cache's text is more fluent early on, which is not evidence. 64 experts is what FP4
  allows under the 80 GB cap; E1 puts the useful range at 96–144 residents.
- So the next build item is 2-bit experts in this runtime (K2, then lattice codes as slot arrays):
  about 155 residents under the same cap. Also open: demand scoring through oMLX's own gate kernel
  to recover the 10–14%; 23% of wanted admissions found no free slot at 4 spares; prefill runs
  through the same code but was only exercised on short prompts.

### K2 result (2026-11-10)

`scripts/v41_build/bench_expert_formats.py`, M4 Max, one process, MLX 0.32.4, 64 experts, 40
chained MoEs per eval, three runs (medians agree within 5%). oMLX rows time its fused decode MoE
(router, routed experts, shared expert) on random weights; the lattice row times this repo's
`e8p_m1` kernel on layer 20's real 2-bit codes, routed experts only. Cost per expert is the
difference between top-6 and top-1, over five.

| Routed-expert format | Bits per weight | Expert, MB | MoE at top-6, ms | Per expert, ms | Fixed, ms |
|---|---:|---:|---:|---:|---:|
| oMLX MXFP4 (checkpoint format) | 4.25 | 18.8 | 0.39–0.41 | 0.042–0.045 | 0.145 |
| oMLX affine 2-bit, group 64 | 2.5 | 11.1 | 0.41 | 0.046–0.048 | 0.124–0.134 |
| oMLX affine 3-bit, group 64 | 3.5 | 15.5 | 0.35–0.36 | 0.035–0.036 | 0.140–0.145 |
| E8P lattice 2-bit (routed only) | 2.04 | 9.0 | 0.29 | 0.036–0.038 | 0.065–0.069 |

- The lattice kernel is not slower per expert than any oMLX format, so K2 does not kill it. It
  is 3.3 times this repo's generic kernel at this shape (0.95 ms).
- What it buys is residents, not speed: under the 80 GB cap about 155 lattice experts fit
  against about 127 at affine 2-bit. E1 puts a cache of 144 within 0.009 KL of the whole pool of
  192, so the lattice format clears that line and affine 2-bit does not.
- Not measured: the lattice kernel inside oMLX's decode step. The router and shared expert then
  run as their own dispatches; from the fixed costs above the MoE would take about 0.43 ms
  against 0.39–0.41, roughly 3–5% of a token.
- Side note: oMLX's 3-bit affine MoE is its fastest, ahead of FP4.
- Next build item: lattice experts as slot arrays in the cache. That needs the codes stored as
  raw per-expert bytes (they are `.npz` today), the routed half of the fused MoE replaced by
  `e8p_m1`, and codes for the experts outside the pruned pool.

### Lattice experts in the cache (2026-11-10)

`scripts/v41_build/omlx_lattice.py`. oMLX (pinned clone, unmodified) loads a six-expert model
made by `omlx_compact.py make --experts 6`; after the load each layer's routed experts are
replaced by 144 lattice experts held as stacked code and scale arrays.

- Storage needed no conversion. The quantizer's `layer-NN.npz` files are stored uncompressed, so
  an expert's codes are a fixed byte range of the file and an install is a positional read into
  the slot's rows.
- The router keeps every row of the pool and scores all 192 once per token with oMLX's gate
  kernel. One small kernel takes two top-6's of those scores: among the residents (every other
  expert's bias is -1e30), which is what runs, and over the pool, which is the demand the policy
  reads two calls later. An install or eviction changes one bias and one entry of an
  expert-to-slot table. This replaced the separate demand matmul of the FP4 cache.
- Decode step: gate kernel, top-k kernel, `e8p_m1` for w1 and w3, oMLX's SwiGLU and act-quant
  kernel, `e8p_m1` for w2, then oMLX's shared-expert kernels, which add the routed sum. The
  shared up kernel runs with no routed column (`TOPK` 0).
- Prefill: oMLX's unfused MoE with the routed experts on this repo's sorted-route kernels.
- `omlx_routed.py` is now the policy with the FP4 storage as one subclass; the lattice cache is
  the other.

Self-check (`omlx_lattice.py check`, two layers of real codes, random routers and shared
experts): the decode step is within 0.3% of the same MoE computed from matrices decoded on the
host, the prefill path within 0.1% of the decode path, and after 83 installs no slot differs
from its stored bytes. The decode-step MoE takes 0.40–0.42 ms against 0.38–0.41 for oMLX's
fused FP4 MoE in the same harness. The lattice kernel is 1.7 times slower on fp32 activation
rows than on 16-bit ones (0.51 against 0.29 ms for the routed half), so it gets bf16.

Serving, M4 Max, pool of 192, 144 residents per layer, tau 0.2, 4 spares:

| | Decode, tok/s | |
|---|---:|---|
| Policy off (fixed 144) | 28.6–30.1 | text reproducible run to run |
| Cache on | 26.0–27.9 | 2.2 installs and 20 MB read per token; 96% of demanded slots resident |
| oMLX FP4, fixed 64 (K1) | 30–31 | |
| FP4 cache, 64 residents | 22–26 | 8.3 installs and 156 MB per token; 75% resident |

- Memory: 62 GiB steady, 68–69 GiB peak during the load, which takes about 25 s.
- Prefill: a 4,341-token prompt took 22.6 s and 39.1 s (192 and 111 tok/s, two runs, no KV
  reuse). Prompts of 38–64 tokens reach the first token in 1.4–2.3 s. A prefill is one forward
  call, so the cache cannot adapt inside it; over the runs with the long prompt 83% of demanded
  slots were resident.
- Policy cost: 1.5 ms of Python per token, and about 2 ms more that is not attributed yet.
  Reading the demand through a lazy reshape forced a GPU sync per layer and cost another 10%
  (22.6–25.8 tok/s) until the raw kernel outputs were kept instead.
- Installs on the real model: 17 checks over 2,431 installs, no slot differing from its bytes.
- 5–11% of wanted admissions found no free slot at 4 spares.

Text, same prompt as before (a script that finds duplicate files by SHA-256):

- Thinking off, temperature 0: a complete script in 1,035–1,182 tokens, and it runs correctly
  on a test tree. A 4,341-token prompt holding `omlx_routed.py` got an accurate five-bullet
  summary and a plausible bug risk.
- Thinking on: no answer within 4,096 tokens in any of three runs. Greedy with the cache on is
  coherent for about 1,000 tokens and then repeats itself; default sampling does not repeat but
  keeps reviewing code it drafted in its reasoning; greedy with the policy off (fixed 144)
  also runs out. So the cache is not the cause.
- The exact pool of 192 at FP4 (stock oMLX offload, 7.1 tok/s, greedy) fails the same way: 4,096
  reasoning tokens, no answer, and from about character 10,500 of 16,235 it repeats the same
  list of options it will not add. The lattice cache starts repeating at about character 5,300;
  the fixed lattice set and the sampled run do not repeat. So neither the 2-bit codes nor the
  cache causes the failure with thinking on: the pruned checkpoint has it at full precision.
  Whether the unpruned model has it too is not known; that model is not on the Mac.
- Given room, thinking does finish under default sampling: the lattice cache with a budget of
  12,000 tokens reasoned for 7,567 tokens without repeating itself, then answered in 620 with
  a script that runs correctly on the test tree. 400 s in all, 20.7 tok/s over the 8,187
  tokens (1.5 installs per token, policy 2.4 ms per token). The drop from 26–28 tok/s at
  1,000 tokens is not attributed; context length is the first suspect and the fixed set was
  never timed this far out.
- With the cache on, greedy output is not reproducible: install timing changes routing. Two
  runs diverged after 18 characters; a third matched the second.

Not done: more than 144 residents (160 would be about 67 GiB steady), memory pressure, and
multi-token verify windows in serving (the self-check covers 8 rows).

### Pool of 384 (2026-11-10)

Every expert of the full checkpoint is now reachable on the Mac at 2 bits.

- Codes. The 192 experts per layer that REAP-50 removed were copied from the Runpod volume
  through a CPU pod (`extract_tensors.py`, 144 GB, 76 minutes, about $0.16) and fitted on the Mac
  with the unchanged Metal search: 2.5 h for 40 layers, weighted error 0.091 on w1 and w3 and
  0.079 on w2 (the first 192: 0.090 and 0.074). The torch port refits 64 rows of nine
  projections to the same codes. All 384 router rows per layer came with them; the rows of the
  192 survivors equal the REAP-50 checkpoint's.
- Cache. `OMLX_LATTICE` may name a pool file: code directories with the pool number of each
  expert, a router file and routed-mass statistics for the starting residents. oMLX still loads
  the REAP-50 checkpoint (everything outside the routed experts is shared), and the cache
  replaces router and experts. Self-check over 384: same gaps as over 192; the decode-step MoE
  takes 0.444 ms (0.40–0.42 over 192).
- Starting residents are the 144 experts with the most routed mass in calibration: 83% of the
  mass on average, 66% in the worst layer. On average 118 of them are REAP-50 survivors.

Serving, 144 residents, tau 0.2, 4 spares, 63 GiB (the pool of 192: 62 GiB):

| Request | Tokens | tok/s | Demanded slots resident | Installs per token | Read |
|---|---:|---:|---:|---:|---:|
| Script, thinking off, greedy (first request) | 1,124 | 26.6 | 88% | 8.9 | 80 MB per token |
| Same again | 748 | 23.7 | 88% | 9.1 | 82 MB per token |
| 4,341-token prompt, summary | 282 | 22.6 | | | prefill 23.6 s (184 tok/s) |
| Script, thinking on, greedy, 4,096 budget | 4,096 | 19.1 | 91% | 6.3 | 1.05 GB/s |
| Script, thinking on, sampled, 12,000 budget | 3,256 | 17.3 | | 8.7 | 1.34 GB/s |

- Against the pool of 192 at the same 144 residents: four times the installs (2.2 per token
  there), 88–91% of demanded slots resident against 96%, and 0–15% lower speed with thinking
  off (26.0–27.9 there).
- Text. The first thinking-off script runs correctly on the test tree. The second, from a
  different cache state, has one wrong token (`sha.hexdigdigest()`) and crashes. One sample;
  the pool of 192 produced no such error in its runs.
- Thinking on, greedy: no answer in 4,096 tokens, repeating from about character 12,800 of
  16,235, as the pruned models do. Thinking on, sampled: 2,704 reasoning tokens, then a
  552-token answer whose script runs correctly, 192 s in all (the pool of 192 needed 7,567
  reasoning tokens; one sample each).
- 7% of wanted admissions found no free slot in the first request, 5% over the thinking runs.

Not measured: memory pressure.

### Tau sweep over the pool of 384 (2026-11-10)

Quality against the full FP4 model needs only its stored logits, not the model: the pod wrote
them for the Mac's 22 eval sessions. `local_eval.py --router` puts the 384-row router on the
REAP-50 checkpoint and takes every routed expert from the 2-bit codes the Mac serves, so the
simulated cache (144 residents, re-route on a miss, one-token lag, starting residents by routed
mass) runs on the Mac. First 1,024 tokens of each session, KL against the full model:

| Variant | KL | Top-1 | Installs per token | Read at 30 tok/s |
|---|---:|---:|---:|---:|
| All 384 resident (2-bit coding alone) | 0.390 | 84.4% | 0 | 0 |
| Cache of 144, best set per record (oracle) | 0.424 | 84.0% | | |
| Cache of 144, tau 0.1 | 0.483 | 82.5% | 17.1 | 4.6 GB/s |
| Cache of 144, tau 0.2 | 0.460 | 82.7% | 7.6 | 2.1 GB/s |
| Cache of 144, tau 0.3 | 0.454 | 82.5% | 2.6 | 0.7 GB/s |
| Cache of 144, tau 0.45 | 0.573 | 80.6% | 1.1 | 0.3 GB/s |
| Static 144 (hottest by routed mass) | 0.763 | 76.2% | 0 | 0 |

For reference, same sessions and tokens: the exact REAP-50 checkpoint (192 FP4 experts) 0.578,
REAP-50 at 2 bits 0.747, the cache of 144 inside the pool of 192 0.756.

- The cache of 144 over the full pool beats the static 144 by 0.309 (standard error 0.052, 21
  of 22 sessions) and the exact REAP-50 checkpoint by 0.12, at 53 GB of resident experts.
- Tau 0.2 and 0.3 are equal (difference 0.005, standard error 0.017) and 0.3 reads a third as
  much. Tau 0.1 is worse than 0.2 (0.024, standard error 0.013): admitting weak slots evicts
  useful experts. Tau 0.45 loses 0.12 in every session.
- At tau 0.3 the cache is 0.030 from the oracle and 0.064 from having every expert resident.
- Tool-use text stays at 1.8–2.0 in every variant, all 384 resident included (1.94). That is
  the off-format teacher found on 2026-11-09 (the eval text carries foreign tool markup), not a
  loss from the cache or the coding. Without that domain: 0.23 with all 384 resident, 0.32 for
  the cache of 144 at tau 0.3, 0.66 for the static 144.
- Run: 11,991 s, peak RSS 63.8 GB (`~/Models/v41-reap-run/jobs/tau384_mac.sh`).

Serving at tau 0.3 (144 residents, 4 spares, 63 GiB), against the tau 0.2 table above:

| Request | Tokens | tok/s | Demanded slots resident | Installs per token | Read |
|---|---:|---:|---:|---:|---:|
| Script, thinking off, greedy (first request) | 1,030 | 27.2 | 80% | 2.5 | 22 MB per token |
| Same again | 1,085 | 26.7 | 83% | 1.4 | 13 MB per token |
| 4,341-token prompt, summary | 310 | 23.0 | 71% | | prefill 24.8 s (175 tok/s) |

Both scripts run correctly on the test tree. Installs match the simulation (2.6 per token).
Tau 0.3 is the serving default from here.

Limits: the cache is simulated with a fetch that lands within one token; 22 sessions, first
1,024 tokens; two generated scripts at tau 0.3; no pod confirmation on longer sessions (E7).

### Drafter on the no-wait cache (2026-11-11)

The checkpoint's drafter (DSpark: 3 stages, 128 experts each, 7.4 GiB) is switched on by
`mtp_enabled` in `<base-path>/model_settings.json`; no code change was needed. A cycle drafts 3
tokens and verifies them in one backbone forward of about 4 rows through the same hook the
single-token path uses. 120 residents (so the footprint stays under the 80 GB cap), tau 0.3,
pool of 384, installed slots byte-checked (0 mismatches):

| Request | Drafter off | Drafter on | Gain |
|---|---:|---:|---:|
| Script, thinking off, greedy | 24.3 tok/s | 31.2 | +28% |
| Same, temperature 1.0, top_p 1.0 | 24.2 | 30.5 | +26% |
| 4,341-token prompt, summary | 21.8 | 23.9 | +9% |
| Script, thinking on, temperature 1.0 | 17.45 (3,872 tokens) | 25.0 (8,492 tokens) | +43% |
| Footprint, steady / peak | 55 / 61 GiB | 62 / 68 GiB | +7 GiB |

- About 2.9 tokens are kept per cycle; a cycle takes 93 ms against 41 ms for a single token.
- Installs per forward row are unchanged; per emitted token they are 1.38 times the no-drafter
  figure (0.67 GB/s read against 0.41 with thinking on).
- oMLX's own offload gains 29% from the drafter (6.6 to 8.5 tok/s) because verification waits
  on experts; here nothing waits.
- A multi-row lattice kernel (one decode per distinct expert in the block) is not worth
  building yet: over the full model's routes, distinct experts per route are 0.84 at 2 rows,
  0.70 at 4 and 0.63 at 6, so it saves at most about 12% of a cycle.
- Job: `~/Models/v41-reap-run/jobs/k1_mtp.sh`.

### Sampling at the card's settings (2026-11-11)

The same script request, 8 samples per row, 4,096-token budget, thinking off. "Correct" means
the script runs and lists the three duplicate files of the test tree.

| Model | Residents | Temperature | Drafter | Correct of 8 |
|---|---:|---:|---|---:|
| Cache over the pool of 384, 2-bit | 120 | 1.0 | off | 4 |
| Same | 120 | 1.0 | on | 4 |
| Same | 144 | 1.0 | off | 6 |
| Same | 120 | 0.6 | off | 7 |
| Exact REAP-50 checkpoint (192 FP4 experts, stock oMLX offload, 4–5 tok/s) | 192 | 1.0 | off | 5 |

- Failures are single-token slips (`hexdigist`, `mun` for `min`, a constant misspelled at its
  use, a missing colon) and two samples with wrong output. Every greedy script was correct.
- With 8 samples no row is separable from another (4 of 8 against 7 of 8 is p = 0.28, Fisher).
  The exact pruned checkpoint at its original precision fails 3 of 8, so the slips are not
  evidence against the 2-bit coding, the cache or the drafter.
- Not known: how often the full model slips at temperature 1.0 (it does not fit on the Mac),
  and whether more samples separate the rows. 16 more per row for the cache of 144 and the
  exact checkpoint are queued behind the simulator run.
- Jobs: `~/Models/v41-reap-run/jobs/k1_mtp_samples.sh`; scorer in the session scratchpad.

### Quality against resident count (C1, 2026-11-11)

Same simulator, sessions and tokens as the tau sweep: pool of 384, tau 0.3, re-route on a miss,
22 sessions, first 1,024 tokens, KL against the full FP4 model. Footprint is the measured
serving model (11.8 GiB outside routed experts plus 0.336 GiB per resident slot).

| Residents per layer | Resident experts | Footprint | KL | KL without tool-use | Top-1 | Read per token |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 11.8 GB | 23 GiB | 0.942 | 0.732 | 73.5% | 118 MB |
| 48 | 17.7 GB | 28 GiB | 0.790 | 0.582 | 77.1% | 82 MB |
| 64 | 23.6 GB | 33 GiB | 0.705 | 0.496 | 78.7% | 61 MB |
| 96 | 35.4 GB | 44 GiB | 0.605 | 0.390 | 81.1% | 39 MB |
| 144 | 53.1 GB | 60 GiB | 0.454 | 0.320 | 82.5% | 24 MB |
| 200 | 73.7 GB | 79 GiB | 0.401 | 0.279 | 83.8% | 13 MB |
| 250 | 92.2 GB | 96 GiB | 0.398 | 0.254 | 84.4% | 5.5 MB |
| 384 (all) | 141.6 GB | | 0.390 | 0.235 | 84.4% | 0 |
| Static 144, for reference | 53.1 GB | 60 GiB | 0.763 | 0.654 | 76.2% | 0 |

- 144 reproduces the tau sweep (0.4542, every session equal).
- A cache of 48 at 28 GiB is level with the static 144 at 60 GiB (0.790 against 0.763; 0.582
  against 0.654 without tool-use), and a cache of 96 at 44 GiB is level with the exact REAP-50
  checkpoint (0.605 against 0.578), which needs 341 GB of FP4 experts on disk.
- 144 to 200 gains 0.053 (standard error 0.009, 22 of 22 sessions). 200 to 250 gains 0.003
  over all sessions and 0.024 without tool-use: breadth is spent near 200 residents.
- Reads grow as the cache shrinks: 2.5 GB/s at 30 tok/s for 48 residents, 3.5 GB/s for 32.
- Run: 12,400 s (`~/Models/v41-reap-run/jobs/c1b1_mac.sh`).

### FP4 on the hottest experts (B1, 2026-11-11)

`local_eval.py --hot NAME=H`: the H hottest experts per layer by routed mass, among those the
REAP-50 checkpoint holds, run from its FP4 weights whenever they are resident (18.8 MB against
9.2 MB at 2 bits). This is the ceiling for any extra bits on those experts.

| Variant | Resident experts | KL | Without tool-use | Top-1 |
|---|---:|---:|---:|---:|
| 144 at 2 bits | 53.1 GB | 0.454 | 0.320 | 82.5% |
| 127 residents, 16 of them FP4 | 53.0 GB | 0.413 | 0.310 | 83.3% |
| 110 residents, 32 of them FP4 | 52.8 GB | 0.476 | 0.331 | 82.9% |
| 177 at 2 bits | 65.2 GB | 0.421 | 0.294 | 83.2% |
| 144 residents, 32 of them FP4 | 65.4 GB | 0.392 | 0.280 | 83.9% |
| 200 at 2 bits | 73.7 GB | 0.401 | 0.279 | 83.8% |
| All 384 at 2 bits | 141.6 GB | 0.390 | 0.235 | 84.4% |
| All 384, 32 of them FP4 | 153.8 GB | 0.268 | 0.167 | 86.9% |

Paired by session (difference, standard error, sessions better):

| Comparison | All 22 sessions | 20 without tool-use |
|---|---|---|
| 16 FP4 against 144 at 2 bits, 53 GB | −0.041 ± 0.023 (12) | −0.010 ± 0.006 (10) |
| 32 FP4 against 144 at 2 bits, 53 GB | +0.022 ± 0.010 (5) | +0.011 ± 0.006 (5) |
| 144 with 32 FP4 against 177 at 2 bits, 65 GB | −0.029 ± 0.013 (14) | −0.015 ± 0.009 (12) |
| 144 with 32 FP4 against 144 at 2 bits (12 GB more) | −0.062 ± 0.016 (19) | −0.041 ± 0.008 (17) |
| All 384 with 32 FP4 against all 384 at 2 bits | −0.122 ± 0.039 (22) | −0.067 ± 0.013 (20) |

- Gate (0.02 over the equal-memory 2-bit rival): met with tool-use counted (0.041 and 0.029),
  not met without it (0.010 and 0.015). The tool-use sessions carry the off-format teacher, so
  the reading without them is the one to trust. At 53 and 65 GB, FP4 on the hot set is a
  small gain at best, and trading 34 residents for 32 FP4 experts is a loss.
- Precision alone is large: with every expert resident, FP4 on 32 experts per layer (8% of
  them) removes 0.067 of the 0.235 left, in all 20 sessions. Those experts take 22% of routed
  slots.
- 144 residents with 32 at FP4 (65 GB) equals 200 residents at 2 bits (74 GB).
- So extra bits pay where breadth is spent, above about 200 residents, which is the budget of
  a 128 GB Mac. That budget was not tested here; 217 residents with 32 FP4 and 183 with 64
  FP4 against 250 at 2 bits (all 92 GB) is the deciding run.
- Run: 8,549 s, peak RSS 61 GB.

### Serving at 48 and 96 residents (C2, 2026-11-11)

Same sampled script request as above, card sampling (temperature 1.0, top_p 1.0), 8 samples,
drafter off, tau 0.3. Fewer slots stand in for a smaller Mac's expert memory, not for its
memory pressure.

| Residents | Footprint | tok/s | Demanded slots resident | Installs per token | Read per token | Correct of 8 |
|---:|---:|---:|---:|---:|---:|---:|
| 48 | 30 GiB | 20–27 | 65% | 5.8 | 53 MB | 0 |
| 96 | 46 GiB | 22–25 | 75% | 3.2 | 28 MB | 2 |
| 120 (above) | 55 GiB | 24 | | | | 4 |
| 144 (above) | 63 GiB | 17–27 | | | | 6 |

- Speed holds at small sizes; the text does not. At 48 residents the output is misspelled
  throughout (`hashllib`, `OSEError`, `o a file`) and 6 of 8 samples run to the 4,096-token
  limit. At 96, 3 of 8 run to the limit.
- Correct scripts rise with residents: 0, 2, 4, 6 of 8 at 48, 96, 120, 144. A KL near 0.8 is
  unusable under the card's sampling and 0.45 is marginal, so the KL curve above overstates
  what a small Mac gets. The exact REAP-50 checkpoint (KL 0.578) gives 5 of 8.
- Open: whether a lower temperature or a trimmed tail (top_p 0.95) makes 96 residents usable,
  and how many of 8 the full model gets. The first is queued (`jobs/after_b1b.sh`).


### FP4 on the hottest experts at 92 GB (B1b, 2026-11-11)

The deciding run named above: the expert budget of a 128 GB Mac (92 GB), pool of 384, tau 0.3,
re-route, 22 sessions, 1,024 tokens. KL against the full model; "no tool-use" drops the two
off-format sessions.

| Variant | Residents | Expert memory | KL | KL, no tool-use | Top-1 |
|---|---|---:|---:|---:|---:|
| 250 at 2 bits (C1) | 250 | 92 GB | 0.398 | 0.254 | 84.4% |
| 217, hottest 32 at FP4 | 217 | 92.3 GB | 0.325 | 0.219 | 85.6% |
| 183, hottest 64 at FP4 | 183 | 92.0 GB | 0.328 | 0.221 | 86.1% |
| All 384 at 2 bits (C1) | 384 | 139 GB | 0.390 | 0.235 | |
| All 384, hottest 32 at FP4 (B1) | 384 | 154 GB | 0.268 | 0.167 | 86.9% |

Paired differences, mean ± standard error (sessions better):

| Pair | All 22 | Without tool-use (20) |
|---|---|---|
| 217 with 32 FP4 against 250 at 2 bits | −0.072 ± 0.027 (17) | −0.036 ± 0.010 (15) |
| 183 with 64 FP4 against 250 at 2 bits | −0.070 ± 0.030 (16) | −0.034 ± 0.011 (14) |
| 217 with 32 FP4 against 200 at 2 bits (74 GB) | −0.075 ± 0.016 (21) | −0.060 ± 0.009 (19) |
| 217 with 32 FP4 against all 384 at 2 bits (139 GB) | −0.064 ± 0.035 (15) | −0.016 ± 0.009 (13) |
| 183 with 64 FP4 against 217 with 32 FP4 | +0.002 ± 0.015 | +0.002 ± 0.003 |

- Gate (0.02 over the equal-memory 2-bit rival): met on both readings, 0.072 with tool-use and
  0.036 without. B continues.
- At 92 GB, 217 residents with 32 at FP4 are at least level with every expert resident at 2
  bits, which takes 139 GB. The gain is in the agentic, networking and cyber sessions (0.32 to
  0.25, 0.25 to 0.16, 0.48 to 0.38); prose and business sessions do not move.
- 32 and 64 FP4 experts per layer score the same, so the hot set is small: 32 is enough, and
  the choice between them can be made on serving cost.
- With B1: bits lose to breadth at 53 GB, draw at 65 GB, win at 92 GB. The crossover is near
  200 residents.
- Serving this needs about 96 GiB of footprint (185 slots at 2 bits, 32 at FP4, 11.8 GiB
  outside the routed experts), above the 80 GB per-process rule on this Mac. Under the rule
  the hot tier can be served at about 130 residents with 32 at FP4.
- Run: 3,339 s, peak RSS 60 GB.

### More samples and two rescue checks (2026-11-11)

Same sampled script request, 8 samples per run, 4,096-token budget, drafter off.

| Model | Residents | Temperature | top_p | Correct | tok/s |
|---|---:|---:|---:|---|---:|
| Cache, pool of 384 at 2 bits | 144 | 1.0 | 1.0 | 17 of 24 (6, 5, 6) | 22–27 |
| Exact REAP-50, FP4, stock offload | 192 | 1.0 | 1.0 | 18 of 24 (5, 6, 7) | 5.6–6.3 |
| Cache | 144 | 1.0 | 0.95 | 5 of 8 | 24–27 |
| Cache | 120 | 0.6 | 1.0 | 7 of 8 (earlier) | |
| Cache | 96 | 1.0 | 1.0 | 2 of 8 (C2) | 22–25 |
| Cache | 96 | 1.0 | 0.95 | 2 of 8 | 24–26 |
| Cache | 96 | 0.6 | 1.0 | 1 of 8 | 20–28 |
| Cache | 48 | 0.6 | 1.0 | 0 of 8 | 23–25 |

- With 24 samples each, the cache of 144 at 2 bits and the exact REAP-50 checkpoint are level
  (71% and 75%) at the card's sampling, and the cache is about four times as fast. Their
  failures are of the same kind: one misspelled name or a wrong output.
- Neither a lower temperature nor top_p 0.95 rescues 96 or 48 residents. The failures at 96
  with temperature 0.6 are still misspelled identifiers (`hexdigdigest`, `diggest`,
  `filnames`), so the damage comes from the re-routed slots (25% of demand at 96 residents,
  35% at 48), not from the sampling tail. top_p 0.95 does nothing at 144 either.
- Between 96 and 120 residents at temperature 0.6 the count goes from 1 to 7 of 8. The usable
  floor on this request is about 120 residents (55 GiB), which is a 96 GB Mac, not a 64 GB one.
- Still unknown: how many of 24 the full model gets (it does not fit on this Mac).

### Serving 200 residents (2026-11-11)

Jack raised the per-process limit to 100 GB for serving. Same sampled script request, card
sampling, 24 samples, drafter off, tau 0.3, 200 residents at 2 bits over the pool of 384.

| Residents | Footprint, peak | Demanded slots resident | Installs per token | Read per token | Correct of 24 |
|---:|---:|---:|---:|---:|---:|
| 144 (above) | 63 GiB | | | | 17 |
| 200 | 87 GiB | 90% | 0.40 | 3.6 MB | 19 |
| Exact REAP-50 (above) | | | | | 18 |

- 19 of 24, and none of the 5 failures is a misspelled name or a syntax error: 4 scripts run
  and print the wrong list, 1 answer has no code block. At 144 residents 5 of 7 failures were
  misspelled names or syntax errors. Fewer re-routed slots (10% of demand against 25% at 96)
  remove the spelling damage.
- Footprint starts at 81 GiB and peaks at 87 GiB over three server runs, 6 to 8 GiB above
  the 11.8 + 0.336 per resident estimate, so the estimate needs that margin at this size.
- Speed was 11–18 tok/s, but the residual fit shared the GPU during these runs; a clean
  speed run is queued (`jobs/after_b2.sh`).

### Third bit on the hottest experts (B2, 2026-11-12)

A 1-bit E8 residual was fitted on the 2-bit base for the 192 survivors
(`keep_quant.py --residual-of quant/e8p-podimp --code-bits 8`, 310 s per layer, 33 GB).
Weighted weight error falls from 0.084 to 0.030 (gate and up) and from 0.047 to 0.024 (down).
A hot expert then runs base + residual, 13.8 MB instead of 18.8 MB for FP4
(`local_eval.py --hot-tier DIR`). Same 92 GB budget and settings as B1b.

| Variant | Residents | KL | KL, no tool-use | Top-1 |
|---|---:|---:|---:|---:|
| 250 at 2 bits (C1) | 250 | 0.398 | 0.254 | 84.4% |
| Hottest 32 at FP4 (B1b) | 217 | 0.325 | 0.219 | 85.6% |
| Hottest 32 at 3 bits | 234 | 0.327 | 0.223 | 85.3% |
| Hottest 64 at 3 bits | 218 | 0.311 | 0.215 | 85.9% |
| Hottest 96 at 3 bits | 202 | 0.308 | 0.222 | 85.8% |

Paired differences, mean ± standard error (sessions better):

| Pair | All 22 | Without tool-use (20) |
|---|---|---|
| 64 at 3 bits against 32 at FP4 | −0.015 ± 0.008 (16) | −0.004 ± 0.003 (14) |
| 32 at 3 bits against 32 at FP4 | +0.002 ± 0.004 (10) | +0.004 ± 0.004 (9) |
| 96 at 3 bits against 64 at 3 bits | −0.002 ± 0.011 (8) | +0.007 ± 0.004 (7) |
| 64 at 3 bits against 250 at 2 bits | −0.087 ± 0.034 (19) | −0.039 ± 0.009 (17) |
| 64 at 3 bits against all 384 at 2 bits (139 GB) | −0.079 ± 0.042 (14) | −0.020 ± 0.008 (12) |

- The residual does the work of the FP4 copy at equal memory: level with it on the reading
  without tool-use, slightly ahead with it. One more bit on a hot expert recovers what FP4 on
  it recovers, because the memory saved goes to more hot experts or more residents.
- 64 hot experts per layer is the best of the three; 96 gives nothing more.
- Unlike an FP4 copy, the residual adds to the base the cache already holds. A hot expert
  needs no second format in the slot array, and losing the residual leaves the expert running
  at 2 bits.
- Run: 5,854 s, peak RSS 63 GB. Smoke (one session, 96 tokens, 144 residents): 1.521 at 2
  bits, 1.309 with 32 hot experts at 3 bits.

### Clean speed at 200 residents, and 180 residents with the drafter (2026-11-12)

Same sampled script request, card sampling, tau 0.3, nothing else on the GPU.

| Configuration | Footprint, peak | tok/s | Correct |
|---|---:|---:|---|
| 200 residents, drafter off | 87 GiB | 21–28 (3 samples) | 19 of 24 (above), 3 of 3 here |
| 180 residents, drafter on | 88 GiB | 22–41, median 32 (24 samples) | 21 of 24 (7, 8, 6) |
| 144 residents, drafter off (above) | 63 GiB | 22–27 | 17 of 24 |
| Exact REAP-50, stock offload (above) | | 5.6–6.3 | 18 of 24 |

- More residents cost no speed: 200 residents decode as fast as 144.
- 180 residents with the drafter is the candidate default for a 128 GB Mac: every expert of
  the full model reachable, 88 GiB, a median of 32 tok/s, and 21 of 24 sampled scripts
  correct, against 18 of 24 at about 6 tok/s for the exact REAP-50 checkpoint on stock oMLX.
  The three failures are a wrong listing, a wrong output format and one wrong keyword
  argument; none is a misspelling.
- 21 against 18 of 24 is not a separable difference (Fisher p about 0.5); the claim is "no
  worse, five times faster", not "better".

### Slots from RAM (C3, 2026-11-12)

`OMLX_LATTICE_SLOTS` now defaults to `auto` (`auto_slots` in `scripts/v41_build/omlx_lattice.py`):
a footprint budget of 70% of RAM (`OMLX_LATTICE_BUDGET_GIB` overrides it), less 20 GiB outside
the routed experts and 8 GiB when `model_settings.json` switches the drafter on, at 0.336 GiB
per resident. The constants are this M4 Max's measured peaks; other chips are unmeasured.

| RAM | Residents, drafter off | Residents, drafter on |
|---:|---:|---:|
| 64 GB | 73 | 49 |
| 96 GB | 140 | 116 |
| 128 GB | 207 | 183 |

- Below 120 residents the server prints that generated code was unreliable in our tests
  (C2), so a 64 GB Mac starts with a warning and a 96 GB Mac should leave the drafter off.
- Checked on this Mac with the drafter on: 183 residents chosen, 83 GiB peak over two
  requests, 39–41 tok/s, both scripts correct.

### One directory, one command (C4, 2026-11-12)

- `scripts/v41_build/slim_checkpoint.py SRC DST` writes a compact model directory as real
  files without the routed experts its loader never reads: 365.8 GB to 225.9 GB for the
  lattice model (the two 101.5 GB Engram shards and three others lose nothing and are cloned,
  43 shards are rewritten and read back against the source by CRC). The six stub experts per
  layer, the drafter and its experts stay.
- Bundle layout, 340 GiB (`~/Models/v41-reap-run/bundle`): `models/v41-lattice` (slim
  checkpoint), `codes/kept` and `codes/removed` (the 2-bit codes of all 384 experts, unchanged),
  `ids-kept.json`, `ids-removed.json`, `router.safetensors`, `mass.npz`, and `pool.json`, whose
  paths are now relative to it.
- `scripts/v41_build/run_v41.sh BUNDLE [PORT]` checks the machine, clones oMLX at the pinned
  commit and builds its venv with uv on the first run, writes `base/model_settings.json`
  (drafter on from 128 GB of RAM) and starts the server; the resident count comes from RAM.
- Checked on this Mac: with the existing oMLX install, and again from an empty oMLX directory
  (fresh clone at `4213c328`, `uv sync`): 183 residents, 83 GiB, 38–40 tok/s, 2 of 3 sampled
  scripts correct in each run.
- `pool384/fp4-removed` (134 GB) was deleted with Jack's confirmation; the hot tier uses
  survivors only. Not done: publishing (needs Jack's OK; the Mac's uplink is 3 MB/s, so an
  upload would run from a pod) and a model card for the bundle.

### Residual kernel for the hot tier (B3, first part, 2026-11-12)

- `src/ramp/kernels/e8_1bit_m1.metal` and `e8_1bit_m1.py`: the token and per-route layouts of
  `e8p_m1` over 8-bit E8 codes, for a residual added to the 2-bit output. A route whose index
  is `NO_RESIDUAL` writes zeros, so one call serves a token whose routes are partly hot.
- `tests/test_e8_1bit_m1.py` (19 cases): both layouts against a float64 decode of the same
  codes, fp32 within 1e-5 and bf16 within 2e-2 of the largest output, with routes that have
  no residual, a ragged last tile and the model's group size; also passes under
  `MTL_SHADER_VALIDATION=1`.
- Timed alone at the model's sizes (one token, six routes, 200 interleaved runs, M4 Max): gate
  0.217 ms against 0.227 ms for the 2-bit kernel, down 0.248 against 0.252; with two of six
  routes hot 0.185 and 0.194. Single calls are launch-bound, so the cost inside the decode step
  is still to be measured.
- Not done: the hot residual arrays, the per-slot hot index and the memory accounting in
  `omlx_lattice.py`.

### Memory pressure (D3, 2026-11-12)

`scripts/v41_build/d3_pressure.py` sends the sampled script request (500 tokens, card sampling)
back to back while a second process fills incompressible memory in 4 GB steps every 45 s up to
40 GB, holds it for 2 minutes, releases it and measures 2 minutes more. Server: the bundle
through `run_v41.sh`, 183 residents, drafter on.

| Phase | tok/s, median (range), tau 0.3 | tok/s, median (range), tau 0.2 |
|---|---|---|
| Before | 36.7 (30.2–42.6) | 39.8 (37.1–42.4) |
| Taking 1–20 GB | 29.5 (25.9–32.2) | 31.4 (27.4–35.8) |
| Taking 21–40 GB | 32.8 (29.0–34.9) | 32.7 (20.7–35.4) |
| Holding 40 GB | 33.5 (30.5–34.5) | 32.2 (20.5–34.8) |
| After release | 35.9 (34.0–37.2) | 31.8 (23.5–34.6) |

- The server keeps its 83 GiB in both runs and answers every request (61 and 62). macOS
  compresses up to 24 GB and swaps 20 GB out (5 to 25 GB of swap); the pressure level reaches
  "warning", never "critical". With 40 GB held, decoding is 9% slower at tau 0.3; the slowest
  phase is the first allocation steps (20%), when the other process is filling memory.
- At tau 0.3 the cache installs 0.17 experts per token (1.5 MB read per token) and 90% of
  demanded slots are resident; at tau 0.2, 3.7 installs and 33.6 MB per token. The heavier
  reads make single requests drop to 20 tok/s under pressure and slow the recovery.
- The tau 0.2 run was a launcher bug: `run_v41.sh` did not set the admission threshold and
  the code's default is 0.2, while every measurement here used 0.3. The launcher now sets 0.3.
  The bundle checks under C4 above (38–40 tok/s) ran at 0.2.
- What this does not show: the server's own pages being taken. Its footprint counts
  compressed and swapped pages too, so whether any of its 83 GiB left RAM is not measured;
  the speed says little of it did.
