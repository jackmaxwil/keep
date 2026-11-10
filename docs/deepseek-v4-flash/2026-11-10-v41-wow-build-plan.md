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
| K1 | Stock oMLX on this Mac, local pruned checkpoint: tok/s, hit rate, resident memory, and a replayed prompt (hit rate near 1) for its no-miss floor. | The base, if the no-miss floor is under about 10 tok/s: switch to Rapid-MLX's decode kernels. |
| K2 | `bench_e8p_m1` at 5120 × 2304 against oMLX's 2-bit affine kernel. | Our lattice kernel in the runtime: ship affine 2-bit instead. |
| K3 | Static 250 and 288 of 384 against the full model, teacher-forced, on the pod (running). | Nothing; it sets how much the cache has to recover and the target C. |
| K4 | E1: expert cache with re-route against static sets, teacher-forced (queued tonight). | The quality case for the cache, if it is not better than a static set. |
| K5 | 20 lines: an MLX array over a 1 GB memory-mapped file through the E8P kernel; check footprint and `mincore`. | Idea 3 (page cache), if it copies. |
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
| 37 | Gate: cache beats static | K4, tonight |
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
