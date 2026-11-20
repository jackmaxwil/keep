# Eleven ideas beyond oMLX, and a plan for 20 to 22 November

Written 2026-11-20 (Friday). Each idea is tied to a number this campaign measured or to what
oMLX's source does at the pinned commit (`4213c328`). "Ceiling" is the most an idea could buy,
not a forecast. Status is updated here as kill tests land.

## Where the error is

At 144 residents per layer (53 GB of experts, the 128 GB configuration) the KL to the full model
is 0.320 nats on the paper's metric. It has two parts.

| Part | Size | Source |
|---|---:|---|
| Coding error | 0.235 | all 384 experts resident at 2 bits |
| Routing error | 0.085 | 0.320 minus 0.235; grows to 0.155 at 96 residents and 0.347 at 48 |
| of which the starting set | 0.065 | each text's own best 144 score 0.255 |

Two more facts shape the list. Below about 120 residents generated scripts break while KL looks
fine (the residency floor), and a static set of 144 writes a working script in 4 of 24 samples
against 17 of 24 for the routed set. oMLX's offload has none of this error, because it is exact;
its cost is speed, since with its offload on it leaves its fused decode path and runs seven
times slower than a static set.

## Ranking by ceiling

| Rank | Idea | Moves | Ceiling | Status |
|---:|---|---|---|---|
| 1 | Better 2-bit codes | KL at every budget | up to 0.235 | open |
| 2 | Replay on a strong miss | generation floor, KL, exactness | 0.085 at 144, 0.347 at 48 | narrowed by its kill test, see below |
| 3 | Exact prefill and residents from the prompt | KL, starting set | 0.065 plus prompt contamination | warm-start half queued |
| 4 | A 1.5-bit code | reach to 64 GB machines | a third more residents per GB | open |
| 5 | Low-rank stand-ins for absent experts | routing error | part of 0.085 to 0.347 | open |
| 6 | Uncertainty-gated replay | cost of idea 2 | same quality, fewer replays | after idea 2 |
| 7 | Domain profiles | starting set | a share of 0.065 | open |
| 8 | Engram in the memory account | speed or residents | unknown | needs a profile |
| 9 | Adaptive active count | decode speed | about 8% | open |
| 10 | Multi-row kernel | drafter speed | at most 12% of a cycle | measured, low |
| 11 | Fitted per-expert thresholds | KL | at most 0.064 | lost twice |

On a 128 GB machine coding error is the larger term, so codes rank first. On 64 and 96 GB
machines routing error and the floor dominate, and ideas 2 and 4 move ahead of it.

## The ideas

### 1. Better 2-bit codes

- **What.** The fit (`keep_quant.py`, `fit_projection`) rounds each group of eight weights to the
  E8P lattice with a per-input-channel importance diagonal and least-squares group scales. It
  has no error feedback between columns and no incoherence processing, both of which QuIP# uses
  with the same codebook.
- **Why.** Coding error is 0.235 of the 0.320 nats at 144 residents and nearly all of the error
  from 250 residents up. Every configuration inherits it.
- **Kill test.** Capture the input second-moment matrix of a few hot experts of one layer on
  calibration text, refit them with error feedback, and compare the weighted output error with
  the current fit (0.079 to 0.091). No gain on output error, no project.
- **Cost.** A capture pass through the model (a heavy job) and a fit change. A full refit is
  about 2.5 h per 192 experts per pass on the Mac.
- **Also.** This is the "trained codes" route for rubric item 7.

### 2. Replay on a strong miss

- **What.** Serve the token with re-routing as now. The demand already rides along in the graph.
  If an expert with gate weight at or above a wait threshold W was absent, install it and rerun
  the token from the first layer that missed. W infinite is today's policy; W zero is exact
  output, without leaving the fused decode path as oMLX's offload does.
- **Why.** Son reports 0.35% perplexity lost when the strongest misses are fetched against 5.7%
  when every miss is substituted. The floor's failures track the re-routed share of demand.
- **Kill test, done 2026-11-20** (`scripts/runpod/v41_reap/miss_rate.py`, the stored demand of
  52 calibration records, 512 tokens each, policy replayed at tau 0.3):

  | Residents | Gate weight resident | First choice missed | W = 0.3 | W = 0.5 | W = 0.7 |
  |---:|---:|---|---|---|---|
  | 96 | 78.1% | 83.0% of tokens, 61.7% of layers rerun | 90.5%, 68.4% | 38.9%, 20.9% | 12.9%, 5.1% |
  | 144 | 84.1% | 73.5%, 50.3% | 83.2%, 57.5% | 31.0%, 15.4% | 10.1%, 3.9% |
  | 200 | 90.1% | 57.5%, 35.1% | 69.1%, 42.4% | 20.9%, 9.3% | 6.5%, 2.4% |

  Each cell is the share of tokens with at least one such miss in any of the 40 layers, and the
  share of all layer computations a replay from the first missed layer would repeat.
- **Reading.** Replaying every missed first choice is dead: with 40 layers, three tokens in four
  have one somewhere, and half the model would run twice. A rule limited to very strong misses
  survives on cost: W = 0.7 reruns 4% of the layer work at 144 residents, W = 0.5 reruns 15%.
  Whether fetching only those misses buys quality is not known.
- **Next test.** A simulator variant in which a demand with gate weight at or above W is served
  exactly (W = 0.5 and 0.7, at 96 and 144 residents), scored on KL, then on sampled scripts.
- **Caveats.** The trace is short calibration text with a cold start per record, so rates over
  a long session should be lower; the demand was recorded under a randomized policy.

### 3. Exact prefill, then residents chosen from the prompt

- **What.** Prefill runs layer by layer over a chunk, so waiting there costs one wait per layer
  and chunk, not per token: route the chunk, read the experts it misses, run. The prompt's own
  demand then picks the starting residents.
- **Why.** Today the prompt is processed under re-routing and its errors stay in the attention
  cache for the whole generation. The starting set is worth 0.065.
- **Kill test.** The warm-start half is coded (`--cache-policy NAME=prompt:P`) and queued. The
  exact half is one more policy: serve the first P tokens with every expert, then continue.
- **Cost.** Simulator runs. In serving, read volume on long prompts; bound it by reading only
  the experts with the most gate mass in the chunk.

### 4. A 1.5-bit code

- **What.** A lattice codebook of 4,096 points for each group of eight weights, between the
  1-bit code (below the knee) and the 2-bit code.
- **Why.** If it is usable, a 64 GB machine holds a third more residents and may clear the
  floor. If the knee sits above 1.5 bits it is worth nothing.
- **Kill test.** Fit one layer and compare weighted error with the 1-bit and 2-bit fits, then
  score all-resident KL for that layer alone.
- **Cost.** A codebook, one small fit on the GPU, one simulator variant.

### 5. Low-rank stand-ins for absent experts

- **What.** Keep a rank-16, activation-aware factorization of every absent expert (about 0.7 MB
  against 9 MB) and run it on a miss instead of re-routing: a fifth miss rule.
- **Why.** Re-routing runs a different expert; a stand-in runs an approximation of the right one.
- **Kill test.** Offline, on stored demand: output error of the stand-in against the output
  error of the re-routed choice, per missed slot.
- **Risk.** Expert weights may not be low-rank enough for this to beat re-routing.

### 6. Uncertainty-gated replay

- **What.** Idea 2, with a replay only when a strong miss coincides with a small margin between
  the top two output tokens.
- **Why.** The floor hypothesis is a confidently wrong token after a re-routed choice. Most
  strong misses change nothing; the margin says which could.
- **Kill test.** In the simulator, the share of top-1 disagreements with the full model that
  fall on tokens with a strong miss and a small margin.
- **Depends on.** Idea 2 showing a quality gain.

### 7. Domain profiles

- **What.** Cluster calibration routes into a few resident sets. After the first tokens of a
  session, install the nearest set in one batch.
- **Why.** A cheap stand-in for predicting a session's experts, and it needs no prompt as long
  as idea 3 does.
- **Kill test.** Offline, on stored demand: gate weight served by the nearest profile against
  the static set (65%) and each text's own best set (96%).

### 8. Engram in the memory account

- **What.** The Engram tables are served from storage. Their hot rows could be pinned in memory
  and traded against experts by the same accounting as everything else.
- **Why.** Unknown. No profile says what Engram page faults cost per token.
- **Kill test.** A Metal system trace and a fault count during decode.

### 9. Adaptive active count

- **What.** Stop adding experts for a token once the cumulative gate weight passes a cutoff,
  instead of always running six.
- **Why.** Skipping slots with gate weight below 0.1 is close to free (+0.001 KL, 7.7% of
  slots). Expert decode is about 9 of the 37 ms of a token, so the ceiling is small.
- **Kill test.** Simulator variants at several cutoffs; average experts run against KL.

### 10. Multi-row lattice kernel

- **What.** Decode each distinct expert once per verification block of the speculative decoder.
- **Status.** Measured: distinct experts per route are 0.84 at two rows and 0.63 at six, so it
  saves at most about 12% of a cycle. Build only if a trace shows expert decode dominating.

### 11. Fitted per-expert thresholds

- **Status.** Two fits lost to a single threshold (+0.022 to +0.026 KL). Prices measured under
  random misses do not rank a policy chosen to exploit them. What is left is a closed-loop
  search with a ceiling of 0.064, behind every idea above.

## Not yet measured, and needed

- Where the 37 ms of a token goes. Expert decode is about 9 ms, and decode uses about 58 GB/s
  of roughly 400 GB/s of memory bandwidth, so it is not bandwidth-bound. No speed idea can be
  ranked properly without a trace.
- Why the median speed of one configuration varies by 25% between runs.

## Plan

One heavy job runs on the Mac at a time, and the held-out confirmation (rubric item 4) holds
that slot until late Friday, with the prompt warm start behind it. So Friday's work is whatever
needs no model forward pass, plus code for the runs that follow.

| When | Work | Needs the heavy slot |
|---|---|---|
| Fri | Idea 2 kill test on stored demand (done) | no |
| Fri | Idea 4: 1.5-bit codebook, fit one layer, compare error | no (small GPU fit) |
| Fri | Ideas 5 and 7 kill tests on stored demand | no |
| Fri | Code and unit tests for the simulator variants of ideas 2, 3 and 9 | no |
| Fri | Idea 1: error-feedback fit, tested on synthetic data | no |
| Fri night | Held-out confirmation, then prompt warm start (queued) | yes |
| Sat | One simulator pass with the variants of ideas 2, 3 and 9 (about 3.5 h) | yes |
| Sat | Idea 1: capture second moments for one layer, refit, compare | yes |
| Sat | Metal trace of the server (ideas 8 and 10) | yes |
| Sun | Sampled scripts for whichever of ideas 2 and 3 improved KL | yes |
| Sun | Idea 6 if idea 2 gained; idea 4 simulator variant if its fit passed | yes |
| Sun | Results into this file, the log and the paper's open items | no |

What this plan delivers by Sunday is a kill test for every idea and a simulator result for the
ones that survive. It does not deliver served implementations; an idea that passes gets one
after that.

Rules for these runs: variants are developed on the Mac's 22 sessions only. The 42 held-out
sessions are scored once, for the configuration that is finally chosen. Idea 11 gets no time
unless everything above it fails.
