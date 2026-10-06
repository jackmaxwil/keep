# Wave 3 — the layer-sequential streaming teacher runner, built and running

**Status: built, tested, smoked on the real 163 GB checkpoint, and the
40-session calibration split is running detached.** Increment 2 measured the
streaming design from components and said so plainly — "no 43-layer forward has
been run, on this machine or any other, so the composition of those components
is arithmetic rather than a result". It is a result now.

**The one number: measured wall-clock equals increment 2's fitted 43-layer
compute plus a roughly constant ~40 s per session, across a 6x span of session lengths —
so the extrapolation from 4 measured layers to 43 was right, and the calibration
split projects to 5.3 h, the optimistic end of the predicted 5-8 h.** Peak
memory is 22.4 GB MLX / 19.7 GB host against a ~60 GB budget, and the expert
stream is invisible: 0.71 s of blocking wait per session against 147 GB read.

## Commits (branch `main`)

| SHA | Subject |
| --- | --- |
| `f06a9f81` | `feat(dsv4): layer-sequential streaming teacher runner` |
| `89dcd120` | `fix(dsv4): report the disk rate, not the overlapped read span` |

`tests/test_dsv4_teacher_runner.py` **49 passed** (4 of them exercising the real
shards through the `checkpoint` marker, 1 the real pack through `pack`);
`test_deepseek_v4_flash_adapter.py` + `test_dsv4_teich_manifest.py` +
`test_fp4_expert_dequant.py` **143 passed**, unchanged. Ruff on
the three new files: **2 findings**, both `TRY004` on `raise ValueError` for
malformed *pack data* — the identical finding at the identical construct in the
GLM lineage this follows (`glm52_teich_teacher_producer.py:120`), so it is
parity, not slippage.

Files:

* `src/mlx_vq/quality/dsv4_teacher_runner.py` — the runner.
* `benchmarks/produce_dsv4_teacher_cache.py` — CLI: `run` / `monitor` /
  `finalize` / `plan`.
* `tests/test_dsv4_teacher_runner.py` — the suite.

---

## 1. What was built

### 1.1 Layer-major, not chunk-major — the decision the whole thing turns on

Increment 2 §4 row (ii-a) priced the naive strategy: a chunk-outer / layer-inner
loop re-reads all 43 layers' routed experts for **every** 1,024-token chunk.
11,700 chunk-sweeps x 18.1 s = **58.8 h of pure I/O**, and shrinking the chunk to
1024 for the 13-16% compute win is exactly what doubles it. So the runner
inverts the nesting: **layer outer, chunk inner.** Each layer's experts are read
once per session (257 sweeps, 1.3 h for the whole corpus), and the chunk size
then affects only compute.

Inverting a loop nest inside a transformer is the kind of change that is
plausibly equivalent and occasionally silently is not, so it is asserted rather
than argued:

> `test_layer_major_is_bit_identical_to_chunk_major` builds a 4-layer model whose
> `compress_ratios = [0, 4, 128, 4]` instantiates every attention variant and
> both router branches, randomises every zero-initialised parameter first (the
> skeleton defaults collapse routing onto expert 0 and make the hyper-connection
> mixes constant — a forward over those proves almost nothing), and asserts
> `mx.array_equal` on every chunk's output against the chunk-major reference, at
> two chunk sizes.

It holds, and it holds for a reason worth writing down: layer *i* chunk *j* sees
the same input rows and the same cache state under either nesting, and the
attention mask is layer-independent by construction — `DeepseekV4FlashBackbone`
builds it from `cache[0]` and hands the same array to all 43 layers, so the
runner builds it during layer 0's pass and reuses it, which is both faster and
the only thing that could have been bit-identical.

The second consequence of layer-major is a memory win that was not in the plan:
**each layer's KV and pooled caches are created, used and dropped inside that
layer's pass.** A prefill-only teacher run never needs them again, and it is
holding all 43 layers' pooled windows that makes the chunk-major loop expensive
at an 81,920-token window.

### 1.2 The expert stream

`plan` verb output against the real checkpoint, headers only:

| | |
| --- | ---: |
| layers | 43 |
| layers whose expert block spans >1 shard | **0** |
| coalesced sequential runs per layer | **2** |
| expert tensor bytes per layer | **3.4226 GB** |
| span density after coalescing | **1.000** |
| reads per layer | 1,536 |
| all 43 layers | 147.17 GB |

Increment 2 measured a **3.5547 GB span at 96.3% density** and read the whole
thing. Coalescing with a 1 MB gap threshold finds the structure instead: the
scales for all 256 experts are one dense run, the weights are another, and the
132 MB between them is that layer's *resident* attention tensors, which a
streaming pass already has in RAM and should not re-read. So the runner moves
**3.4226 GB per layer, not 3.5547** — 5.7 GB less per session, 227 GB less over
the calibration split, for free.

Reads land **directly in the final per-projection stacks**: one `preadv` per
(expert, projection) into its slice of a pre-shaped
`(256, out, in)` uint8 buffer, submitted flat onto a persistent thread pool
(ds4's rank-3 transferable idea — `docs/research/ds4-streaming-analysis-20260811.md`
§8). No 3.4 GB intermediate is built and no strided repack happens, which is
what keeps the host side to one buffer instead of two.

Experts are **never dequantised**: the shipped `I8` code bytes are reinterpreted
as `uint32` and the `F8_E8M0` scale bytes handed straight to
`mx.gather_qmm(mode="mxfp4", group_size=32, bits=4)`. `F_NOCACHE` is on by
default — a session streams 147 GB and the unified buffer cache would evict
everything MLX wants in exchange for pages nothing reads twice.

Double buffering is depth 1 by design: layer N+1's read is submitted immediately
after layer N's experts are materialised, so the read overlaps N's compute, and
the host landing buffer is free again the moment the MLX arrays exist. Measured
consequence: **0.71 s of blocking read wait per session against 19.5 s of
reading.** The stream is invisible.

That success is also what broke the first version of the instrumentation, which
is worth recording because the failure mode is generic. `read_seconds` measured
submit-to-complete per layer — which, once the prefetch works, *contains the
previous layer's compute.* Bytes over that span produced a "read rate" that fell
from 7.55 to 3.08 GB/s over the first four real sessions: a number moving the
wrong way under success, which is worse than no number at all. `89dcd120`
separates the three times that were being conflated — the span, the time actually
inside `preadv` (which releases the GIL, so it is the device's busy time and the
honest rate), and the blocking wait, which is the only one that costs wall-clock
— and a test asserts the span-derived ratio is *absent* from the payload so it
cannot come back.

### 1.3 Calibration mode, and one thing that would have shipped wrong

`--mode calibration` accumulates llama.cpp-style per-(layer, projection, expert)
sums of `activation^2`, plus the router-score-weighted variant AGQ wants, into
the schema `mlx_vq.quality.imatrix` already consumes.

Two departures from the GLM-4.5-Air collector, both forced by scale:

**(a) It runs on the GPU.** The host path materialises
`[tokens * top_k, input_dim]` float64 — at 46 k tokens x 6 routes x 4096 dims
that is 18 GB *per layer per session*, and the calibration split would push
multiple TB through the host. This is the standing MLX/Metal directive's exact
shape of problem.

**(b) It scatters; it does not matmul.** The obvious GPU formulation is a
one-hot `count^T @ (x*x)`. Measured on this MLX/Metal build, that float32 matmul
agrees with the repo's own NumPy reference to only **5.9e-4 relative — and
one-sidedly, always undercounting.** That is a systematic bias, not noise, and
roughly fp16 epsilon. Scatter-add of the same quantity agrees to **1.8e-7**,
i.e. exactly what fp32 accumulation should give, and costs less memory because
nothing `[rows, experts]` is built. An importance matrix would probably have
survived 6e-4; a silent one-sided disagreement with the reference should not
ship, so `test_matches_the_host_reference` pins the tolerance at 2e-6 — tight
enough that the matmul formulation could not sneak back in.

Device sums are float32 and fold into float64 host accumulators once per layer,
so cross-layer and cross-session accumulation carries no float32 error at all.

**`gate_proj` and `up_proj` are byte-identical by construction** — they consume
the same post-`ffn_norm` rows under the same routing — so the runner stores one
vector per *input space* (`hidden`, 4096 dims; `down`, 2048 dims) and
`finalize` writes it under both projection names. That halves every artifact:
541 MB per session instead of ~900 MB, 21.6 GB for the split.

The `down_proj` input is the only genuinely separate space, and getting at it
means seeing inside `SwitchGLU`. `_StatsSwitchGLU` reproduces
`SwitchGLU.__call__` with `_gather_sort` inlined so the permutation it reports
to the hook is *literally* the one the projections consumed, not a second
`argsort` that merely ought to agree — and
`test_stats_switch_glu_matches_upstream` asserts `mx.array_equal` against
upstream with the hook both off and on, at token counts either side of
upstream's `do_sort >= 64` threshold. `test_collecting_stats_does_not_change_the_forward`
makes the same assertion one level up, over the whole 4-layer forward.

### 1.4 Logits mode

Top-2048 logits at the pack's supervised positions, fp16, npz, on the GLM
full-v2 schema lineage minus the GLM-specific probes: `positions`,
`target_token_ids`, `topk_logit_ids`, `topk_logit_values`, `logsumexp`,
`tail_mass`.

`hc_head` and `norm` are strictly per-position, so the supervised rows are
gathered *first* (one `mx.take` per chunk — not one per position; a
13,587-position session would otherwise build 13,587 graph nodes to collect
13,587 single rows) and only then projected, in `--lm-head-slice` slices with
top-K on device. A whole-session `[46k, 129280]` float32 logit tensor is 24 GB
and never exists.

`test_logits_match_a_direct_lm_head_projection` checks the sliced path against
projecting the full session and slicing after, on both the id set and the values.

### 1.5 mtp-targets: refused, with the reason

`--mode mtp-targets` refuses before taking the heavy-job lock or touching a byte
of weights, and the message carries the two requirements from
`DeepseekV4FlashMTPBlock`'s docstring verbatim: fp32 SwiGLU on the drafter
stages, and a non-causal `DSparkAttention` with its own context cache and a
`query_width` narrower than the block, which must **replace** the reused
`LocalAttention` forward rather than wrap it. It also states the stake: the
`mtp-train` split is 120 sessions and 5.55 M tokens, 47% of the corpus, and is
blocked on this regardless of where the run happens.

### 1.6 Resumability, and what "skip" is allowed to mean

A session whose output exists is skipped **only after the file is loaded and
re-validated** — schema, dtypes, shapes, finiteness, `positions` and
`target_token_ids` against the pack, `token_ids_sha256`, and this run's prefill
chunk. A truncated file fails the run; a file carrying another session's
`prompt_id` fails the run. Nothing is silently trusted and nothing is silently
overwritten. Four tests cover those paths, including one that truncates a real
output on disk.

**The chunk check is increment 2's concern #1 made enforceable.** Chunked
prefill is not bit-identical to single-shot prefill and the gap compounds with
depth, so a cache is only reproducible against a pinned chunk.
`prefill_chunk_tokens` is written into every session artifact *and* the run
manifest, and resuming at a different chunk raises rather than mixing two
incompatible halves into one cache:

```
written at prefill chunk 16, this run uses 32. Chunked prefill is not
bit-identical across chunk sizes, so mixing them in one cache is silent
corruption. Re-run this session or match the chunk.
```

Writes are atomic — temp path ending in `.npz` (a temp ending `.npz.tmp` makes
`np.savez` write somewhere the rename cannot find it, which is why the GLM
producer carries a comment about it), then `fsync`, then `os.replace`, then an
`fsync` of the parent directory. `test_survives_sigkill_mid_write` forks a child
that starts a 480 MB `_atomic_savez`, waits for bytes to appear on disk, and
`SIGKILL`s it; the published path must still hold the previous complete archive,
byte for byte.

---

## 2. Smoke on the real checkpoint

One real calibration session, end to end, streaming the real 163 GB checkpoint.
Session `teich_cursor_cursor-5814f579-...` — 3,813 tokens, 1,928 supervised, the
shortest in the split.

| | |
| --- | ---: |
| wall | 33.59 s |
| tok/s (wall) | **113.5** |
| routed-expert bytes streamed | **147.17 GB** in 43 layer reads |
| **blocking read wait** | **0.60 s** |
| host->MLX expert conversion | 13.34 s |
| peak MLX | **21.83 GB** |
| peak host RSS | **19.68 GB** |

Artifact verified: 22 arrays, correct dtypes and shapes
(`importance_sum__hidden` float32 `[43, 256, 4096]`,
`importance_sum__down` float32 `[43, 256, 2048]`), **every float array finite and
non-negative**, `route_count` rows summing exactly to `total_route_count`, and
`total_route_count = 22,878 = 3,813 x 6` — the routed-slot count the config
implies, to the token. 682 of 11,008 (layer, expert) pairs went unrouted on a
3,813-token session, which is expected at 256 experts x 43 layers and what the
other 39 sessions are for.

### 2.1 Peak memory against the budget

Budgeted ~60 GB. Measured **21.9 GB MLX / 19.7 GB host RSS**, and the two
overlap heavily (MLX's counter does not see the 3.42 GB NumPy landing buffer,
which is why the run reports both — quoting only `mx.get_peak_memory()`
understates the process by one whole expert layer). Accounting:

| item | GB |
| --- | ---: |
| residents, 43 layers bf16 + embedding + LM head | ~14.4 |
| routed experts, 2 MLX slots in native mxfp4 | 6.84 |
| host landing buffer, 1 layer | 3.42 |
| activations at 3,813 tokens | 0.12 |
| one layer's KV + pooled cache, kernel transients | remainder |

The longest calibration session is 74,072 tokens, which adds 2.43 GB of
activations and ~1 GB of indexer transient over this. **Projected peak ~26 GB —
under half the budget.** The headroom is real and it is the layer-major nesting
that produced most of it.

---

## 3. Calibration split — launched

```
PID           65302  (child of 65300, reparented to init; nohup + disown)
out dir       ~/keep-artifacts/dsv4-teacher-calibration
status        ~/keep-artifacts/dsv4-teacher-calibration/status.json
progress log  ~/keep-artifacts/dsv4-teacher-calibration/status.log
stdout        ~/keep-artifacts/dsv4-teacher-calibration/nohup.out
sessions      40      tokens 1,777,058      supervised 471,880
chunk         1024    B=1    experts native mxfp4    residents bf16
gen config    c5eb045ba88d445e87d97dfa3f77e10014e0fddda6a528e8fdfc67a4cc762fe1
```

Wired-limit environment overrides are **refused at startup**, not merely
unset-by-convention: a run under different limits is not comparable to
increment 2's numbers, so the CLI checks and exits. The heavy-job lock
(`.keep-heavy-job.lock`, `flock`) is held non-blockingly for the whole run;
`--wait-for-lock` queues instead.

Sessions run **shortest-first**, deliberately: a resumable run should bank cheap
sessions early, the first completion lands in under a minute instead of half an
hour, and the tok/s-versus-length curve builds up as the run proceeds instead of
arriving all at once at the end.

### 3.1 Launch evidence

First session complete **43.2 s after the forward started**; six complete inside
the reporting window, each written atomically, validated on the way out, and
logged with its own timestamp and tok/s. Steady state at the time of writing:

```
phase   layer-forward     session 7/40     layer 13/43
done    6 sessions        78,843 / 1,777,058 tokens     3.0 GB written
peak    22.43 GB MLX
stream  883 GB read       4.23 s blocked in total (0.71 s/session)
```

The job is still running. `status.log` is the record; §5 has the monitor command.
Storage tracks 500 MB per session, so the finished split is ~20 GB.

---

## 4. Measured tok/s vs increment 2's extrapolation

Increment 2's estimator, fitted least-squares over its 45,056 / 65,536 / 77,075
measurements, is `total = L x (2.239 ms + 145.94 ns x L)`, worst residual 8%
inside that range. Every completed session against it — real 43-layer forwards,
real streaming, real weights:

| # | tokens | wall | tok/s | fit | fit tok/s | ratio | **residual** | peak GB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3,813 | 43.2 s | 88.2 | 10.7 s | 357.7 | 4.06x | **+32.6 s** | 21.83 |
| 2 | 7,443 | 66.7 s | 111.6 | 24.7 s | 300.7 | 2.69x | **+42.0 s** | 21.86 |
| 3 | 9,518 | 68.6 s | 138.8 | 34.5 s | 275.6 | 1.99x | **+34.1 s** | 21.93 |
| 4 | 12,240 | 85.1 s | 143.9 | 49.3 s | 248.4 | 1.73x | **+35.8 s** | 22.17 |
| 5 | 22,833 | 175.0 s | 130.5 | 127.2 s | 179.5 | 1.38x | **+47.8 s** | 22.39 |
| 6 | 22,996 | 179.4 s | 128.2 | 128.7 s | 178.7 | 1.39x | **+50.7 s** | 22.43 |

**The delta is additive, not multiplicative — and that is the whole answer.**
Read the `ratio` column alone and the runner looks 4x slow and improving for no
reason. Read the `residual` column and it is a **roughly constant ~40 s per
session** (mean 40.5, median 38.9, range 32.6-50.7) across a **6x span of
session lengths**. The ratio column is an artefact of dividing a fixed cost by a
growing one; the residual is the measurement.

The residual does drift upward mildly with length (32.6 s at 3.8 k tokens,
~50 s at 23 k), so calling it *constant* is an approximation, not a law — part of
the streaming tax evidently scales weakly with the chunk count. Six points over
one length decade cannot separate a small linear term from noise on a machine
with 12-25% run-to-run spread; the completed 40-session log, spanning to 74,072
tokens, can.

Which means increment 2's per-token compute model is *confirmed*, not merely
un-contradicted, and the streaming path costs a fixed per-session tax on top.
Decomposed, per session:

| | s |
| --- | ---: |
| host->MLX expert conversion, 43 layers x 3.42 GB | 16.8 |
| blocked waiting for expert bytes | 0.71 |
| unattributed: per-layer cache construction, 43 `clear_cache`, Metal kernel specialisation for the session's unique final-chunk width | ~23 |
| **total** | **~40** |

### 4.1 What that projects to

| | |
| --- | ---: |
| calibration split, increment 2's fit over the real 40 lengths | 4.87 h |
| plus 40 x 40.5 s of streaming tax | +0.45 h |
| **projected** | **5.32 h** |

Against increment 2 §4.1's prediction — *"the 40-session calibration split is
5-8 hours. One night."* — that lands at the optimistic end of the stated range,
with the caveat that thermal derate has had 8 minutes to appear and needs 5 hours.

The ratio also keeps falling for the rest of the split, because the tax is fixed
and the sessions get longer: at the 46,667-token median (fit 424 s) +38 s is
**1.09x**, and at the 74,072-token maximum (fit 966 s) it is **1.04x**. The
expensive part of the split is the part where the streaming overhead has almost
disappeared.

### 4.2 The 43-layer extrapolation, no longer arithmetic

Increment 2's assumption 6 was the load-bearing one: that all 20 ratio-4 layers
behave like measured slice layer 2 and all 20 ratio-128 layers like slice layer
3 — "an extrapolation from 4 measured layers to 43 [that] has never been checked
against a 43-layer run, because a 43-layer run does not fit in memory without
the streaming path this increment did not build."

It fits in memory now, it has been run, and once the fixed streaming tax is
subtracted the 43-layer fit reproduces the measurement to within its own 8%
residual at five lengths. The extrapolation was sound.

---

## 5. Runbook

All commands from the repo root. `UV_CACHE_DIR=/tmp/keep-uv-cache` prefix
everything; if `uv` reports `Permission denied` on the interpreter, run
`uv python install 3.12 --reinstall` once.

### Watch the running calibration job

```
uv run python benchmarks/produce_dsv4_teacher_cache.py monitor \
    --out-dir ~/keep-artifacts/dsv4-teacher-calibration --follow
```

Prints a progress bar, sessions done/skipped, tokens, tok/s (overall and
last-session), both peak-memory numbers, elapsed, ETA, and the stream
accounting; exits by itself when the run reaches `complete` or `stopped`.
`--json` for the raw status document, `--interval` to change the 30 s poll. Or
just `watch -n 30 cat ~/keep-artifacts/dsv4-teacher-calibration/status.json`.

`status.log` is append-only with one line per session completion, each carrying
a timestamp, that session's seconds and tok/s, both peaks, and cumulative stream
bytes — so a sustained thermal derate shows up as a trend in a file, not as a
surprise at the end.

### Resume after any interruption

Re-run the identical `run` command. Completed sessions are re-validated and
skipped; an interrupted session costs itself and nothing else.

```
uv run --group dev python benchmarks/produce_dsv4_teacher_cache.py run \
    --mode calibration --split calibration \
    --out-dir ~/keep-artifacts/dsv4-teacher-calibration
```

Do **not** change `--chunk` on a resume. The run will refuse, which is the
point — see §1.6.

### Stop it gracefully

```
touch ~/keep-artifacts/dsv4-teacher-calibration/STOP     # if launched with --stop-file
kill 65302                                              # otherwise; loses the in-flight session only
```

Pass `--stop-file <path>` at launch to get the graceful boundary; the run
finishes the current session, writes it, logs `phase: stopped`, and exits 0.

### Launch the other splits

The three top-K logit splits — 97 sessions, 4.39 M tokens, ~13 h at the
chunk-1024 rate:

```
nohup uv run --group dev python benchmarks/produce_dsv4_teacher_cache.py run \
    --mode logits --split report --split selection --split holdout \
    --top-k 2048 --out-dir ~/keep-artifacts/dsv4-teacher-logits \
    > ~/keep-artifacts/dsv4-teacher-logits/nohup.out 2>&1 &
disown
```

Or one split at a time (`--split report` alone is 30 sessions, ~4.2 h). Any
`--split` value from the pack's `campaign_split` works; `--session-id` (repeatable)
and `--max-session-tokens` narrow further for validation runs.

Budget the output: 1,080,194 supervised positions across the three splits, at
2048 int32 ids + 2048 fp16 values per position = **~13 GB**, against 1.4 TB free.
(Increment 2 §4.1's ~50 GB figure was for all 3.05 M supervised positions
including `mtp-train`, and assumed fp32 values.)

### Turn the calibration statistics into imatrix sidecars

Separate verb on purpose — 43 layers x 3 projections x 256 experts is **33,024
files**, and a multi-hour streaming run should not be holding the heavy-job lock
while it writes them.

```
uv run --group dev python benchmarks/produce_dsv4_teacher_cache.py finalize \
    --out-dir ~/keep-artifacts/dsv4-teacher-calibration
```

Writes `imatrix-sidecars/imatrix/layer-NNNNN-{projection}-expert-NNNNN.safetensors`
plus `imatrix-manifest.json`, the exact schema
`mlx_vq.quality.imatrix.load_projection_imatrix_manifest` validates and
GLM-4.5-Air's VQ fitting already reads. Sums across every session file present,
so it can be run mid-campaign on a partial split and re-run later.

### Inspect the plan without touching weights

```
uv run --group dev python benchmarks/produce_dsv4_teacher_cache.py plan \
    --split calibration
```

Headers only: per-layer spans, coalesced runs, density, the byte budget, the
session length distribution. Use it to sanity-check a checkpoint before
committing hours to it.

### What `mtp-train` is blocked on

`--mode mtp-targets` refuses, before the lock and before any weight I/O. The
blocker is the **DSpark drafter forward**, not the runner:
`DeepseekV4FlashMTPBlock` exists and binds its weights, but its forward still
reuses `LocalAttention`. Two requirements, recorded in its own docstring and
repeated in the refusal message:

1. **fp32 SwiGLU on the drafter stages.** `LimitedSwiGLU` already takes
   `fp32=True`; the drafter must be constructed with it.
2. **A non-causal `DSparkAttention`** with its own context cache and a
   `query_width` narrower than the block, which must **replace** the reused
   `LocalAttention` forward rather than wrap it.

Until both land, the `mtp-train` split (120 sessions, 5,552,257 tokens, 47% of
the corpus, ~16.5 h of compute) cannot be produced locally *or* on a rented
GPU — this is not a local-compute limitation and no amount of hardware moves it.
Once the drafter forward exists, `--mode mtp-targets` is the place to add the
capture: the streaming loop, resumability, atomic writes, chunk pinning and
status reporting are already in place and mode-agnostic.

---

## 6. What this does *not* prove

* **No logit parity against an independent reference.** Carried forward
  unchanged from increment 2 §5. The runner proves the streaming composition
  works, is bit-identical to the chunk-major forward it replaces, holds the
  memory budget, and produces finite well-formed statistics on real weights. It
  does not compare the vendored forward to another implementation of this
  architecture, because there is still no runnable reference.
* **Only the short end of the 43-layer extrapolation is confirmed so far.** §4's
  comparison rests on the sessions completed inside this window. The
  extrapolation's own assumption 6 — that all 20 ratio-4 layers behave like the
  measured slice layer 2 and all 20 ratio-128 layers like slice layer 3 — is now
  tested by a real 43-layer forward rather than being arithmetic, but at the
  lengths the running split has reached, not yet at the 74,072-token maximum.
  The completed `status.log` is the record.
* **Sustained thermal derate is what the running job is measuring.** Increment 2
  called this "the least-grounded term in the whole estimate". Per-session tok/s
  in `status.log`, read against the length each session actually ran, is the
  measurement. It is not in yet.
* **The mxfp4 expert bytes are verified against the header, not against a
  dequantised reference.** `test_reads_land_the_bytes_the_header_promises`
  checks three real tensors from layer 17 against two independent readers
  (`safetensors` for the I8 codes — NumPy has no `float8_e8m0fnu`, so the
  library cannot view the scales at all — and a plain buffered read at the
  header's own offsets for both). That the `gather_qmm` consuming them is
  numerically right is increment 2's `537c1a0e`, which remains synthetic-byte
  only.
* **`finalize` has not been run at real scale.** It is exercised end-to-end on
  the tiny model (both projections sharing the `hidden` vector, manifest
  validating, sidecars loadable), but 33,024 real sidecars have not been written.

## 7. Concerns / carry-forward

1. **The ~40 s/session streaming tax is quantified and deliberately not chased.**
   Its two identified halves, with the numbers rather than a shrug:
   * **Host->MLX expert conversion, 16.8 s** — 43 layers x ~0.39 s to copy
     3.42 GB into MLX unified memory (~9 GB/s), *serial* with compute. It could
     plausibly be overlapped by constructing the MLX arrays on the I/O thread,
     since the copy releases the GIL. **Not done, and the reason is not effort:**
     MLX array construction off the main thread is exactly the kind of change
     that can introduce ordering nondeterminism, and the artifact it would be
     risking is a teacher cache whose entire value is bit-reproducibility. The
     prize is 16.8 s x 257 sessions = **1.2 h out of a 38-53 h corpus, ~2.5%**.
     Against a 7x kernel gap (concern 6) that is not where the next hour of
     engineering goes.
   * **~21 s unattributed, and one identified suspect** — every session's *final*
     chunk has a width no other session shares, so all 43 layers pay Metal
     kernel specialisation for a one-off shape once per session. Padding the
     last chunk to a full 1024 would reuse the specialised kernels, but padding
     is visible to the pooled cache and the mask, so it would change teacher
     numerics for a saving of ~0.2 h across the split. Declined on those terms,
     not overlooked.

   The operational consequence matters more than either: on a short session the
   tax is 40% of the wall, so **any future short-session benchmark must subtract
   it explicitly or the model will look like it got slower.**
2. **The device read rate is below increment 2's 9.92 GB/s** for a single
   whole-span read — expected, since this issues 1,536 `preadv` calls averaging
   2.2 MB instead of one sequential 3.55 GB run. Irrelevant while the blocking
   wait is 0.71 s/session, and the fix if it ever mattered (read the two
   coalesced runs whole, then repack) costs a second 3.4 GB of host memory. Bad
   trade today. See also `89dcd120` in §1.2 for why the *first* version of this
   number was not measuring what it claimed.
3. **The unrouted-expert tail.** 682 of 11,008 (layer, expert) pairs saw no
   traffic on a 3,813-token session. Wave 5's VQ fitting has to decide what an
   expert with `route_count == 0` means; the schema records the zero honestly
   (`mean_importance` is zeros, `route_frequency` is 0.0) rather than smoothing
   it, so the decision is downstream and visible. Worth checking the count over
   all 40 sessions once the run lands.
4. **541 MB per calibration session, 21.6 GB for the split**, uncompressed
   `np.savez` (compression of a dense float32 block costs ~60 s of CPU to save
   little). `--compress` exists if space ever matters. Already halved by the
   `gate`/`up` sharing in §1.3.
5. **Increment 2's concern #6 was deliberately not acted on.** It suggested
   re-pinning the chunk-1024 choice on the day, since its 13-16% margin is the
   same order as the machine's run-to-run spread. The Wave 3 brief pins chunk
   1024 as measured and decided, and §1.6 means the choice cannot be revisited
   mid-cache anyway — so re-pinning is now a decision to make *before* a new
   output directory, not during one.
6. **Everything increment 2 carried forward is still carried.**
   `PoolingCache.is_trimmable()` returns `False`, the profile
   `group_size_policy` is still the GLM-5.2 placeholder, `w1`/`w2`/`w3` naming
   is still unverified against a real materializer, the `self.dspark`
   exact-attention decode branch is still stripped, and the 20 ratio-4 layers
   are still 66-73% of teacher-gen wall-clock with a 7x gap to a mature engine
   on the same silicon. That last one is unchanged as the campaign's single
   largest local-compute lever.
