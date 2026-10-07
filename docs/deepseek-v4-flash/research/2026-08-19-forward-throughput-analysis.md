# DSV4-Flash VQ forward throughput: decomposing the 3.3x

**Date:** 2026-08-19 · **Campaign:** `docs/deepseek-v4-flash/2026-08-11-campaign-plan.md`
**Status:** analysis only — no GPU work was run. A quality-gate eval (PID 4976) held
the heavy-job lock throughout at ~23.5 tok/s; every number below is read from
telemetry already on disk.

## 0. Verdict up front

The 3.3x is **not** metric overhead and **not** artifact I/O. Both of those are
measured, and both are negligible or favour the VQ path. The gap is the E8P
kernel, and it is real.

| Term | Factor | Kind |
| --- | ---: | --- |
| Prompt mix + harness forward overhead (teacher runner → eval harness, source model) | 1.25x | **harness-only** |
| ├ metric computation (KLD / top-k / NLL) | 1.024x | harness-only |
| └ unattributed harness forward delta | 1.227x | harness-only |
| VQ vs source **inside the same harness, same 8 sessions** | **2.40x** | **real forward cost** |
| ├ artifact I/O | ~1.00x (favours VQ) | — |
| ├ host→device convert | ~1.00x (favours VQ) | — |
| └ `gather_vqmm` vs native mxfp4 matmul | **~2.49x** | **real** |
| Session mix + machine contention (stratified run → full run) | 1.09x | measurement condition |
| **Total** | **3.28x** | 77.0 → 23.5 tok/s |

**The campaign's headline is a speed claim, and the speed claim is in trouble.**
Not because of this eval — this eval measures *prefill*, which is not what the wow
benchmark sells — but because the decode-regime evidence already on disk points
the same direction and harder. See §5. If you read only one section, read §5.

---

## 1. The measurement chain

The premise handed to this investigation ("96.6 / 78 tok/s vs 23.5 tok/s") compares
two runs that differ in *three* ways at once: the model, the harness, and the prompt
mix. The artifacts on disk let all three be separated, because
`~/keep-artifacts/dsv4-quality-gate/rows.jsonl` contains **both** engines run through
**the same harness** over **the same 8 sessions / 361,366 tokens**.

Matching the 8 prompts across all three runs
(`~/keep-artifacts/dsv4-teacher-logits-eval/status.log` vs
`~/keep-artifacts/dsv4-quality-gate/rows.jsonl`):

| Stage | Seconds (8 sessions) | tok/s | Step factor |
| --- | ---: | ---: | ---: |
| Teacher runner, source mxfp4 | 4,693.8 | 77.0 | — |
| Eval harness, source mxfp4 | 5,870.7 | 61.6 | 1.251x |
| Eval harness, VQ E8P | 14,105.9 | 25.6 | 2.404x |
| Full 30-session run (in flight) | — | 23.5 | 1.090x |

1.251 × 2.404 × 1.090 = **3.28x**, which reproduces the reported 3.3x exactly.

### 1a. The 78.2 tok/s planning basis was never a constant

The eval plan assumed 78.2 tok/s. That figure is a *cumulative average* sampled at
one point (`session_index=45`) of a teacher run whose **per-session** rate ranged
from **33.9 to 189.3 tok/s** — a 5.6x swing (`dsv4-teacher-logits-eval/status.log`).
The same run's `read_wait_seconds` climbed monotonically from 0.36 s to 292 s across
its second phase, i.e. the run itself degraded under I/O contention as it went.

On the 8 sessions the quality gate actually uses, the teacher runner averaged
**77.0 tok/s** — so 78.2 was a fair number for *that* model on *that* mix, and the
4.92 h estimate error is not a mis-estimate of the source model. It is the 2.40x VQ
term plus the 1.25x harness term, neither of which the plan modelled.

---

## 2. Term 1 — metric computation: measured, and it is ~2%

The eval times the forward and the metrics with two disjoint clocks:

- `src/keep/quality/dsv4_teacher_agreement.py:1590-1599` — `forward_seconds` wraps
  **only** `layer_major_prefill`.
- `src/keep/quality/dsv4_teacher_agreement.py:1603-1612` — `metric_seconds` wraps
  the `_student_logit_slices` → `_SessionAccumulator.add` loop.

`forward_seconds + metric_seconds` reproduces the session wall time to within 0.2 s
in every row, so the split is complete and there is no unaccounted third bucket.

Summed over the 8 matched sessions:

| Engine | forward_s | metric_s | metric share |
| --- | ---: | ---: | ---: |
| source_mxfp4 | 5,758.1 | 112.6 | **1.92 %** |
| vq_e8p | 13,998.9 | 107.0 | **0.76 %** |

The metric cost is not merely small, it is **essentially identical in absolute terms
between the two engines** (112.6 s vs 107.0 s) — as it must be, since it is the same
top-2048 comparison against the same teacher `.npz`. It therefore contributes
**0.99x** to the VQ/source ratio, i.e. it very slightly *favours* VQ.

**On the specific things the brief asked to look for:**

- *Per-position Python loops:* none. `_student_compact_summary`
  (`:1009-1036`) does every reduction on-device — `logsumexp`, `take_along_axis`,
  `argsort` — and the host sees only `[positions, K]` arrays.
- *Host syncs:* three `mx.eval` calls per position slice (`:994`, `:1003`, `:1030`)
  and four `np.asarray` copies (`:1032-1035`). Real, but amortised over
  `position_slice = 256` positions, and the totals above bound the whole cost at
  ~2%.
- *`clear_cache` in a hot loop:* yes — `mx.clear_cache()` per position slice at
  `:1006`. This is the same anti-pattern fixed once in the teacher runner. With
  ~19,800 positions / 256 that is ~78 calls per session. It is inside the 0.76–1.92%
  bucket, so **fixing it cannot pay back more than ~2%**.

> **Term 1 is closed. Metrics are not the story, and no amount of metric optimisation
> will move the eval's ETA meaningfully.**

---

## 3. Term 2 — artifact I/O: the VQ path is *better*, not worse

The premise expected the 138-file VQ layout to read worse than the source's single
contiguous span. The telemetry says the opposite on every axis.

Both engines share one streaming loop (`layer_major_prefill`,
`src/keep/quality/dsv4_teacher_runner.py:1097`) and one stats object
(`Dsv4StreamStats`, `:521-554`). Per 43-layer pass, from `rows.jsonl` deltas:

| | source mxfp4 | VQ E8P | |
| --- | ---: | ---: | --- |
| bytes/layer | 3.4226 GB | **1.6358 GB** | VQ reads 2.09x less |
| bytes/session (43 layers) | 147.17 GB | **70.34 GB** | |
| `preadv` calls per layer | **1,536** | **6** | |
| read sizes | 768 × 4 MiB + 768 × 256 KiB | 3 × 512 MiB + 3 × 8 MiB | |
| `io_busy_seconds` (thread-seconds) | ~100 | **~17** | |
| `read_wait_seconds` | 0.38–0.92 | **0.13–0.23** | |
| `convert_seconds` | 17.5–25.6 | **7.2–14.1** | |

**`read_wait_seconds` is the decisive field.** It is the time the compute loop
actually blocked on I/O (`dsv4_teacher_runner.py:722`). Against forward times of
145–3,074 s per session, it is **sub-second in both engines**. Prefetch depth 1 with
an 8-thread pool is fully hiding the reads. I/O contributes **~0.0%** to the gap.

Corrections to the figures in the brief, from the code and manifests:

- Source expert spans are **147.17 GB/session**, not ~157 GB.
- The VQ artifact is 75.25 GB on disk, but the eval opens only the **129 backbone
  files (43 layers × 3) = 70.34 GB**. The 9 `mtp-*` files (~4.91 GB),
  `imatrix-cache.npz`, and the roundtrip JSONs are never read.
- The VQ path reads **no** sidecars beyond scales: scales are 25.17 MB/layer
  (1.54% of the payload); the codebook is ~1 KB read **once per run**
  (`read_vq_codebook`, `dsv4_teacher_agreement.py:592-607`). The manifest confirms
  `continuous_sidecar_count: 0`, `dense_routed_experts: false`, `fallback_layers: []`.
- The span index is built **once per process**, not per session
  (`build_dsv4_vq_block_span_index`, `:507-589`, called from `:898-902` before the
  session loop). No per-chunk rebinding, no file reopening, no per-session index
  rebuild.

### 3a. One real telemetry defect (does not affect the conclusion)

`io_gb_per_s` divides by `io_threads` unconditionally:

```python
busy_wall = self.io_busy_seconds / max(self.io_threads, 1)   # dsv4_teacher_runner.py:545
"io_gb_per_s": round(gb / busy_wall, 2) if busy_wall > 0 else None,
```

The source path issues 1,536 reads per layer and genuinely saturates all 8 threads,
so its reported 9.9–11.0 GB/s is honest. The VQ path issues only **6** futures per
layer, of which **3** carry bulk bytes — so its reported **33.0 GB/s is inflated
~2.7x** and is not comparable to the source number despite the class docstring
claiming it is (`dsv4_teacher_agreement.py:671-679`). Corrected wall throughput is
~12.3 GB/s (VQ) vs ~11.8 GB/s (source): comparable, with VQ still ahead per byte
(4.12 vs 1.47 GB/s **per stream**).

> **Term 2 is closed, with the sign flipped. The VQ artifact layout is a modest win.
> Do not spend effort here.**

---

## 4. Term 3 — `gather_vqmm` vs native mxfp4: this is the whole gap

By subtraction, using the same 8 sessions:

| | source | VQ | ratio |
| --- | ---: | ---: | ---: |
| forward_seconds | 5,758.1 | 13,998.9 | 2.43x |
| − convert_seconds | 168 | 72 | |
| **= compute** | **5,590** | **13,927** | **2.49x** |

Since `read_wait ≈ 0` and metrics are timed separately, **~2.49x of pure arithmetic
is the entire discrepancy.** The VQ path reads half the bytes, waits less on I/O,
converts less, uses ~7 GB less RSS (18.24 vs 25.27 GB) — and still takes 2.4x
longer.

### 4a. Which kernel this actually is

The eval binds `QuantizedVQSwitchLinear` with `route_strategy` left at its default
`"auto"` (`switch_linear.py:108`); the bind proof records
`"route_backend": "gather_vqmm_auto"`. `auto` resolves at
`src/ramp/ops/vq_switch.py:926-942`:

```python
if sorted_indices or route_count >= _SORTED_TILED_ROUTE_THRESHOLD:   # 128
    selected_strategy = "sorted_tiled"
elif auto_selects_per_route_decoded(...):
    selected_strategy = "per_route_decoded"
else:
    selected_strategy = "direct"
```

Prefill runs `chunk = 1024` tokens at top-6 → `route_count = 6144 ≫ 128` → the eval
is measuring **`sorted_tiled` at M=1024**. That is the *good* case: at M=1024 each
codeword read is amortised over 1024 rows, so the kernel is ALU-bound, not
bandwidth-bound, and the 2.49x is the raw cost of decoding E8P codewords versus
MLX's native mxfp4 dequant.

**The eval never exercises the decode kernel at all.** That matters enormously, and
it is §5.

### 4b. The artifact is `code_bits=16`, which locks out every fast kernel

From `run-vq_e8p_streamed.json` and `bind-proof-vq_e8p_streamed.json`:
`code_bits: 16`, `group_size: 512`, `routed_bpw: 2.03125`.

Three separate fast paths are gated on `code_bits == 8` and are therefore
**structurally unreachable for the shipped artifact**:

- `gather_vqmm_m1_kernel` — `switch_linear.py:335-340` requires `self.code_bits == 8`.
- `gather_vqmm_m1_per_route_kernel_unchecked` — `switch_linear.py:377-383`, same gate.
- `per_route_decoded` — `vq_switch.py:413`: `if implementation != "metal" or code_bits != 8: return False`.

The docstring is explicit about the consequence
(`vq_switch.py:376-393`):

> "Both spend one 256-thread threadgroup per output element per route; Wave 4
> increment 1 measured that structure at 3.2-4.5x slower than the per-route decoded
> kernel … **code_bits=16 (E8P) has no decoded verify kernel, so it stays scalar.**"

So the Wave 4 "3.2–4.5x from dispatch alone" win that
`docs/deepseek-v4-flash/research/wave4-increment1-report.md:238` says the compounding argument
"survives via" is **explicitly `code_bits=8`-only**
(`docs/deepseek-v4-flash/research/wave4-wide-m-vqmm-survey.md:325`). It does not apply to this
artifact. That is a bookkeeping error in the campaign's own survival argument, and
it should be corrected before Wave 5 depends on it.

---

## 5. The decode regime — where the thesis actually lives

Everything above measures **prefill**. The compounding thesis is about **decode**:

```
tok/s ≈ (bytes moved per token)⁻¹ × (accepted tokens per verify pass)
```

At decode, top-6 of 256 experts, 43 layers, the DSV4 per-token expert traffic is:

| | bytes/token | |
| --- | ---: | --- |
| source mxfp4 | 3.449 GB | |
| VQ E8P | **1.649 GB** | **2.09x less** |

That 2.09x is the entire left-hand factor of the thesis, and the artifact delivers
it. The question is whether the kernel can *spend* it.

### 5a. Direct decode evidence already on disk says no

`artifacts/benchmarks/glm45-air-decode-component-profile.jsonl` is a per-component
decode profile of GLM-4.5-Air, same harness, same prompt, same 128 generated tokens,
varying **only** the expert quantization. Per-call GPU section times:

| component | VQ E8P (1.036 bpw) | MLX affine 2-bit (2.25 bpw) | VQ/MLX |
| --- | ---: | ---: | ---: |
| `moe.down_proj` | 3.3500 ms | 0.8180 ms | **4.10x** |
| `moe.gate_proj` | 1.9163 ms | 0.7797 ms | **2.46x** |
| `moe.up_proj` | 1.3286 ms | 0.7679 ms | **1.73x** |
| `moe.router` | 0.8248 ms | 0.6769 ms | 1.22x |
| `attention` | 1.9922 ms | 1.5788 ms | 1.26x |
| `mlp` | 1.7076 ms | 1.3583 ms | 1.26x |

Attention / mlp / router use identical weights in both runs, so their consistent
~1.25x is a **run-condition offset** and must be divided out. Correcting for it:

- Routed-expert projections: 37.99 s vs 13.63 s = 2.79x raw → **~2.23x** offset-corrected.
- Bytes moved: 12.91 GB vs 26.47 GB = **2.05x fewer** for VQ.
- **Effective bytes/second deficit vs MLX's native quantized matmul: ~4.6x**
  (5.72x uncorrected).
- End-to-end decode: **2.02 tok/s (VQ) vs 4.11 tok/s (MLX 2-bit)** — VQ carries
  2.17x fewer bits per weight and decodes **2.03x slower**.

This is the thesis's left-hand factor being *consumed and overdrawn* by the kernel.

And note the direction of the bias: that profile ran **`code_bits=8`**, which gets
`gather_vqmm_m1_decoded.metal` — the *fastest* M=1 path KEEP has. The DSV4 artifact
is `code_bits=16` and falls to the **scalar** `direct` kernel, which Wave 4 measured
at a further **3.2–4.5x** off the decoded kernel (§4b). **The DSV4 decode path is
strictly worse than the one that produced the 2.03x loss above.**

### 5b. The mechanism, and why it will not fix itself

`docs/deepseek-v4-flash/research/wave4-wide-m-vqmm-survey.md:55-60, 154-156`:

> "Both baseline and verify kernel sit at **32–71 GB/s** of code-read bandwidth
> (median by M: 71 at M=2, 50 at M=4, 32 at M=8) on a machine with roughly
> 500 GB/s, and neither ALU nor bandwidth is saturated. The lane discipline is
> latency-bound."

At 6–14% of achievable bandwidth, a 2.09x reduction in bytes cannot win. The survey
also records that the two obvious ILP fixes were **tried and measured**: vectorizing
the activation load "gained almost nothing", and unrolling the codeword walk 2/4/8-deep
"made things" worse. The remaining proposal — widening code reads to `uint`/`uint2`
so a lane owns 4–8 consecutive codewords — **changes the byte-exact reference**
(`:335-336`), so it is not a free win either.

### 5c. What this means for the compounding thesis — plainly

Taking the offset-corrected decode numbers at face value and assuming DSV4's
`code_bits=16` scalar path is merely *as bad as* the `code_bits=8` decoded path
(a generous assumption — §4b says it is 3.2–4.5x worse):

- VQ decode is **~2.0–2.2x slower per token** than a native-quantized baseline,
  despite moving 2.05–2.09x fewer bytes.
- MTP would have to deliver **>2.0–2.2 accepted tokens per verify pass just to reach
  parity**, before delivering any net win at all.
- Published MTP acceptance at depth 1–2 is typically 1.5–2.5x. **The entire MTP gain
  is spent paying off the kernel, and the compounding claim nets out to roughly
  1.0x.**
- If the `code_bits=16` scalar penalty is real on top of that, the product is
  **net negative** — the compressed model would be slower than the source it
  compresses, and no MTP depth recovers it.

**The compressed model is, on current kernels, intrinsically slower to forward than
the FP4 source. This is not measurement scaffolding.** The eval harness costs are
real but small (1.25x, §2 and §6); strip them all and the compressed artifact is
still 2.4x slower at prefill and — on the closest decode evidence available —
~2x slower at decode while carrying half the bytes.

The campaign can still honestly claim a **memory/footprint** win (2.03 bpw, 2.09x
fewer bytes/token, 7 GB less resident). It cannot currently claim a **speed** win.

---

## 6. Harness-only vs real: the split that matters

**(a) Eval-harness-only — will NOT affect the wow benchmark:**

| Cost | Size | Evidence |
| --- | ---: | --- |
| Metric computation (KLD/top-k/NLL vs teacher top-2048) | 1.024x | §2 |
| `mx.clear_cache()` per position slice | ⊂ above | `:1006` |
| Retaining all chunk hidden states for the supervised gather | ⊂ 1.227x | `layer_major_prefill` returns per-chunk hidden; page-outs 496–9,466/session |
| Unattributed harness forward delta | 1.227x | §1, **not separated — see §7** |
| Prompt-mix / contention between runs | 1.09x | §1 |

**(b) Real forward-path — WILL affect the wow benchmark:**

| Cost | Size | Evidence |
| --- | ---: | --- |
| `gather_vqmm sorted_tiled` vs native mxfp4, prefill M=1024 | **2.49x** | §4 |
| `gather_vqmm direct` (scalar) at decode M=1, `code_bits=16` | **≥2.0x**, plausibly 3–5x | §5, bounded not measured |
| Code-read bandwidth 32–71 GB/s vs ~500 available | mechanism | Wave 4 survey |

Note that (a) totals 1.25x and is entirely on the *source-vs-VQ-neutral* side: it
inflates both engines equally. Removing all of it would take the eval from 23.5 to
~29 tok/s. It would not change the 2.40x.

---

## 7. What cannot be separated without the GPU

Three questions are unresolved and each needs the machine free. All are small,
minutes-scale jobs.

1. **The 1.227x harness forward delta on the source model.** `forward_seconds`
   covers identical work (`layer_major_prefill`) in both the teacher runner and the
   eval, yet the eval is 22.7% slower on the same 8 sessions. Candidates: the eval
   retains every chunk's hidden state for the later supervised gather (page-outs are
   non-zero in every row, `memory_clean: false`), versus machine-condition drift.
   **Experiment:** run the eval harness on 2 short sessions with
   `--engine source_mxfp4_streamed` twice — once as-is, once with the metric loop
   stubbed and `hidden` dropped per chunk — on a quiet machine. ~10 min.

2. **The DSV4 decode ratio.** Everything in §5a is GLM-4.5-Air at `code_bits=8`.
   **Experiment:** `bench_gather_vqmm.py` / `bench_vq_qmv.py` at the DSV4 routed
   shapes (K=4096→N=2048 gate/up, K=2048→N=4096 down), 256 experts, top-6, M=1,
   `code_bits=16`, `route_strategy` ∈ {`auto`, `direct`, `sorted_tiled`}, against
   MLX native mxfp4 `QuantizedMatmul` at the same shapes. This is the single
   highest-value measurement in the campaign — it settles the speed claim. ~15 min.

3. **Whether a `code_bits=16` decoded M=1 kernel closes the gap.** Wave 4's 3.2–4.5x
   dispatch win exists only for `code_bits=8`. **Experiment:** before writing any
   Metal, benchmark a `code_bits=8` DSV4 artifact slice against the `code_bits=16`
   one at M=1 to size the prize. If the answer is "still 2x slower than mxfp4", the
   decoded kernel is not worth building.

---

## 8. Ranked fixes

Ordered by expected gain to the **campaign thesis**, not to the eval's ETA.

| # | Fix | Expected gain | Risk | Affects |
| --- | --- | --- | --- | --- |
| 1 | **Measure the DSV4 M=1 decode ratio (§7.2) before any further optimisation.** The campaign is defending a speed claim it has never measured in the regime it sells. | decision-quality | none | thesis |
| 2 | **Correct the Wave-4 survival argument.** `docs/deepseek-v4-flash/research/wave4-increment1-report.md:238` claims the thesis survives via a 3.2–4.5x dispatch win that `vq_switch.py:413` restricts to `code_bits=8`. The shipped artifact is `code_bits=16`. | none (correctness) | none | thesis |
| 3 | **Attack the latency-bound codeword walk** (`uint`/`uint2` widened code reads, lane owns 4–8 consecutive codewords). Wave 4 names this as the only remaining headroom: 32–71 → target ~200+ GB/s. | up to ~3x on the real term | **high** — changes the byte-exact reference; needs a parity re-baseline in lockstep at M=1 | real forward |
| 4 | **Build a `code_bits=16` decoded M=1/per-route kernel**, or ship a `code_bits=8` artifact variant to unlock the three existing fast paths. | 3.2–4.5x at decode *if* §7.3 sizes it | medium — new Metal, or a requantization campaign | real forward |
| 5 | Split each 512 MiB VQ codes `preadv` into ~4 chunks to raise queue depth 3 → 12. | ~0 — I/O is already fully hidden (`read_wait ≈ 0.15 s`) | low | neither. **Do not do this.** |
| 6 | Fix `io_gb_per_s` to divide by *active* streams, not `io_threads` (`dsv4_teacher_runner.py:545`). | 0 perf; stops a 2.7x-inflated number misleading the next investigation | none | telemetry |
| 7 | Hoist `mx.clear_cache()` out of the per-position-slice loop (`:1006`). | ≤2% of eval wall | low | harness only |
| 8 | Re-baseline the eval's ETA on 25.6 tok/s (stratified, measured) rather than 78.2 (a moving cumulative average from a different model and mix). | estimate accuracy | none | planning |

Items 5 and 7 are listed to be explicitly **de-prioritised**: they are where the
premise pointed, and the telemetry says they are worth ~0% and ~2% respectively.

---

## 9. Provenance

All figures are read from files already on disk; nothing here was benchmarked for
this document. The machine was under load average ~18 with a heavy eval holding the
GPU for the entire analysis window, so any timing quoted from the **in-flight** run
(the 23.5 tok/s column) is a lower bound on speed. The 8-session comparison
(§1, §2, §3, §4) is drawn from two **completed** runs and is unaffected.

Primary sources:

- `~/keep-artifacts/dsv4-quality-gate/rows.jsonl` — 26 rows, per-session
  `forward_seconds` / `metric_seconds` / `stream{}` for both engines.
- `~/keep-artifacts/dsv4-quality-gate/{src,vq,vq-full}-status.log`
- `~/keep-artifacts/dsv4-teacher-logits-eval/status.log` — 97 sessions.
- `~/keep-artifacts/dsv4-quality-gate/{run,bind-proof}-*.json`
- `artifacts/benchmarks/glm45-air-decode-component-profile.jsonl`
- `docs/deepseek-v4-flash/research/wave4-wide-m-vqmm-survey.md`
- `src/keep/quality/dsv4_teacher_agreement.py`,
  `src/keep/quality/dsv4_teacher_runner.py`,
  `src/ramp/ops/vq_switch.py`, `src/ramp/nn/switch_linear.py`
