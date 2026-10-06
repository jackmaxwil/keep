# Wave 5 pilot — the sweep is one overnight, and group size is a rate dial rather than a quality lever

**Status: complete. The 43-layer schedule the campaign asked for is published
below: 11.2 h for the whole materialization, peak RSS 4.3 GB, artifact 83.6 GB —
one unattended overnight and comfortably inside the 98–105 GB envelope, against
a campaign budget of "multiple overnights". The wall-clock was never the risk.**

**The two findings that should change a decision are both about the ladder, not
the clock. First, on this expert geometry the group-size knob returns 0.17–0.24
fractional error per bpw while the codebook step returns 0.71, so the GLM-5.2
placeholder's *value* (512) survives — but for a reason nobody had measured, and
its companion `default_code_bits: 8` would have shipped a model with a 0.56
block-level cosine. Second, E8P beats affine quantization everywhere below
~3.1 bpw and loses above it, and the local envelope caps the routed rate at
~2.61 bpw — so the whole local deliverable lives on the E8P side of the
crossover and the campaign's "3.0 bpw ladder" does not fit the machine at all.**

## Commits (branch `main`)

| SHA | Subject |
| --- | --- |
| `dded37e0` | `feat(dsv4): Wave 5 VQ materialization pilot` |
| _this_ | `docs(sdd): Wave 5 pilot report -- group size is a rate dial` |

`tests/test_dsv4_vq_pilot.py` + `tests/test_model_profiles.py` **73 passed**.
Ruff clean on all three new files.

Files:

* `src/mlx_vq/convert/dsv4_vq_pilot.py` — the measurement logic.
* `benchmarks/pilot_dsv4_vq_materialization.py` — CLI: `sweep` / `control` /
  `block-probe` / `layer-fit` / `schedule` / `monitor`.
* `tests/test_dsv4_vq_pilot.py` — 57 headless tests (no checkpoint, no GPU).
* `models/deepseek-v4-flash-0731.yaml` — policy replaced with the derivation.

Evidence committed to `artifacts/quality/dsv4-vq-pilot-20260813/` (72 KB,
force-added per the directory's convention) so the profile change above stays
verifiable: `group-size-sweep.json`, `affine-control.json`, `block-probe.json`,
`layer-fit-{1,20}.json`, `schedule-resident-q{85,65}.json`.

Reproduce:

```
uv run --group dev python benchmarks/pilot_dsv4_vq_materialization.py sweep \
  --layers 1 20 --experts 16 --iterations 8 --out <out>/group-size-sweep.json
uv run --group dev python benchmarks/pilot_dsv4_vq_materialization.py layer-fit \
  --layer 20 --iterations 8 --out <out>/layer-fit-20.json
uv run --group dev python benchmarks/pilot_dsv4_vq_materialization.py schedule \
  --layer-fit <out>/layer-fit-20.json <out>/layer-fit-1.json --out <out>/schedule.json
```

Everything composes shipped machinery: the fit is
`quantize_weight_importance_aware` (the GLM lineage's alternating weighted
assignment / exact weighted-least-squares scale update), the objective is the
llama.cpp imatrix diagonal `sum(activation^2)` the Wave 3 calibration run
captured, the source decode is `keep.convert.fp4_expert`, and the checkpoint
reads reuse the Wave 3 runner's coalesced span index.

---

## 1. What the reference actually is

The campaign asks for error "vs the FP4 source" and "vs a bf16-dequantized
reference so the comparison is honest". Having built both, they are the same
number, and the reason is worth stating once:

**Every FP4 value in this release is exactly representable in bf16.** e2m1 has
one mantissa bit and a power-of-two E8M0 group scale; bf16 has eight mantissa
bits. `test_fp4_values_survive_a_bfloat16_cast` asserts bit-equality on all 256
codes. So "dequantise to bf16" and "dequantise to fp32" produce identical
weights, and there is no second reference hiding on this machine — the FP4
source *is* the ground truth available locally.

What cannot be measured here is DeepSeek's own bf16 → FP4 step, because the
pre-quantisation weights were never released. `estimate_source_fp4_step`
brackets it: re-encoding the released weight through the same FP4/group-32
grid round-trips at **0.0** relative error (it is already on the grid), and the
same grid applied to a Gaussian with the weight's own per-group RMS costs
**1.36e-2** relative MSE. So the honest framing of every number below:

> The VQ fit's error against the source is **~6.6x** the error DeepSeek's own
> 4-bit step plausibly spent. Nothing in this pilot is "lossless"; the VQ step
> dominates the total distance to the true weights by a wide margin.

## 2. Group-size policy, re-derived

Sweep: layers 1 and 20, 16 stratified experts each (route counts span three
orders of magnitude within a layer, so the sample is spaced by route-count
rank, not taken contiguously), all three projections, 8 fit iterations,
route-count weighted to the layer. Imatrix-weighted relative MSE:

| code_bits | group | bpw | gate L1 | gate L20 | down L1 | down L20 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 (E8-1bit) | 512 | 1.031 | 3.148e-1 | 3.104e-1 | 2.665e-1 | 2.690e-1 |
| 8 | 32 | 1.500 | 3.041e-1 | 2.996e-1 | 2.552e-1 | 2.583e-1 |
| **16 (E8P)** | **512** | **2.031** | **9.139e-2** | **9.002e-2** | **7.906e-2** | **7.920e-2** |
| 16 | 256 | 2.062 | 9.071e-2 | 8.936e-2 | 7.810e-2 | 7.875e-2 |
| 16 | 128 | 2.125 | 8.949e-2 | 8.815e-2 | 7.654e-2 | 7.755e-2 |
| 16 | 64 | 2.250 | 8.725e-2 | 8.587e-2 | 7.408e-2 | 7.519e-2 |
| 16 | 32 | 2.500 | 8.306e-2 | 8.165e-2 | 7.044e-2 | 7.139e-2 |
| 16 | 16 | 3.000 | 7.612e-2 | 7.492e-2 | 6.509e-2 | 6.594e-2 |

The decision rule is marginal return — fractional error reduction per extra
bpw — because that is the only way to compare a group-size step against any
other use of the same bits:

| step | Δbpw | gate L20 | down L20 |
| --- | ---: | ---: | ---: |
| **E8-1bit → E8P** (g=512) | 1.000 | **0.710** | **0.706** |
| g 512 → 256 | 0.031 | 0.235 | 0.181 |
| g 256 → 128 | 0.063 | 0.215 | 0.244 |
| g 128 → 64 | 0.125 | 0.208 | 0.243 |
| g 64 → 32 | 0.250 | 0.196 | 0.202 |
| g 32 → 16 | 0.500 | 0.165 | 0.153 |

**Recommendation: `code_bits: 16`, `group_size: 512` uniform on gate/up/down.**
Committed to `models/deepseek-v4-flash-0731.yaml`. The evidence:

1. **The codebook is where bits go.** The E8→E8P step returns 0.71/bpw, three
   times any group-size step. The inherited `default_code_bits: 8` was not a
   harmless placeholder — at 1.03 bpw it measures 3.1e-1 relative error and a
   **0.56** block-level cosine (§4), which is not a usable model. This is the
   one profile field that had to change.
2. **Uniform beats non-uniform at matched rate.** At exactly 2.50 bpw,
   `gate 64 / up 64 / down 16` sums to 23.76 (per-projection rel-MSE ×100)
   against uniform g=32's 23.47 — uniform wins by 1.2%.
3. **There is no per-projection signal to encode.** gate and up agree to three
   significant figures (same input space, same shape). down's first-step
   advantage flips sign between layers — 0.385/bpw at layer 1, 0.181/bpw at
   layer 20 — so it is sample noise, not structure. The campaign's "down_proj
   protected above base" intuition is not supported by the measurement.
4. **512 because the memory is worth more than the 9%.** Group size is a rate
   dial: within E8P the curve is smooth and mildly concave, so at a fixed bpw
   target the group size is *determined*, not chosen. Moving 512 → 32 buys a 9%
   error reduction for **17.4 GB** of a 128 GB machine (§5). On a deliverable
   whose entire premise is fitting resident with room for KV cache, that trade
   is bad. Bank the memory.

The full-layer fits validate the 16-expert sample: layer 20 gate at g=256
measured 8.936e-2 on 16 stratified experts and 8.960e-2 on all 256 — 0.3% apart.

### Levers tested and rejected

* **RHT / incoherence processing** (`mlx_vq.quality.rotation_search`). It
  *hurts* here. Rotating destroys the per-column imatrix (a Hadamard makes the
  diagonal uniform at its mean), and on `down` — whose importance spans 180x
  across input columns — that costs more than the incoherence gain: 9.18e-2
  rotated vs 7.97e-2 unrotated at g=512, **15% worse**. On gate it is a wash
  (9.21e-2 vs 9.09e-2). V4-Flash's experts are already fairly incoherent
  (peak/RMS 7.8 on gate) so there was little to win. Separately, the shipped
  `select_projection_rotation` is a pure-NumPy exhaustive search over the
  65,536-entry grid and would take hours per projection at this geometry — it
  cannot be used at V4 scale as written.
* **More fit iterations.** 3 → 8 buys 1.7% (gate) / 4.4% (down); 8 → 20 buys
  0.04%. The knee is at 8. The schedule below prices both.

## 3. What "2.5 and 3.0 bpw" means, reconciled

A rate-accounting discrepancy worth naming before anyone reads a ladder label
as a storage number. `mlx_vq.quality.dynamic_precision`'s tier menu labels E8P
as `effective_bits_per_weight: 3.0` and E8-1bit as `2.0` — those are *planner
budget* labels. Actual storage from `estimate_vq_storage` is
`code_bits/8 + 16/group_size`: **2.031** and **1.031** bpw. A 1 bpw gap.

The campaign's size table is in *storage* bpw, and reproduces exactly:
46 MoE blocks × 256 experts × 3 × 2048 × 4096 = 296.35 G routed weights at
3.0 bpw, plus 7.83 G residents at 8 bpw, gives **119.5 GB** against the
doc's "~120 GB" row. (Parameter counts verified against all 48 shard headers:
277.025 G routed in `layers.*` + 19.327 G in `mtp.*` + 7.828 G resident =
304.180 G, matching the profile's measured 304.18 B exactly.) Note this
confirms the envelope **includes the three DSpark drafter blocks** — 43 layers
alone does not reach the published figure.

## 4. Quality probe on two layers — and it is a proxy

Two probes, both on the real Wave 3 calibration statistics (40 sessions,
1.08 M supervised positions). Layer 1 has `compress_ratio` 0 (dense attention),
layer 20 has ratio 4 (sparse) — the two structural families in the config.

**Probe A, per-projection (linear).** The imatrix diagonal is the measured
per-column second moment of real routed activations, so under a diagonal
input-covariance model the imatrix-weighted relative error *is* the expected
relative output MSE, and the same weighting gives an expected cosine. This is
asserted against Monte Carlo in
`test_weighted_relative_mse_equals_the_expected_output_mse_ratio`. Full-layer
fits, all 256 experts:

| layer | ratio | gate rel / cos | up rel / cos | down rel / cos |
| ---: | ---: | --- | --- | --- |
| 1 | 0 | 9.047e-2 / 0.9537 | 9.045e-2 / 0.9537 | 7.658e-2 / 0.9609 |
| 20 | 4 | 8.960e-2 / 0.9542 | 8.955e-2 / 0.9542 | 7.577e-2 / 0.9614 |

**The attention compression ratio does not predict expert quantizability.** The
two layer families differ by 1.0% on gate and 1.1% on down — smaller than the
spread across experts within either layer (worst-expert rel-MSE 9.1e-2 vs
route-weighted 9.0e-2). The 43-layer sweep can treat layers as homogeneous,
which is exactly what the schedule assumes.

**Probe B, whole-expert (through the SwiGLU).** Probe A is linear and cannot see
what the nonlinearity does. `expert_block_proxy` pushes 512 synthetic hidden
states at the measured per-column activation RMS through
`down(silu(min(gate(x),10)) · clip(up(x),±10))` — the release's own clamped
SwiGLU, asserted equal to the adapter's implementation — for reference and
reconstruction:

| bpw | L1 block rel / cos | L20 block rel / cos |
| ---: | --- | --- |
| 1.031 (E8) | 6.891e-1 / **0.5576** | 6.705e-1 / **0.5771** |
| 2.031 (E8P g512) | 2.561e-1 / 0.8625 | 2.405e-1 / 0.8720 |
| 2.062 | 2.545e-1 / 0.8634 | 2.399e-1 / 0.8724 |
| 2.125 | 2.514e-1 / 0.8653 | 2.374e-1 / 0.8739 |
| 2.500 | 2.364e-1 / 0.8740 | 2.251e-1 / 0.8813 |
| 3.000 | 2.204e-1 / 0.8831 | 2.120e-1 / 0.8889 |

**The SwiGLU amplifies: block-level error is ~2.8x the per-projection error**
(0.240 vs 0.086 at layer 20, 2.031 bpw), because the gate and up errors compound through the product
and the clamp is not error-preserving. This is the number that should worry
people, not the 9%.

**Both probes are proxies and neither is teacher agreement.** Named limits:
the diagonal-Gaussian input has the right per-channel scale but not the right
shape (real hidden states are heavy-tailed with correlated outlier channels);
no router, shared expert, hyper-connection, or residual stream is modelled, and
the residual stream in particular will dilute a per-expert error substantially;
6-of-256 routing means each token sees six of these errors averaged, not one.
A block cosine of 0.87 is **not** a prediction of 0.87 teacher agreement — it is
a lower-bound-ish signal that the fit is not broken. The eval-teacher logits
(97 sessions, report/selection splits) exist and are the real gate; converting
these proxies into teacher agreement requires the Wave 4 kernels and a running
model, which is the next wave's job. Holdout untouched.

## 5. The deliverable — 43-layer sweep schedule

Measured, whole layers, all 256 experts × 3 projections, 8 iterations,
`code_bits 16`, no other process on the GPU:

| term | layer 20 (ratio 4) | layer 1 (ratio 0) | mean |
| --- | ---: | ---: | ---: |
| shard read (3.42 GB, one coalesced span) | 0.93 s | 0.54 s | 0.74 s |
| FP4 → fp32 decode (768 tensors) | 46.68 s | 40.37 s | 43.53 s |
| VQ fit (768 projections) | 899.07 s | 766.75 s | 832.91 s |
| **production cost / layer** | **946.68 s** | **807.66 s** | **877.17 s** |
| pilot-only error metrics | 94.32 s | 80.85 s | — |
| wall (incl. pilot-only work) | 1040.07 s | 887.98 s | — |
| **peak RSS** | **4.31 GB** | **4.31 GB** | **4.31 GB** |

Arithmetic: 877.17 s × 46 MoE blocks = 40,350 s = **11.21 h** (43 layers alone:
37,718 s = 10.48 h). Group size does not change this — the E8P search cost is
independent of it — so every row below is the same wall-clock:

| ladder | routed bpw | wall-clock | peak RSS | routed GB | + q8 residents | artifact GB | fits 98–105 GB? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| E8P g=512 **(recommended)** | 2.031 | 11.21 h | 4.3 GB | 75.25 | 8.32 | **83.56** | **yes**, 14 GB spare |
| E8P g=256 | 2.062 | 11.21 h | 4.3 GB | 76.40 | 8.32 | 84.72 | yes |
| E8P g=128 | 2.125 | 11.21 h | 4.3 GB | 78.72 | 8.32 | 87.04 | yes |
| E8P g=32 | 2.500 | 11.21 h | 4.3 GB | 92.61 | 8.32 | 100.93 | yes, tight |
| E8P g=16 | 3.000 | 11.21 h | 4.3 GB | 111.13 | 8.32 | 119.45 | **no**, +14 GB over |
| affine-q3 g=128 | 3.250 | — | — | 120.39 | 8.32 | 128.71 | **no**, +24 GB over |

Routed bytes = 296.35e9 × bpw / 8. Residents = 7.828e9 × 8.5 / 8 = 8.32 GB
(q8 plus group scales); at q6/q4 (`resident_bpw` 6.5) every artifact figure
drops 1.96 GB and the verdicts do not change.

**Cheaper variants, both measured:**

* `iterations=3` (the shipped default) costs 4 assign passes instead of 9:
  832.91 × 4/9 = 370.2 s of fit, **5.30 h total**, for +1.7% (gate) / +4.4%
  (down) error. Reasonable if the sweep needs re-running.
* **The fit is 65% NumPy-on-CPU.** Profiled at g=32, 8 iterations, one
  [2048, 4096] projection: Metal E8P search **35.1%**, codebook gather 28.7%,
  float64 scale-update sums 30.6%, normalise/broadcast 5.5%. Porting the gather
  and the scale update to MLX — squarely the standing MLX/Metal
  standardization directive — would cut the sweep to ~5.7 h at 8 iterations
  (Amdahl at 5x on the 65%), or ~2.9 h combined with `iterations=3`. This is a
  known win and it is *not* deferred silently: it is the single highest-leverage
  change to Wave 5's cost and it is called out as a task below.

Assumptions the estimate carries, each named in the artifact's `assumptions`
field: per-layer cost is the mean of two measured layers assumed uniform (the
two differ by 15%, plausibly thermal — layer 20 ran hot immediately after the
sweep — so treat 11.2 h as ±1 h); the three MTP drafter blocks are counted as
full MoE layers at the same cost and rate; residents are 7.828 G weights at the
stated bpw; no thermal derating, no batching speed-up, single process. Peak RSS
is 4.3 GB against 128 GB, so several layer workers could run in parallel — the
fit is CPU-bound, so this is real headroom, unmeasured.

## 6. E8P vs affine — the crossover, and what it means for the 3.0 ladder

The campaign names a same-machine affine control but never priced it against
VQ at matched rate. Measured on the same experts, same objective, route-weighted
(affine stores fp16 scale + fp16 bias per group, so bpw = bits + 32/group):

| scheme | bpw | L20 gate rel | L20 down rel |
| --- | ---: | ---: | ---: |
| E8P g=512 | 2.031 | 9.002e-2 | 7.920e-2 |
| E8P g=32 | 2.500 | 8.165e-2 | 7.139e-2 |
| affine-q2 g=32 | 3.000 | 1.397e-1 | 1.460e-1 |
| **E8P g=16** | **3.000** | **7.492e-2** | **6.594e-2** |
| **affine-q3 g=128** | **3.250** | **5.455e-2** | **5.769e-2** |
| affine-q3 g=64 | 3.500 | 4.359e-2 | 4.664e-2 |
| affine-q3 g=32 | 4.000 | 3.292e-2 | 3.460e-2 |
| affine-q4 g=64 | 4.500 | 9.261e-3 | 9.744e-3 |
| _(DeepSeek's own FP4 step, estimated)_ | 4.0 | _1.362e-2_ | _1.362e-2_ |

**The crossover is at ~3.1 bpw.** Below it E8P dominates decisively — at
3.0 bpw E8P is **1.87x better** than affine-q2, and at 2.031 bpw E8P already
beats affine-q2 at 3.0 bpw by 1.55x. Above it affine-q3 wins by 27% at a 0.25 bpw
premium. (affine-q4 at 4.5 bpw finally undercuts DeepSeek's own 4-bit step,
which is where "better than the source" would start — 130 GB, off this machine.)

**This settles the codebook question for the local artifact, and it kills the
3.0 ladder.** The envelope caps the routed rate: 105 GB minus 8.32 GB of q8
residents leaves 96.68 GB, i.e. **2.61 bpw** maximum (2.66 with q6 residents).
Every reachable local rate is below the crossover, so:

> **Recommendation: run the local sweep at E8P only, and drop the 3.0 bpw
> ladder from the local deliverable.** It does not fit (119.5 GB vs a 105 GB
> ceiling), and at the one rate where more bits would genuinely pay, the right
> answer is a different quantizer family that Wave 4's kernels do not cover.
> If a 3.0-class artifact is wanted as an off-Mac reference, build it as
> affine-q3 g=128 at 3.25 bpw / 128.7 GB — 27% better than E8P g=16 for 9 GB
> more on gate (13% on down) — and name it a reference, not a deliverable.

The tradeoff, quantified: dropping the 3.0 ladder gives up 7.5e-2 → nothing
(it was never runnable locally) and saves one 11.2 h sweep. Choosing E8P g=512
over g=32 for the surviving ladder gives up 9.3% of per-projection error and 6.4%
of block error, and banks 17.4 GB — on a 128 GB machine that is the difference
between a tight fit and one with room for KV cache and activations.

## 7. Concerns

1. **The proxy cannot clear the quality gate and nobody should treat it as if
   it can.** A 0.87 block cosine at 2.03 bpw is the honest headline of §4 and
   its relationship to teacher NLL/KLD/top-1 is unknown. The teacher logits are
   already on disk; the conversion needs Wave 4's kernels and a running model.
   **The 43-layer sweep should not be treated as gated by this pilot** — it is
   cheap enough (11.2 h, $0) to run speculatively, but the artifact it produces
   must face the real eval before anything is called an RC.
2. **2-bit is at its rate-distortion limit, not at an implementation bug.** A
   2-bit E8 lattice on a Gaussian source is expected around 1.0e-1 relative MSE;
   we measure 8.9e-2. The fit is working. QuIP#-class improvements come from
   two-sided incoherence plus fine-tuning, and the one-sided RHT that KEEP ships
   measured *negative* here (§2). If 2 bpw quality is insufficient, the answer
   is more bits or a recovery/fine-tune pass, not a better search.
3. **Group sizes below 512 are unvalidated at the kernel.** `gather_vqmm`
   accepts any 8-aligned group size that divides the input dim, but Wave 4
   benchmarked 512 only, and g=32 multiplies the runtime scale traffic 16x. The
   recommendation avoids this by staying at 512; if the rate target ever moves,
   the kernel throughput must be re-measured before the profile follows.
4. **`select_projection_rotation` is NumPy-only** and unusable at V4 scale
   (exhaustive search over 65,536 codewords × 1 M vectors per projection). It is
   dead weight for this model unless ported; flagged rather than fixed because
   §2 says the lever is not worth having here anyway.
5. **65% of the sweep is CPU NumPy** (§5). Standing directive says this belongs
   on Metal. Not deferred quietly — it is the top follow-up.
6. **The 15% layer-to-layer timing spread is unexplained.** Most likely thermal
   (layer 20 ran immediately after a 25-minute sweep), but it could be
   data-dependent allocator behaviour. Two layers is not enough to tell, and it
   is why the schedule is quoted as 11.2 h ± 1 h rather than to the minute.
7. **MTP drafter blocks are assumed to cost and quantize like ordinary layers.**
   They were never measured — they are three full MoE blocks at `mtp.{0,1,2}`
   with their own 256 routed experts, and the schedule counts them at the layer
   mean. Cheap to check with one `layer-fit` run once the adapter exposes them.
