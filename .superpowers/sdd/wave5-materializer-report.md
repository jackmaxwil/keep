# Wave 5 materializer — the naming contract is now proven, and the sweep is 8x cheaper than the pilot priced it

**Status: built, verified, launched. The production materializer behind
`deepseek_v4_vq_groups` exists, one real layer has been materialized *and* bound
through the adapter's own path, and the full 46-block sweep is running
unattended (PID 21643, ETA ~1.4 h, not 11.2 h).**

**Three findings that change what the next wave should expect. First, the
`w1 -> gate_proj / w3 -> up_proj / w2 -> down_proj` mapping is no longer
"semantically unconfirmed": bound-module forward output matches the fit's own
reconstruction at 3.46e-14 relative MSE, and the gate/up-swapped control lands
4.3e10x worse — but its cosine is still 0.9993, which is exactly why "it loaded
and looked fine" was never evidence. Second, the MLX port of the fit is 2.20x
with codes *and* scales byte-identical to the NumPy reference on real weights;
the fully-GPU variant is 3.14x and is not bit-reproducible, so it is not the
default. Third, the pilot's 11.2 h schedule does not reproduce on this machine
today — the same NumPy fit measures 0.205 s/projection now against the pilot's
1.171 s, a 5.7x gap that appears identically in the decode stage and therefore
is machine state, not code. The sweep is one lunch break, not one overnight.**

## Commits (branch `main`)

| SHA | Subject |
| --- | --- |
| `8f263b0a` | `feat(dsv4): Wave 5 production VQ materializer` |
| `4914e098` | `docs(sdd): Wave 5 materializer report -- the naming contract is proven` |
| _this_ | `fix(dsv4): count routed weights off the codes each file holds` |

`tests/test_dsv4_materializer.py` + `tests/test_deepseek_v4_flash_adapter.py`
**105 passed**. Also green after the shared-module changes:
`tests/test_dsv4_teacher_runner.py` 49, `tests/test_dsv4_vq_pilot.py` +
`tests/test_model_profiles.py` 73. Ruff clean on all new and touched files.

Files:

* `src/mlx_vq/convert/dsv4_vq_materialize.py` — the materializer: block plan,
  per-block/per-expert/per-projection fit, atomic publication, SHA-256-verified
  resume, family-template manifest + audit, and the bind roundtrip.
* `src/mlx_vq/convert/dsv4_vq_fit_mlx.py` — the MLX port of the fit hot paths,
  with the parity contract stated stage by stage.
* `benchmarks/materialize_dsv4_vq.py` — CLI: `imatrix` / `plan` / `parity` /
  `block` / `verify-roundtrip` / `sweep` / `manifest` / `monitor`.
* `tests/test_dsv4_materializer.py` — 49 headless tests over a synthetic FP4
  block (no checkpoint, no calibration run).
* `src/mlx_vq/quality/dsv4_teacher_runner.py` — the coalesced span index
  generalized from "backbone layer" to "any MoE block", so `mtp.{0,1,2}` are
  reachable. `build_dsv4_expert_span_index(layers=...)` keeps its int-keyed
  contract and is now a wrapper.
* `src/mlx_vq/quant/e8p_metal.py` — `encode_e8p_diagonal_hessian_fused_mx`, the
  device-resident twin of the fused E8P search.

Evidence committed to `artifacts/quality/dsv4-vq-materializer-20260814/` (32 KB,
force-added per the directory's convention): `roundtrip-layer0.json`,
`fit-parity-layer20.json`, `block-record-layer0.json`,
`imatrix-cache-provenance.json`.

Run directory: `~/keep-artifacts/dsv4-vq-e8p-g512/` — `manifest.json`,
`status.json` / `status.log`, `records/*.json`, `roundtrip.json`,
`fit-parity.json`, `imatrix-cache.npz` (+ `.json` provenance).

Reproduce:

```
uv run --group dev python benchmarks/materialize_dsv4_vq.py imatrix
uv run --group dev python benchmarks/materialize_dsv4_vq.py parity \
  --out ~/keep-artifacts/dsv4-vq-e8p-g512/fit-parity.json
uv run --group dev python benchmarks/materialize_dsv4_vq.py block --block layers.0
uv run --group dev python benchmarks/materialize_dsv4_vq.py verify-roundtrip --block layers.0
uv run --group dev python benchmarks/materialize_dsv4_vq.py sweep
uv run --group dev python benchmarks/materialize_dsv4_vq.py monitor
```

Everything composes the pilot: the fit is
`quantize_weight_importance_aware` (or its MLX twin), the objective is the Wave 3
imatrix diagonal, the source decode is `keep.convert.fp4_expert`, the reads are
the Wave 3 runner's span index, and the error metrics are
`dsv4_vq_pilot.projection_error`. The pilot module was imported, not copied.

---

## 1. The naming contract, closed

Two reviews flagged `w1 -> gate` / `w3 -> up` / `w2 -> down` as asserted but not
semantically confirmed. It is now confirmed, on the real layer 0 of the real
checkpoint, through the adapter's own bind path
(`bind_deepseek_v4_flash_vq_experts` → `load_deepseek_v4_flash_vq_switch_glu` →
`QuantizedVQSwitchGLU`), with the artifact this materializer wrote:

| claim | checked by | result |
| --- | --- | --- |
| nothing unbound | `has_unbound_deepseek_v4_flash_vq_experts` | `false` |
| nothing dense | `dense_deepseek_v4_flash_routed_parameter_names` | `()` |
| forward == fit reconstruction | 4 tokens × 6 routes, fp64 reference | rel-MSE **3.46e-14**, cos 1.0 |
| gate/up swapped is *not* it | same reference, gate and up exchanged | rel-MSE **1.48e-3** |

Discrimination ratio **4.27e10**. Evidence:
`~/keep-artifacts/dsv4-vq-e8p-g512/roundtrip.json`.

**The control is the whole point, and its number is the thing worth
remembering.** A swapped gate/up artifact loads, binds, produces finite output,
and has a **0.9993 cosine** against the correct answer. No smoke test, no "does
it run", and no eyeball on a cosine would have caught it. Only a comparison
against both hypotheses does — which is why the check compares against both and
why `passed` requires the correct mapping to be 100x closer, not merely close.

Why the correct mapping lands at 1e-14 rather than at a tolerance: the reference
is the *fit's own* reconstruction, decoded back out of the published safetensors
(so the file, the codes, the scales and the codebook all participate), and the
release's FP4 source is exactly representable, so the only gap between the
kernel and the fp64 reference is the kernel's own accumulation.

`verify_bind_roundtrip` is a library function, not a script body: the same code
runs in `tests/test_dsv4_materializer.py` against a 64-wide toy artifact (fast,
in CI, no checkpoint) and against real layer 0 from the CLI. The test asserts the
swap discrimination too, so a future refactor that quietly stopped
discriminating fails a test rather than passing a vacuous one.

## 2. The MLX port — 2.20x, byte-identical, measured on real weights

The pilot's profile said 65% of the fit was CPU NumPy: search 35.1% (Metal),
gather 28.7%, float64 scale sums 30.6%, normalise/broadcast 5.5%. Each stage was
ported and each carries a named parity contract, because "faster" is worthless
here if the artifact's bytes move.

| stage | ported to | parity contract | evidence |
| --- | --- | --- | --- |
| normalise (`grouped / scales`) | `mx.divide` | **byte-exact** — IEEE fp32 divide | `test_mlx_normalise_is_byte_exact` |
| search (weighted nearest E8P) | new device-resident kernel entry | **same kernel, same bytes** — what was removed is the per-iteration host round-trip, not the arithmetic | full-fit test below |
| gather (`table[codes]`) | `mx.take` | **byte-exact** — data movement, no arithmetic | `test_mlx_gather_is_byte_exact` |
| products (`h*w*decoded`) | `mx.multiply`, reference's own order | **byte-exact** — IEEE fp32 multiply | `test_mlx_products_are_byte_exact` |
| reduce (float64 scale sums) | **kept in NumPy** | see below | — |

**The reduction is the one that cannot go to Metal, and it was measured rather
than assumed.** Metal has no float64. An `mx.sum` fp32 tree reduction of the
512-element groups lands within ~3e-8 relative of NumPy's float64 pairwise sum —
but ~30% of the per-group scales then differ in their final float32 bit, those
scales feed the next iteration's assignment, and a differing bit flips an argmin
on a near-tie codeword. That changes the artifact's bytes. So the default
backend (`mlx-exact`) computes the fp32 product tensors on the GPU and brings
them back for the identical `np.sum(..., dtype=np.float64)`. The transfer is
~67 MB per iteration on unified memory and costs nothing measurable next to the
gather it replaces.

Measured on real layer-20 weights, 3 experts × 3 projections, g=512,
8 iterations (`fit-parity.json`):

| backend | s/projection | speed-up | codes byte-identical | scales byte-identical | max code disagreement | max rel-MSE delta |
| --- | ---: | ---: | :---: | :---: | ---: | ---: |
| `numpy` (reference) | 0.2052 | 1.00x | — | — | — | — |
| **`mlx-exact` (default)** | **0.0932** | **2.20x** | **yes, 9/9** | **yes, 9/9** | **0** | **0** |
| `mlx-fp32` | 0.0652 | 3.14x | no | no | 1.9e-6 | 6.8e-9 |

So the standing MLX/Metal directive is satisfied without giving up
bit-reproducibility, and the extra 43% that the inexact mode would buy is
available, priced, and *not* the default. `mlx-fp32`'s quality cost is nil
(6.8e-9 absolute rel-MSE delta — the flips are near-ties, as expected); the
reason to refuse it is reproducibility of the artifact, not quality.

`resolve_fit_backend` also refuses to downgrade silently: on a host without
Metal, or for `code_bits=8` (not ported — the ship ladder is E8P), it returns
`"numpy"`, and the *resolved* backend is what the per-block record and the
manifest carry.

## 3. The measured schedule, and why it disagrees with the pilot

Measured, real blocks, all 256 experts × 3 projections, g=512, 8 iterations,
`mlx-exact`:

| term | pilot (layer 20 / layer 1) | production (layers.0, idle machine) |
| --- | ---: | ---: |
| shard read (3.42 GB, one coalesced span) | 0.93 / 0.54 s | **0.29 s** |
| FP4 → fp32 decode (768 tensors) | 46.68 / 40.37 s | **9.28 s** |
| VQ fit (768 projections) | 899.07 / 766.75 s | **74.30 s** |
| artifact write (1.64 GB, 3 files) | — | 0.12 s |
| **per block** | **946.7 / 807.7 s** | **99.9 s** |
| peak RSS | 4.31 GB | 5.51 GB |

46 blocks × ~100–130 s = **1.3–1.7 h**, against the pilot's 11.21 h. The live
sweep's own `status.json` carries the authoritative number (`eta_h`, with
`eta_basis` naming what it is the mean of) and the first restarted blocks are
running at 115–130 s while pytest shared the GPU.

**The 8.5x is only partly the port, and the rest is not explained by anything in
the code.** The MLX port accounts for 2.20x of it. The remaining ~3.9x is that
the *same NumPy code* on the *same machine* now measures 0.205 s/projection
where the pilot measured 1.171 s (899.07/768), and the decode stage shows the
same ratio independently (9.3 s vs 43.5 s, 4.7x). Two unrelated stages moving by
the same factor points at machine state, not at an optimisation: the pilot's own
concern #6 already flagged a 15% layer-to-layer spread it suspected was thermal,
and this looks like the same effect an order of magnitude larger. Consequences,
stated plainly:

* The pilot's absolute wall-clock table should not be quoted as this machine's
  cost. Its *ratios* (iterations, group size, rate ladder) still hold — those
  were measured within one session.
* If the machine re-enters whatever state the pilot ran in, the sweep's ETA will
  move by up to ~6x, and the status log's per-block line is what will show it as
  a trend rather than a surprise.
* Peak RSS is 5.51 GB against 128 GB. The fit is now GPU-bound rather than
  CPU-bound, so the pilot's "several layer workers in parallel" headroom is no
  longer obviously real; it was never measured and still is not.

## 4. Artifact shape, and the DSpark/MTP caveat answered

Per block: three files, 545,261,632 bytes each = 1.636 GB. 46 blocks =
**75.25 GB** of routed payload, which reproduces the pilot's 75.25 GB projection
exactly. The 83.6 GB headline figure adds 8.32 GB of q8 residents, which this
materializer does not write — residents are bound from source by
`bind_deepseek_v4_flash_non_vq_weights`.

| tensor | dtype | shape (gate/up) | shape (down) |
| --- | --- | --- | --- |
| `….{projection}.codes` | uint16 | [256, 2048, 512] | [256, 4096, 256] |
| `….{projection}.scales` | float16 | [256, 2048, 8] | [256, 4096, 4] |
| `model.vq_codebook.e8` | uint32 | packed E8P grid (`E8P_PACKED_ABS_SHA256`) | same |

Cross-validation against the pilot's independent 16-expert sweep at the same
rate point — different sampling, different code path, same policy:

| layer | projection | pilot sweep (g=512) | production (all 256 experts) | delta |
| ---: | --- | ---: | ---: | ---: |
| 1 | gate | 9.139e-2 | 9.115e-2 | −0.3% |
| 1 | down | 7.906e-2 | 7.909e-2 | +0.04% |

**The MTP caveat, answered.** The pilot never measured a drafter block. Measured
here: `mtp.0` reads the same 3.42 GB span from shard 46, carries the same
256 experts × [2048, 4096] geometry, and fits at the same per-projection cost, so
the schedule's assumption that the three drafter blocks cost like ordinary layers
is now a measurement rather than an assumption. The sweep measures the first real
one and `status.json` reports `mtp_seconds_per_block` separately, with
`eta_basis` saying in words whether MTP has been measured yet.

**What the drafter blocks do *not* have is calibration.** The Wave 3 run captured
`num_hidden_layers=43` and never saw `mtp.*`, so there is no measured imatrix for
a drafter expert. The policy is explicit and recorded per block in
`importance_sources`:

* `backbone-mean` (default) — layer 42's route-count-weighted mean *column
  shape*, normalised to unit mean, shared across drafter experts. The rationale
  is that the drafter operates on the same residual-stream hidden space, so the
  per-column structure is the part that plausibly transfers; the per-expert
  routing structure explicitly does not, and is not pretended to.
* `uniform` — flat ones, i.e. plain unweighted least squares.

Neither is measured on the drafter. The manifest says so, and the fit error it
produces on `mtp.0` (8.81e-2 gate / 8.89e-2 down) sits in the same band as the
backbone, which is reassuring and is not the same thing as being right.

## 5. Manifest, audit, and resume

`manifest.json` carries the family template's required fields: schema version 1,
family `deepseek-v4-flash`, converter `deepseek_v4_vq_groups`, source identity
with revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` plus `config.json` and
index SHA-256, `seed_artifact: null` / `seed_artifact_mutated: false`, geometry
(46 blocks / 43 backbone / 3 mtp / 256 experts / 3 projections / shapes), the
policy block, codebook name and hash, per-block file inventory with SHA-256 and
shapes, per-block route-weighted fit metrics, per-block timing, the
`tensor_mapping` block (both directions, both filename conventions, and
`mtp_binder_exists: false`), calibration provenance (40 sessions, cache hash
`2165aa4e…`), and evidence paths.

The audit reports the template's routed-MoE fields: projection counts expected
and present, missing files, rewritten vs symlinked (46×3 vs 0), high-precision
routed projections (0), sidecar counts (0), code-bit and group-size
distributions, effective routed bpw (2.03125), NAX/Metal group-size
compatibility, and fallback layers (none).

**`dense_routed_experts` and `unbound_vq_experts` are deliberately `null` in the
file audit.** A tree of files cannot prove that nothing fell back to a dense
expert; only a bound model can. `audit_artifact_tree` therefore refuses to
answer, and `build_manifest` fills both fields from the roundtrip's `bind_proof`
— the adapter's *own* `has_unbound_…` and `dense_…_parameter_names` results on a
loaded layer, quoted with the function names that produced them.
`test_audit_cannot_claim_dense_or_unbound_without_a_bind_proof` locks that
refusal in.

Resume is per block and does not trust its own bookkeeping: a block counts as
complete only when its record parses, names this exact rate, covers the full
256-expert set, and every file it claims is still on disk at the recorded size
*and* the recorded SHA-256. Five tests cover the ways that can fail, including
the one that would otherwise bite — a `block --experts 0 1` smoke run leaving a
two-expert file for the sweep to inherit.

## 6. Sweep launch evidence

```
$ python benchmarks/materialize_dsv4_vq.py monitor
phase=running block=layers.3 3/46 blocks  elapsed=61.4s  eta=…  peak=5.52 GB
eta basis: mean of N measured block(s) at …s; MTP measured at …s/block
```

* **PID 21643** (`benchmarks/materialize_dsv4_vq.py sweep`), nohup, holding
  `.keep-heavy-job.lock`.
* Output `~/keep-artifacts/dsv4-vq-e8p-g512/`; console log `nohup.out`;
  `status.json` (atomic, polled by `monitor`) and `status.log` (append-only, one
  line per completed block with its timing and fit metrics).
* Blocks `layers.{0,1,2}` were already complete and were skipped after SHA-256
  verification; the run resumes at `layers.3` and ends at `mtp.2`.
* ETA **~1.4 h** from the measured per-block mean; the status file recomputes it
  every block and states its basis in words, including whether MTP has been
  measured yet.
* `manifest.json` is rewritten after every block, so the tree is auditable
  mid-flight and a crash leaves a manifest describing exactly what completed.

## 6a. The layer-40 crash: one cold expert in 11,008

The first sweep died at **block 41 of 46** — `layers.40`, `gate_proj`, expert 170
— with `ValueError: weighted reference energy is zero` out of
`projection_error`. 40 blocks were already verified on disk and none were lost.

**Root cause is a materializer bug, not a data problem, and not the
late-layer indexing hazard it looked like.** Exactly one pair in the 43 × 256
grid has no calibration evidence at all:

| | |
| --- | --- |
| pairs with zero measured importance (`hidden`) | **1 of 11,008** — `layers.40` expert 170 |
| pairs with zero measured importance (`down`) | **1** — the same pair |
| pairs with `route_count == 0` | **1** — the same pair |
| that expert's weights | 87% nonzero, RMS 0.025 — indistinguishable from any other |

The corpus simply never routed a token to it in 40 sessions. Nothing is wrong
with the expert, the layer, or the indexing — layers 41 and 42 are clean, and so
are the other 255 experts of layer 40.

What actually broke: `quantize_weight_importance_aware` *already* coerces an
all-zero importance vector to ones (`_validate_importance`), so the fit ran
correctly and produced valid codes. `materialize_block` then scored those codes
against the **raw zero vector**, and `projection_error` correctly refused —
a zero weighted reference energy is meaningless, and it is right to raise rather
than return a NaN. The fit and the metrics were looking at different objectives,
and only the metrics noticed.

**The fix makes the substitution explicit rather than moving the guard.**
`block_importance` now detects a non-positive measured row, returns *ones* as
the effective vector, and labels it `uniform_zero_importance_fallback`. The fit
and the metrics therefore see the same objective by construction. `BlockRecord`
records `importance_fallback_experts` — the expert indices, not just a count,
because "which experts have nothing behind them" is a question an audit asks —
and the manifest audit aggregates them per block as
`importance_fallback_experts` / `importance_fallback_expert_count`.

Uniform importance is the right objective here, not a patch: with no evidence
about which input columns matter, equal weights is the honest prior. Skipping the
expert is not an option — the binder requires all 256. Verified on the real
failing case:

| projection | source | rel-MSE | cosine |
| --- | --- | ---: | ---: |
| gate_proj | `uniform_zero_importance_fallback` | 9.385e-2 | 0.9519 |
| up_proj | `uniform_zero_importance_fallback` | 9.394e-2 | 0.9519 |
| down_proj | `uniform_zero_importance_fallback` | 9.626e-2 | 0.9507 |

In line with the rest of the model (~9e-2), and expert 169 still reports
`measured` with all 4,096 columns positive — the fallback is per expert, not per
layer. Two tests cover it:
`test_an_unrouted_expert_gets_uniform_importance_and_is_named` and
`test_a_block_with_an_unrouted_expert_materializes_and_reports_it` (the
regression, asserting the block materializes, the expert is named, and the
manifest reports it).

Note for the eval wave: because the expert's route count is zero, it contributes
zero weight to every route-weighted layer metric, so its 9.4e-2 does not move
any headline number. It will also, by the same token, almost never be selected at
inference — but "almost never" on the calibration corpus is not "never" on the
holdout, so it had to be fitted properly rather than filled with zeros.

## 7. Concerns

1. **The pilot's schedule does not reproduce and nobody knows why (§3).** Two
   independent stages are ~5x faster today than in the pilot, on the same code
   and the same machine. Whatever caused it can come back mid-sweep. The status
   log's per-block trend is the instrument; there is no fix here, only
   visibility.
2. **The quality gate is still untouched.** These artifacts are fit to a proxy.
   The route-weighted per-projection error is ~9e-2 and the pilot's block-level
   proxy cosine was 0.87 at this rate; neither is teacher agreement. The
   eval-teacher logits are on disk and converting the proxy into agreement needs
   Wave 4's kernels and a running model. **Nothing in this wave licenses calling
   the artifact an RC.**
3. **MTP importance is a policy, not a measurement (§4).** The drafter blocks are
   being fit against layer 42's column shape. If the drafter matters to the
   deliverable, the honest fix is a calibration pass that actually captures
   `mtp.*` activations, and until then the three drafter blocks carry a weaker
   provenance than the 43 backbone ones — recorded per block, not hidden.
4. **There is no MTP binder, so the drafter artifacts are unverified by
   roundtrip.** They are written to the convention `mtp_drafter.blocks.{S}.…`
   that the adapter's own parameter tree implies, and `verify-roundtrip` refuses
   drafter blocks rather than pretending. The naming contract for the drafter is
   therefore in exactly the state the backbone's was before today: plausible and
   unconfirmed. It needs the binder Wave 4/6 owes.
5. **Durability of the write is weaker than the atomicity.** Publication is
   partial-file + `os.replace` + `fsync`, which is atomic against a crash, but
   1.64 GB landing in 0.12 s says the bytes are in the page cache and macOS's
   real barrier is `F_FULLFSYNC`, which is not used (it would cost real time
   across 138 files). Integrity is recovered on resume by re-hashing from disk,
   so a lost write is *detected*; it is not *prevented* by a power cut.
6. **One expert in the artifact has no calibration evidence (§6a).** `layers.40`
   expert 170 is fitted against uniform importance. It is named in the block
   record and the manifest audit, so it can never be mistaken for a measured
   fit, but if the eval ever routes to it the quality there is unmeasured. The
   general lesson is worth keeping: a 40-session corpus does not cover 11,008
   (layer, expert) pairs, and any future per-expert lever needs the same
   explicit no-evidence branch.
7. **`code_bits=8` has no MLX port and the fit falls back to NumPy.** Deliberate
   — the ship ladder is E8P and the pilot showed E8-1bit is not a usable model —
   but a future experiment at 8 bits will silently run 2.2x slower, which
   `resolve_fit_backend`'s return value records rather than hides.
8. **The full-tree audit has not run yet** because the sweep has not finished.
   `manifest.json` is rewritten per block, so the audit is correct for what
   exists. Three things to do when the sweep lands: re-run `manifest` (the
   sweep process is holding the pre-fix `routed_weight_count`, which was
   under-counting 3x -- fixed in this commit, and the running process cannot
   pick it up), `verify-roundtrip` on a second layer, and a `du` against the
   75.25 GB projection.
9. **The sweep is sharing the GPU with whatever else runs.** The per-block time
   moved from 100 s idle to 144 s while this session ran pytest against the same
   Metal device. That is visible in `status.log` as a rate trend and is the
   reason `eta_h` drifted from 1.23 h to 1.67 h; it is contention, not
   regression. Nothing should be scheduled on the GPU until the sweep lands.

---

# FINAL — the sweep landed, and the artifact is 20 GB *under* the envelope, not against it

**Status: 46/46 blocks complete and SHA-256-verified, manifest rebuilt off the
post-fix code, full-tree audit clean, and the naming contract now proven on a
late backbone layer and on a drafter block as well as on layer 0. The routed
payload reproduces the pilot's 75.25 GB projection to the byte. The full runnable
artifact, priced off the real shard headers rather than an assumed resident
count, is 82.5–85.1 GB — 13 to 22 GB below the 98–105 GB envelope's floor.**

**Two things worth carrying forward. First, the cold expert stayed a population
of one: `layers.40` expert 170 is the only unrouted (layer, expert) pair in the
whole model, and this is now settled two independent ways rather than asserted —
the aggregated calibration cache has exactly 1 of 11,008 pairs with zero measured
importance and zero route count, and the 46 block records account for all 35,328
projections as 33,021 `measured` + 2,304 `backbone_mean_layer_42` + 3
`uniform_zero_importance_fallback` (that one expert's gate/up/down). Second, the
envelope's slack is real headroom and not an invitation to spend it: the pilot
already measured that buying the 2.500 bpw point costs 17 GB for 9% of
per-projection error, and nothing in the finished artifact changes that
arithmetic — but the decision is now being made against a measured 84 GB, not a
projected 83.6 GB.**

## F1. Manifest verdict

`manifest` re-run at HEAD (`2591325b`) against the finished tree. It reads the 46
block records, re-verifies every one (record parses, names this exact rate,
covers all 256 experts, every claimed file present at the recorded size *and*
recorded SHA-256), and rewrites `manifest.json`. All 46 verified; none rejected.

| audit field | value |
| --- | --- |
| `total_routed_projection_files_expected` / `_present` | 138 / 138 |
| `missing_projection_files` | `[]` |
| `rewritten_projection_count` / `symlinked_projection_count` | 138 / 0 |
| `high_precision_routed_projection_count` | 0 |
| `sparse_residual_sidecar_count` / `continuous_sidecar_count` | 0 / 0 |
| `code_bits_distribution` / `group_size_distribution` | `{16: 138}` / `{512: 138}` |
| `effective_routed_bpw` | 2.03125 |
| `nax_metal_compatible_group_size` | `true` |
| **`routed_weight_count`** | **296,352,743,424** |
| **`importance_fallback_expert_count`** | **1** |
| **`importance_fallback_experts`** | **`{"layers.40": [170]}`** |
| `fallback_layers` | `[]` |
| `artifact_payload_bytes` | 75,246,105,792 |

**The two fields §7 item 8 said were stale are now right.** `routed_weight_count`
is 296,352,743,424, which is exactly 46 blocks × 256 experts × 3 projections ×
8,388,608 weights and matches the checkpoint's own routed logical count computed
independently from the shard headers (F3). The pre-fix sweep process was holding
a value 3x low; it cannot pick up a fix mid-run, so this re-run was the only way
to get it, and it was worth doing rather than annotating.

`importance_fallback_experts` / `_count` now exist in the manifest and carry the
single cold expert. One honest caveat, stated because the manifest cannot state
it itself: **the 40 block records written before commit `3b39bd0f` do not carry
the `importance_fallback_experts` field at all** — the field did not exist when
they were written, and re-deriving it would mean re-fitting 40 blocks. The
aggregate is nonetheless sound, for two independent reasons:

1. Those 40 records *do* carry `importance_sources`, and every one of them reads
   `{"measured": 768}` — all 768 projections of each block fit against a measured
   importance vector, so none of them had a fallback to report.
2. The pre-fix code could not have completed a block containing a zero-importance
   expert: `projection_error` raised `ValueError: weighted reference energy is
   zero`, which is precisely how `layers.40` was found. Forty blocks finishing
   under that code is itself proof that none of them contained one.

The manifest's `bind_proof` was also refreshed so it names its own route:
`bind_path: "binder"`, `file_discovery_exercised: true`, `gap: null` — the
strongest of the three proofs, and now labelled as such in the file rather than
implied.

## F2. Audit — clean, and the two fields a file tree cannot answer are answered

| claim | value | who answered it |
| --- | --- | --- |
| `dense_routed_experts` | **`false`** | `dense_deepseek_v4_flash_routed_parameter_names` → `[]`, on a bound layer |
| `unbound_vq_experts` | **`false`** | `has_unbound_deepseek_v4_flash_vq_experts` → `false`, on a bound layer |
| total importance-fallback experts, all 46 blocks | **1** | `layers.40` expert 170, and nothing else |

Both booleans come from the adapter's own predicates on a loaded model via
`build_manifest`'s `bind_proof` wiring, not from the file audit —
`audit_artifact_tree` still refuses to answer them from a tree of files, and
`test_audit_cannot_claim_dense_or_unbound_without_a_bind_proof` still locks that
refusal in.

**Nothing new appeared in the final six blocks.** The concern worth checking was
that `layers.40` was found at block 41 of 46, so the last stretch was the least
observed. It is clean: `layers.41`, `layers.42` and `mtp.{0,1,2}` all report
`importance_sources` with zero fallback entries, and the direct census of the
aggregated calibration cache finds exactly one zero pair in the full 43 × 256
grid:

```
importance__hidden  zero-importance (layer, expert) pairs: [(40, 170)]
importance__down    zero-importance (layer, expert) pairs: [(40, 170)]
route_count         zero route-count  (layer, expert) pairs: [(40, 170)]
total pairs: 11008
```

Projection-source accounting across all 46 records, which must and does sum to
46 × 768 = 35,328:

| source | projections |
| --- | ---: |
| `measured` | 33,021 |
| `backbone_mean_layer_42` (the three drafter blocks) | 2,304 |
| `uniform_zero_importance_fallback` | **3** |

## F3. Size accounting — measured payload, and residents read off the shard headers

**Routed VQ payload: 75,246,105,792 bytes = 75.246 GB = 70.08 GiB, against a
75.25 GB projection.** 138 files at 545,261,632 bytes each (three of the 138
carry an extra 8 bytes of safetensors header padding). The whole run directory,
including the 271 MB imatrix cache, the 46 records, the manifest and the logs, is
75.52 GB / 70.33 GiB. The projection is reproduced to five significant figures;
there is nothing to explain.

**The residents are now measured, not assumed.** The pilot's 8.32 GB line came
from an assumed 7.828e9 resident weights at 8.5 bpw. Parsing all 48 shard headers
of the pinned revision and classifying every tensor gives:

| bucket | tensors | stored elements | source dtype |
| --- | ---: | ---: | --- |
| routed expert weights | 35,328 | 148,176,371,712 | `I8` (FP4, two per byte) → **296,352,743,424 logical** |
| routed expert scales | 35,328 | 9,261,023,232 | `F8_E8M0`, group-32 — *replaced by the VQ scales* |
| resident block-FP8 (attention, shared experts, MTP `main_proj`) | 390 | 6,304,038,912 | `F8_E4M3` + `F8_E8M0` 128×128 block scale |
| resident BF16 (norms, gate weights, `markov_head`, `confidence_head`) | 443 | 424,505,728 | `BF16` |
| resident F32 (`attn_sink`, hyper-connection scalars, `compressor.ape`) | 339 | 37,741,352 | `F32` |
| `embed.weight` | 1 | 529,530,880 | `BF16` |
| `head.weight` | 1 | 529,530,880 | `BF16` |
| `ffn.gate.tid2eid` | 3 | 2,327,040 | `I64`, never quantized |

Residents excluding embeddings and head: **6,766,285,992** weights. Including
them: **7,825,347,752** — so the pilot's 7.828e9 was 0.03% high and, more
usefully, was *lumping embed + head in with the residents*, which matters because
those two tensors are the ones a resident bpw dial does not obviously apply to.
The logical total is 304,180,418,216 weights, which independently reproduces the
profile's measured 304.18 B and is the arithmetic check that this classification
is not dropping tensors.

**The full runnable artifact.** Routed payload measured; block-FP8 residents
priced at the resident dial; small BF16/F32 residents kept verbatim (they are
0.85 GB and 0.15 GB — quantizing them buys nothing and costs provenance);
`tid2eid` verbatim at `I64`.

| configuration | routed | FP8 res. | BF16 | F32 | embed+head | tid2eid | **total** | vs 98–105 GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q8 residents, BF16 embed/head | 75.25 | 6.70 | 0.85 | 0.15 | 2.12 | 0.02 | **85.08 GB** (79.24 GiB) | 12.9 GB under floor |
| q8 residents, q8 embed/head | 75.25 | 6.70 | 0.85 | 0.15 | 1.13 | 0.02 | **84.09 GB** (78.31 GiB) | 13.9 GB under |
| q6 residents, BF16 embed/head | 75.25 | 5.12 | 0.85 | 0.15 | 2.12 | 0.02 | **83.50 GB** (77.77 GiB) | 14.5 GB under |
| q6 residents, q8 embed/head | 75.25 | 5.12 | 0.85 | 0.15 | 1.13 | 0.02 | **82.51 GB** (76.85 GiB) | 15.5 GB under |

**Stated plainly: the shipped artifact is about 84 GB and the envelope is
98–105 GB, so it fits with 14–22 GB to spare.** That is 4 GB *better* than the
pilot's 83.56 GB headline predicted for the same ladder once the residents are
priced honestly — the pilot's single 8.32 GB resident line becomes 6.70 + 0.85 +
0.15 + 2.12 = 9.82 GB at q8/BF16, i.e. the pilot was 1.5 GB optimistic on
residents and the routed payload is exactly as advertised. The direction of the
error does not threaten anything; the point of re-deriving it is that the next
wave's memory budget (KV cache, activations, the drafter's own working set) is
now working against a number with shard headers behind it.

Evidence: `artifacts/quality/dsv4-vq-materializer-20260814/size-accounting-final.json`
carries the full census and all four pricings.

## F4. Roundtrip spot-checks — a late backbone layer and a drafter block

Both run at HEAD through the committed `bind_path` option, both reproducing the
values from the pre-commit runs bit for bit.

| block | bind path | rel-MSE vs the fit's own reconstruction | cosine | swapped-control rel-MSE | **discrimination ratio** |
| --- | --- | ---: | ---: | ---: | ---: |
| `layers.0` (reference, binder route) | `binder` | 3.465e-14 | 1.000000000 | 1.480e-3 | **4.27e10** |
| **`layers.42`** | `loader` | **3.395e-14** | **0.9999999999999832** | 1.977e-3 | **5.82e10** |
| **`mtp.1`** | `loader` | **2.972e-14** | **0.9999999999999849** | 9.845e-3 | **3.31e11** |

Both report `dense_routed_experts: false` and `unbound_vq_experts: false` from
the adapter's own predicates, and both `passed: true` at a 1e-3 tolerance with
the correct mapping required to be 100x closer than the control.

**The gate/up-swapped control's cosine is again the number to remember.** On
`layers.42` it is **0.9990**; on `mtp.1` it is **0.9951**. A wrongly-named
artifact would have loaded, bound, produced finite output, and looked fine on any
cosine eyeball at a late layer and at a drafter block just as it did at layer 0.
The discrimination ratio is the only thing that separates them, and it is 5.8e10
and 3.3e11 respectively.

**What is *not* proven, precisely.** Both spot checks ran the `loader` route, and
each record says so in its own `bind_proof.gap` field rather than leaving it to
this document:

* `layers.42`: file discovery was not exercised — the caller named the files
  through the loader's `artifact_paths` indirection. `bind_deepseek_v4_flash_vq_experts`
  *can* reach a late backbone layer, but only from a probe deep enough to contain
  it, which would leave layers 0–41 unbound and make the whole-model unbound
  check vacuous. So the choice was between a real unbound check at layer 0 and a
  real forward check at layer 42; both were taken, separately, and neither is
  claimed to be the other.
* `mtp.1`: the same discovery gap, and it is a *ceiling* rather than a choice —
  **there is still no MTP binder** (`tensor_mapping.mtp_binder_exists: false`).
  The drafter forward agreement at 2.97e-14 proves that the bytes this
  materializer wrote to `mtp_drafter.blocks.1.ffn.switch_mlp.{gate,up,down}_proj`
  decode and multiply correctly through `QuantizedVQSwitchGLU` with the same
  projection naming as the backbone. It does **not** prove that a real drafter
  load would find those files, because no code path exists yet that looks for
  them. §7 item 4 is therefore narrowed, not closed: the drafter's *naming* is now
  verified through the same kernel and the same fp64 reference as the backbone's;
  the drafter's *discovery* is still owed a binder.

## F5. Cost of the finished sweep, and the schedule concern in hindsight

From the 46 block records' own timings, not from an estimate:

| term | value |
| --- | --- |
| total wall across 46 blocks | **6,379.9 s = 1.77 h** |
| mean / min / max per block | 138.7 s / 99.9 s (`layers.0`) / 220.5 s (`layers.25`) |
| shard read (3,422,552,064 B, one coalesced span, every block) | mean 0.30 s |
| FP4 → fp32 decode | mean 11.59 s |
| VQ fit | mean 105.21 s (min 74.30, max 152.58) |
| artifact write (1.64 GB) | mean 0.16 s |
| peak RSS, worst block | 6.34 GB |

Against the pilot's 11.21 h: **6.3x cheaper**, and against this report's own
1.3–1.7 h launch estimate, 4% over the top of the range. §7 item 1's worry — that
whatever made the machine 5x faster than the pilot could come back mid-sweep —
did not materialise as a regression to pilot rates; the 2.2x spread between the
fastest and slowest block is GPU contention (§7 item 9) and it is visible in
`status.log` exactly as designed. The concern is not resolved, because nothing
was learned about *why* the pilot was slow; it simply did not bite.

Route-weighted fit error over the finished 46 blocks, which is the proxy the
eval wave will replace:

| projection | mean rel-MSE | min (block) | max (block) | mean expected cosine |
| --- | ---: | --- | --- | ---: |
| gate_proj | 9.000e-2 | 8.504e-2 (`layers.0`) | 9.146e-2 (`layers.11`) | 0.9539 |
| up_proj | 9.001e-2 | 8.646e-2 (`layers.0`) | 9.147e-2 (`layers.11`) | 0.9539 |
| down_proj | 7.768e-2 | 6.018e-2 (`layers.0`) | 9.027e-2 (`mtp.2`) | 0.9604 |

The three drafter blocks land at 8.81–8.82e-2 gate and 8.90–9.03e-2 down, inside
the backbone's band — which is reassuring about the `backbone-mean` importance
policy and, as §4 already said, is not the same thing as it being right.

## F6. What remains unverified

Carried forward from §7, with the finalization's effect on each stated:

1. **The quality gate is still untouched, and this is still not an RC.** Every
   number above is fit error against a proxy objective or a bit-level agreement
   with the materializer's own reconstruction. Teacher agreement has not been
   measured. **Nothing in this finalization licenses calling the artifact an RC**;
   that is the next task and it is deliberately not started here.
2. **MTP importance is still a policy, not a measurement** (§4, §7.3). The three
   drafter blocks are fit against layer 42's column shape because the Wave 3
   capture never saw `mtp.*`. Recorded per block as
   `backbone_mean_layer_42` on all 2,304 drafter projections.
3. **There is still no MTP binder** (§7.4, narrowed in F4). Drafter naming
   verified through the loader; drafter file discovery unverifiable until the
   binder exists.
4. **File discovery is verified at layer 0 only.** The 45 other blocks are
   verified as bytes on disk (size + SHA-256, re-checked by this manifest run)
   and two of them additionally through a forward pass, but no code path has yet
   been asked to *find* a block other than layer 0 by itself.
5. **The 40 pre-fix block records lack `importance_fallback_experts`** (F1). The
   aggregate is sound via `importance_sources` and via the pre-fix code's own
   inability to complete such a block, but the field itself is absent from those
   records and re-deriving it would cost a 40-block re-fit.
6. **`layers.40` expert 170 is fitted against uniform importance** (§6a, §7.6).
   Now confirmed to be the *only* such expert in 11,008 pairs. Its own fit lands
   at 9.4e-2, in band, and its zero route count means it contributes zero weight
   to every route-weighted headline — but if the holdout routes to it, the quality
   there is unmeasured.
7. **Durability is still weaker than the atomicity** (§7.5). `F_FULLFSYNC` is
   still not used. Integrity is *detected* on resume by re-hashing — and this
   manifest run re-hashed all 138 files and found none corrupt, which is evidence
   about this tree today and not a change in the guarantee.
8. **The residents in F3 are priced, not built.** No resident quantization has
   been run for this family; `bind_deepseek_v4_flash_non_vq_weights` binds them
   from source. The 82.5–85.1 GB range is an accounting result over measured
   parameter counts, not a measured artifact on disk.
9. **`code_bits=8` still has no MLX port** (§7.7). Unchanged and deliberate.

## F7. Evidence and reproduction

Run directory `~/keep-artifacts/dsv4-vq-e8p-g512/`: `manifest.json` (rebuilt,
`created_utc` 2026-08-17T21:07:34Z), 46 `records/*.json`, `roundtrip.json`
(layer 0, binder route), `roundtrip-layers42.json`, `roundtrip-mtp1.json`,
`fit-parity.json`, `status.json` / `status.log`, `imatrix-cache.npz` + `.json`.

Committed to `artifacts/quality/dsv4-vq-materializer-20260814/`:
`manifest-final-audit.json` (the manifest minus the per-block file inventory and
the calibration prompt-id list), `roundtrip-layer0.json`, `roundtrip-layer42.json`,
`roundtrip-mtp1.json`, `size-accounting-final.json`, plus the earlier
`fit-parity-layer20.json`, `block-record-layer0.json`,
`imatrix-cache-provenance.json`.

```
uv run --group dev python benchmarks/materialize_dsv4_vq.py manifest
uv run --group dev python benchmarks/materialize_dsv4_vq.py verify-roundtrip \
  --block layers.0  --bind-path binder
uv run --group dev python benchmarks/materialize_dsv4_vq.py verify-roundtrip \
  --block layers.42 --bind-path loader \
  --out ~/keep-artifacts/dsv4-vq-e8p-g512/roundtrip-layers42.json
uv run --group dev python benchmarks/materialize_dsv4_vq.py verify-roundtrip \
  --block mtp.1     --bind-path loader \
  --out ~/keep-artifacts/dsv4-vq-e8p-g512/roundtrip-mtp1.json
```

`tests/test_dsv4_materializer.py` **53 passed** at HEAD. No source changes were
made in this finalization — it is manifest, audit, measurement and this section.
