# KEEP/RAMP Campaign: VQ × MTP Compounding on DeepSeek-V4-Flash

**Date:** 2026-08-11 · **Owner:** Jack Mazac · **Status:** Planned, awaiting go

## Headline

> **First VQ-expert × MTP-verify runtime on Apple Silicon.**
> `tok/s ≈ (bytes/token)⁻¹ × (tokens accepted per verify pass)` — KEEP VQ
> shrinks the left factor, DeepSeek-V4-Flash's built-in MTP head grows the
> right factor. Target: **3–5x aggregate over the AR baseline**, and the
> concrete wow: **a ~284B-parameter frontier model (AA intelligence 52)
> running with speculative speedup on a 128 GB MacBook.**

Nobody has shipped this combination. The four pillars:

1. **Wide-M VQ verify kernels** (Metal) — MTP verify is M=2–8; today's
   `gather_vqmm` is decode-shaped (M=1). Without these the compounding dies.
2. **MTP acceptance rate as a quantization quality gate** — cheaper and more
   sensitive than benchmark deltas; degradation shows up as acceptance drop
   before NLL moves resolve.
3. **MTP-head LoRA recovery post-VQ** — measured from the full 48-shard scan
   (2026-08-11): the drafter is *not* one small dense layer; `mtp.{0,1,2}`
   are three full transformer blocks (dspark_target_layer_ids [40,41,42]),
   each with its own 256-expert FP4 MoE, FP8 attention residents, and a
   `confidence_head` — ~19.3 B logical params of the model's 304.18 B total.
   Still small against the 43-layer backbone, so targeted recovery training
   against the teacher remains cheap relative to the model — but it is
   MoE-shaped recovery, and the drafter's own experts are also VQ candidates.
4. **Teacher campaign, now trivial** — the ~172 GB source (FP4 experts, FP8
   residents) fits one 8xH100 node entirely in HBM with room to spare.
   GLM-5.2's 1 TB never did.

Decision lineage: `docs/deepseek-v4-flash/2026-08-11-pivot-decision.md` →
`docs/deepseek-v4-flash/research/2026-08-11-inspiration-repos.md` → this plan.

## Ground truth

### Local hardware (measured 2026-08-11)

| Item | Value |
| --- | --- |
| Machine | Apple **M5 Max**, 40-core GPU, **128 GB** unified memory |
| Disk free | 1.4 TB |
| NAX tensor units | **Yes** (M5 family) — dflash's NAX M=16 verify path and oMLX's `gather_qmm_rhs_nax` apply |

### Model placement math (measured, all 48 shards: 304.18 B logical params — 277.1 B backbone routed + 19.3 B MTP-drafter routed + ~7.8 B resident)

Wave 1 measurement (2026-08-11) settled the published 284-vs-304 discrepancy:
304.18 B logical weight parameters at revision `7872f01b` (FP4-in-I8 expert
weights counted at two logical params per stored byte, scales excluded).
Sizes corrected 2026-08-11 against the real snapshot: **the source is FP4-in-I8
routed experts + FP8 e4m3 residents, not FP8 throughout**, so there is no
284 GB artifact and no separate "community mxfp4" tier below the source. The
release *is* the 4-bit-expert artifact.

| Artifact | Est. size | Fits 128 GB resident? |
| --- | ---: | --- |
| Source as released (FP4-in-I8 experts + `F8_E4M3` residents) | **~172 GB measured** (48 shards) | **No** — ~44 GB over the machine, before any KV or activation budget. Teacher work goes to AWS or layer-streams, and the same-machine *control* is this artifact, so the baseline cannot run resident locally either. |
| Source dequantized to BF16 (the ratio denominator) | ~568 GB | **No** — never resident anywhere but the 8xH100 node |
| KEEP VQ ~3.0 bpw experts + q8 residents | ~120 GB | Borderline — over the safe wired ceiling |
| KEEP VQ ~2.5 bpw experts + q6/q4 residents (mtplx `proj_quant` pattern) | ~98–105 GB | **Yes, tight** — this is the local deliverable envelope |

Consequences the plan is built around:

- **Every full-model teacher pass is an AWS job.** Local teacher = GLM-style
  layer streaming, which the GLM campaign proved is weeks-slow. Don't.
- **The local runnable artifact is ≤ ~2.5–3 bpw experts with quantized
  residents.** That is not a compromise — sub-mxfp4 bpw at better quality is
  exactly KEEP's pitch.
- **Same-machine control problem:** the family gate requires a same-machine
  baseline, but the released 4-bit-expert source (~172 GB) doesn't fit the Mac
  either. Wave 2 decision: control = RAMP-streamed source (beyond-RAM plans
  exist in `docs/superpowers/plans/`), or affine q2/q3 resident artifact, or
  both. Gate language must name whichever is chosen.

### Dataset — teich corpus (Jack's real traces), reused as directed

Retained at
`s3://keep-glm52-models-246813579024-us-west-2/teich-pack/glm52-coding-agent-initial-v2-20260713.json`
(94.6 MB, 257 coding-agent sessions, 10.8 M raw / 2.5 M supervised tokens
under the **GLM-5.2 tokenizer**). V4's tokenizer differs (vocab 129,280 vs
154,880), so all token counts shift — Wave 1 re-tokenizes and re-splits.

Split plan (non-overlapping, per the family-template eval contract):

| Split | Purpose | Sessions (target) |
| --- | --- | ---: |
| `calibration` | imatrix / activation stats for VQ fitting | ~40 |
| `mtp-train` | MTP-head LoRA recovery targets | ~120 |
| `report` | eval rows for recipe iteration | ~30 |
| `selection` | recipe/lever selection | ~30 |
| `holdout` | touched once, final RC gate | ~37 |

Teacher outputs needed per split: top-K logits (report/selection/holdout),
hidden states before MTP head + MTP logits (mtp-train), activation
statistics (calibration). **All from the same AWS block run.**

## Orchestration model

Dynamic workflow, Jack-approved: I orchestrate; subagents execute.

| Role | Model | Work |
| --- | --- | --- |
| Orchestrator (this session) | Opus 5 | Wave sequencing, gate reviews, AWS mutations, spend, commits |
| Design/kernel/verify agents | **Opus 5** | Metal kernels, adapter architecture, numerics parity, adversarial verification of results |
| Breadth/mechanical agents | **Sonnet 5** | Test scaffolds, renames, conversions, doc updates, split tooling, monitors |

Standing rules carried forward: heavy jobs hold `.keep-heavy-job.lock`;
long host jobs via `nohup … & disown` with wired-limit vars unset; explicit
`git add` paths; AWS only under `keep-gpu` behind
`assert_rnd_aws_account.sh`; **no new paid spend without explicit approval
per purchase** — each capacity block is a separate ask with exact price.
Every wave ends with an adversarial verify pass (agents try to refute the
wave's claimed results) before the gate counts as passed.

---

## Waves

### Wave 0 — Decoupling foundation (no GPU; ~1 session)

Execute the already-written cutover plan
(`docs/superpowers/plans/2026-08-11-keep-deepseek-v4-flash-cutover.md`),
tasks 1–10: commit pending sky fixes + pivot docs, open converter registry,
family registry in `ramp.models`, V4 profile + policy, FP8 e4m3 block
dequant (**resident path only** — attention + shared experts; the routed
experts are FP4-in-I8 with E8M0 group-32 scales and need their own decoder,
deferred to Wave 2), CFN watchdog param, docs.

- **Agents:** Sonnet 5 per task (mechanical, tests specified in plan);
  Opus 5 reviewer between tasks.
- **Gate:** full test suite green; 10 commits landed.

### Wave 1 — Source acquisition & measurement (network + CPU; download hours, no GPU)

1. Download `deepseek-ai/DeepSeek-V4-Flash-0731` weights (~172 GB, 48 shards)
   to local disk (1.4 TB free; keep ≥ 400 GB headroom). Pin `revision`.
2. Measurement pass (headless, shard-index only — no model load): true
   parameter count (settles 284 B vs 304 B), routed/resident split, the
   three structural layer layouts (hash/gs32, score/gs32, score/gs64 per
   mtplx), `first_k_dense_replace`, per-layer `compress_ratios`, MTP tensor
   names, tokenizer hash. Update `models/deepseek-v4-flash-0731.yaml` with
   measured values; refresh placement table above.
3. Re-tokenize teich pack with the V4 tokenizer; recompute supervised
   positions; write the 5-way split manifest with per-split sha256; commit
   manifest (not the corpus).
4. S3 mirror of source weights **deferred** until the Wave 3 block is
   booked (upload ~172 GB from laptop is slower than HF-download on the GPU
   node — the node pulls from HF directly, S3 caches for reruns).

- **Agents:** Sonnet 5 (download tooling, split tooling), Opus 5
  (measurement analysis + profile update review).
- **Gate:** profile validates against downloaded `config.json`; split
  manifest deterministic; placement table re-issued with measured numbers.

### Wave 2 — V4 adapter + local runnable proof (local Metal; overnight-scale runs begin)

1. Vendor oMLX `patches/deepseek_v4` model (Apache 2.0, mlx-lm PR 1192
   lineage) into `ramp/models/deepseek_v4_flash_adapter.py`; strip
   server-specific paths; keep DSpark attention + hyper-connections +
   LimitedSwiGLU; **load the MTP block intentionally** (not skipped — pillar
   3 and 4 need it).
2. Bind surface: swap their `SwitchGLU` for `QuantizedVQSwitchGLU`; satisfy
   the Wave-0 family-registry contract (`bind_vq_experts`,
   `bind_non_vq_weights`, `has_unbound`); MTP tensors bound to a dedicated
   sub-module.
3. Resident quantization pass (mtplx `proj_quant` pattern): q8 and q6/q4
   variants of attention + shared-expert + MTP residents from the FP8
   source via `keep.convert.fp8_block`.
3b. **Routed-expert FP4/E8M0 decoder** (`keep.convert`): unpack FP4 two-per-byte
   `I8` weights against `F8_E8M0` group-32 scales to BF16. This is the
   prerequisite for every VQ fit in Wave 5 and has no existing KEEP path —
   `fp8_block` decodes residents only, `convert/nvfp4.py` is a different
   (NVFP4) layout. Byte-exactness verified against the reference
   implementation's dequant on a sampled expert set.
4. **Local smoke:** first N layers resident on Metal, logit parity vs
   teacher reference slices (from Wave 3 cache when available; structural
   parity checks before then).
5. Same-machine-control decision (see placement math): build the control
   artifact (streamed mxfp4 or resident affine-q3) and name it in the gate.

- **Agents:** Opus 5 (adapter + binding + numerics), Sonnet 5 (resident
  conversion runs, artifact audits, tests).
- **Long runs:** resident conversion + artifact audits — hours each,
  overnight local, lock held.
- **Gate:** adapter binds with zero dense routed experts and zero unbound
  VQ experts on a synthetic-VQ artifact; resident artifacts audited;
  control artifact chosen and reproducible.

### Wave 3 — Teacher block run on AWS (the one big GPU rental)

**Everything the campaign ever needs from the full-precision teacher, in one
24 h p5.48xlarge capacity block (~$997 at the 2026-08-11 offering price;
requote at booking).** Batching all teacher work into one block is the
no-deferred-known-wins move — a second block later means another ~$1k and
another calendar day.

On-node sequence (single 8xH100, model fully in HBM, FP8 native):

1. Pull weights from HF; verify shard sha256s; mirror to S3 `source-hub/`
   for reruns.
2. **Teacher-cache generation:** top-2048 logits per supervised position,
   report + selection + holdout splits (~97 sessions). GLM-experience says
   the corpus prefill is minutes-to-hours when the model is resident —
   budget 6 h with I/O.
3. **MTP-train targets:** final hidden states + MTP-head logits for the
   ~120 `mtp-train` sessions — budget 6 h.
4. **Calibration statistics:** activation ranges / imatrix over the
   `calibration` split — budget 2 h.
5. **DSpark divergence probe** (research question from the repo analysis):
   teacher logits sparse vs full attention on a 10-session subset; answers
   which mode the distillation should match — budget 1 h.
6. Upload everything to S3 with manifests + sha256; **spot-verify from the
   Mac before releasing the node.**

Infra notes: reuse `keep-glm52-gpu` VPC/IAM/budget stack as-is (names are
cosmetic); new run id `dsv4-teach-<date>`; submission uses the fixed
`--detach-run` path; watchdog rule re-enabled for the block window via the
new CFN parameter, disabled after. Capacity block purchase via
`guarded_purchase_capacity_block.py` — **requires Jack's explicit yes on the
exact quote**; the approver's API-throttling warning respected (blocks are
reserved — zero RunInstances retry loops).

- **Agents:** Opus 5 (run scripts + on-node verification design),
  Sonnet 5 (monitors, S3 audits). Orchestrator babysits the block window.
- **Gate:** teacher cache + MTP targets + calibration stats in S3, sha256
  manifests verified from the Mac, total spend = block price only.

### Wave 4 — Wide-M VQ verify kernels (local Metal; parallel with Wave 3)

No dependency on weights — synthetic shapes. Starts as soon as Wave 0 lands.

1. Port dflash `verify_qmm` dispatch structure onto `gather_vqmm`:
   M∈{2,3,4,8} variants, K-split for K∈{2048,4096}, simdgroup-MMA baseline,
   **NAX M=16 tensor-unit path** (M5 Max has it), steel fallback.
2. V4-Flash shapes: gate/up K=4096 N=2048, down K=2048 N=4096, top-6
   routing, 256-expert gather.
3. Byte-exact parity harness vs the existing reference kernel
   (MLX/Metal-standardization directive: GPU-batched, parity-gated).
4. Microbenchmarks: verify-shaped route counts, target ≥ 2.5x over running
   M=1 decode kernels M times.

- **Agents:** Opus 5 exclusively (kernel work + adversarial parity review);
  Sonnet 5 for benchmark scaffolds.
- **Long runs:** benchmark sweeps, hours, local.
- **Gate:** parity byte-exact on all shapes; microbenchmark table committed.

### Wave 5 — VQ materialization + MTP integration (local; the heavy overnight phase)

1. **Pilot:** single layer E8/E8P VQ fit from FP4-decoded experts (via the
   Wave-2 FP4/E8M0 decoder, *not* `keep.convert.fp8_block`) using
   Wave-3 calibration stats; measure wall-clock; extrapolate 43 layers ×
   256 experts × 3 projections; publish the schedule before committing to
   the full sweep.
2. Full materialization at 2.5 and 3.0 bpw ladders with oQ-style
   Super-Weights floors (down_proj protected above base). Overnight
   local runs, lock held, resumable per-layer.
3. **MTP verify runtime:** depth 1–2 greedy-first (exactness ladder later),
   using Wave-4 kernels; acceptance-rate instrumentation.
4. **Acceptance gate wiring:** acceptance-vs-teacher at fixed depth joins
   NLL/KLD/top-1 in the family gate rows.
5. **MTP-head LoRA recovery:** train on Wave-3 mtp-train targets, local
   (single small dense block — Mac-trainable); accept only if acceptance
   improves with report-split NLL flat, mtplx-Forge-style honest verify.

- **Agents:** Opus 5 (runtime integration, recovery trainer), Sonnet 5
  (sweep monitors, eval row tooling).
- **Long runs:** materialization sweep (estimate from pilot; expect
  multiple overnights), recovery training (hours).
- **Gate:** artifact audits pass (`dense_routed_experts=false`,
  `unbound_vq_experts=false`); acceptance within threshold of source after
  recovery; eval rows on report split within recipe targets.

### Wave 6 — Wow measurement + RC (local; the headline number)

1. Lane-S-style same-machine benchmark, quiet-window discipline, fresh
   process rows: AR baseline (named control) vs VQ+MTP at tuned depth.
   Pageouts/swapouts are acceptance gates.
2. Holdout split touched once: final quality gate.
3. RC packaging per family template: manifests, audits, eval evidence,
   benchmark rows, reproduction commands.
4. Publish decision (HF upload, blog) — **separate approval, not covered
   by this campaign's authority.**

- **Agents:** Opus 5 (gate adjudication + write-up), Sonnet 5 (packaging).
- **Gate:** measured aggregate speedup vs named baseline with quality
  within holdout thresholds — the 3–5x claim replaced by a real number.

---

## Long-run & spend schedule

| Wave | Run | Where | Est. duration | Est. cost |
| --- | --- | --- | --- | ---: |
| 1 | Weight download ~172 GB | Local (network) | hours (bandwidth) | $0 |
| 2 | Resident conversions + audits | Local Metal | 2–3 overnights | $0 |
| 3 | **Teacher block: cache + MTP targets + calibration + probe** | **AWS p5.48xlarge capacity block** | 24 h block, ~15 h planned work | **~$997, one purchase, pre-approved quote required** |
| 4 | Kernel benchmark sweeps | Local Metal | hours, repeated | $0 |
| 5 | VQ materialization ladders | Local Metal | pilot then N overnights (pilot decides N) | $0 |
| 5 | MTP LoRA recovery | Local Metal | hours | $0 |
| 6 | Wow benchmark + holdout | Local Metal | 1 overnight | $0 |

AWS fallback triggers (local → AWS offload, each its own approval): pilot
extrapolation puts the materialization sweep beyond ~5 overnights; local
recovery training exceeds ~24 h; or a second teacher pass becomes necessary
(missing split, corrupted cache — mitigated by on-block verification).

Parallelism: Wave 4 runs concurrent with Waves 1–3 (no weight dependency).
Waves 2 and 3 can overlap once Wave 1's measurement lands (adapter work
needs shapes, not the teacher). Critical path: 0 → 1 → 3 → 5 → 6.

## Risks / open decision points

1. **mlx-lm PR 1192 merge status** — if merged upstream, depend instead of
   vendor (Wave 2 checks first).
2. **Same-machine control** — streamed mxfp4 vs resident affine-q3; decided
   in Wave 2, named in every gate after.
3. **2.5 bpw quality on 256 small experts** — group-size policy is
   unproven on this expert geometry; Wave 5 pilot includes a 2-layer
   quality probe before the full sweep.
4. **MTP exactness ladder** — greedy acceptance first (dflash-style,
   lossless vs greedy AR); Leviathan–Chen residual sampling (mtplx-style,
   distribution-exact at temperature) as a stretch inside Wave 5.
5. **Wired-limit ceiling at ~105 GB artifacts** — if the 2.5 bpw + q6
   residents build doesn't sit under the ceiling with KV headroom, the
   RAMP streaming plans (`docs/superpowers/plans/`, currently off critical
   path) come back on.

## Immediate next steps (recommended order)

1. **Go/no-go on this plan** — Jack.
2. **Launch Wave 0 now** (no cost, no downloads): subagent-driven execution
   of the cutover plan's 10 tasks.
3. **Start the Wave 1 download tonight** (disk is free, bandwidth is the
   only cost) so measurement can run tomorrow.
4. **Kick Wave 4 kernel work in parallel** right after Wave 0 lands —
   longest-lead Opus work, zero dependencies.
5. **Book the Wave 3 capacity block only after Wave 1 measurement**
   confirms corpus token counts and the on-node run scripts exist — then
   one purchase approval with the live quote.
