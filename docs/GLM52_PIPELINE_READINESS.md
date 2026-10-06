# GLM-5.2-REAP One-Command RC Pipeline — Readiness Plan

**Purpose.** GLM-4.5-Air was the stepping stone to validate levers and harden the
`base -> community-wow RC` pipeline. The real target is **GLM-5.2-REAP** (176B / 504B
variants) on **rented GPU capacity**, where research iteration is expensive. Everything
below is about making the pipeline run **clean, deterministic, and cost-minimal on the
first paid run** — no research-on-the-clock.

This doc consolidates the 6-lever campaign (2026-07-07/08) into concrete pipeline
requirements. Source of truth for numbers: `DISCOVERY.md`.

---

## 1. What the Air campaign settled (bake these into the pipeline)

| Lever | Verdict | Pipeline action for GLM-5.2 |
|---|---|---|
| **plan1 recovery sidecar** (L1) | **WIN** — rank-4 layer-45 gate/up/down KD sidecar on route+math rows reached top1 0.852 (met wow bar), speed-tied with the accepted RC | The recovery template: a **small low-rank KD sidecar on a late hard layer**, trained on domain-balanced rows. Port as the recovery step. |
| **AGQ affinity imatrix** (L5) | **GO (weak)** — +0.6–1.0pp global top1 over routing-weighting, free at calibration stage | Default `--importance-key affinity_weighted_importance` in the sweep step. Re-verify post-recovery. |
| **P-step closed-form scale refit** (L4) | **KEEPER** — `fit_scale_delta_sidecar_least_squares` cuts block-local KLD 53% vs RTN, no training loop | Add as a **deterministic pre-recovery scale-refit step** (cheap, no GPU training). |
| **NAX Option B decode->gather_mm** (L2) | **Conditional win** — 1.04x q2 at 4k gate/up prefill; motivates fused Option A | Ship as a prefill engine variant selectable by (projection, token-count). For long-context serving. Not required for RC acceptance. |
| **PV-Tuning coord-descent** (L4) | **NO-GO** — V-step drags away from scale-optimum on frozen E8 | Do NOT implement. |
| **YAQA Kronecker H_out** (L3) | see DISCOVERY (running) | Proxy H_out moved KLD ~0; true-gradient disambiguation pending. Likely NOT worth the infra on rented GPU unless the disambiguation flips. |
| **565 logit-bias** (L6) | **superseded** — eval-fitted, diagnostic-taint; plan1 legitimately reaches its ceiling | Keep diagnostic-only. Never promotable. |

**Net recipe shape for GLM-5.2:** `collect-imatrix (affinity) -> materialize-sweep
(AGQ importance-key, rerounds) -> P-step scale refit -> low-rank KD recovery sidecar on
late hard layer(s) -> RC gates`. Every step already exists as a build op for Air.

---

## 2. Robustness fixes that MUST be in before paid GPU (silent-wrong-verdict class)

These are the bugs that would waste rented-GPU clock by silently shelving a good RC or
reporting fallback numbers. Two are fixed; the rest are required.

1. **Lane S floating-denominator (FIXED 2026-07-07).** `run_glm45_air_rc_pipeline.py`
   now guards the q2 denominator against a golden baseline and reclassifies
   suspect-window ratio rejects as `RECOVER_lane_s_remeasure_needed`. **GLM-5.2 needs
   its own golden q2 baseline measured once** (see `_load_golden_q2_baseline`, env
   `GLM45_AIR_GOLDEN_Q2_SECONDS` or `artifacts/quality/golden-q2-baseline-prefill1k.json`).
   Generalize the path/env to be model-scoped.
2. **materialize-sweep silent fallback (REQUIRED FIX).** When a recipe requests
   `budget: X` but the tier allocator can't emit a distinct `bpwXpY` candidate, eval
   silently consumes a fallback artifact (this produced the bogus 0.4424 "route
   collapse"). Fix: **fail loud** when the requested `artifact_bpwXpY` output is not
   produced with `reround_action_count > 0`. (`_sweep_output_names` / `_materialize_links`.)
3. **Engine/tier compatibility (REQUIRED GUARD).** `vq_e1_routed_nax_e8` requires
   `code_bits=8` on **every** group — any mixed E8/E8P (mid-tier) candidate crashes at
   eval. The pipeline must either (a) validate engine⇄tier compatibility before eval, or
   (b) pick the eval engine from the candidate's actual tier map. On GLM-5.2, dynamic
   precision that mixes tiers needs an engine that supports mixed code_bits, or the RC
   must stay all-one-tier.
4. **Memory-safe serial execution (REQUIRED).** Concurrent full-model loads OOM-crash a
   128GB Mac (observed this session). The runner must **serialize model-loading steps**
   (a global heavy-job lock). On rented multi-GPU this maps to per-GPU job scheduling.
5. **Interleaved Lane S measurement (RECOMMENDED).** Beyond the golden guard, measure
   candidate+q2 in one memory-quiet window so the ratio never depends on window luck.

---

## 3. Parameterization gaps (Air-hardcoded -> model-agnostic) — the porting audit

The recipe *system* (`src/mlx_vq/build/`) is model-agnostic, but the Air *pipeline*
(`run_glm45_air_rc_pipeline.py`, `glm45_air_rc.sh`, the primitives) hardcodes Air
specifics. Porting to GLM-5.2-REAP requires parameterizing:

- **Model identity**: `GLM45_AIR_MODEL_ID`, source dir, config/index paths.
- **Architecture constants**: layer count, hidden/intermediate dims, expert count/top-k,
  the "hard layers" set (Air used 31/36/41/45; REAP will differ — derive from an
  attribution probe, not a hardcoded list), fragile-GLU layers, down_proj tier floors.
- **Prompt sets / teacher caches**: `air_imatrix_calib_v1`, `air_vq_ladder_*_v1`,
  the 128-row report/selection/holdout caches — regenerate for GLM-5.2.
- **Gate thresholds**: `rc_gates.py` COMMUNITY_WOW / BALANCED_HARD targets are model-
  independent (top1>=0.85 etc.) and can stay; bpw/Lane S targets may need per-model tuning.
- **REAP specifics**: REAP is a pruned-expert variant — expert count differs from stock
  GLM-5.2; the router/expert-count assumptions in the adapter must read from config.

**Concrete deliverable:** a `keep build recipes/glm52reap__*.yaml` full base->RC recipe
+ a model-profile config so the same runner drives any model. This is the P2/P4 core.

---

## 4. Cost-minimal validation order on rented GPU

To avoid research-on-the-clock, validate in this order (cheap -> expensive), stopping if
a gate fails:

1. **Dry-run + validate** the recipe (`keep validate`, `keep build --dry-run`) — free,
   local, no GPU. Confirms lineage, argv, gate profiles.
2. **Golden q2 baseline** (one quiet measurement) — cheap, unblocks Lane S.
3. **collect-imatrix + sweep + P-step refit** — the quantization is deterministic/cheap;
   check bpw + reround counts + engine⇄tier compatibility BEFORE any recovery training.
4. **RC gates on the bare quant** — confirm the base is sane before spending on recovery.
5. **Recovery sidecar training** — the only real GPU-training cost. Use the plan1
   template (small rank, late hard layer, domain-balanced rows). Keep row counts bounded.
6. **Full RC eval + Lane S** — final acceptance.

Everything through step 4 should be provable on cheap/spot capacity; only step 5 needs
sustained GPU. The pipeline should **checkpoint after each step** (the ledger already
does) so a failed paid run resumes without re-paying for completed steps.

---

## 5. Open decisions (for the human)

- **REAP variant**: 176B vs 504B — pick per target quality/serving budget. The 504B-GGUF
  variant implies llama.cpp serving; KEEP's runtime is MLX/Metal — confirm the serving
  target (MLX on Apple silicon vs GGUF/llama.cpp vs rented CUDA) before quantizing, since
  the kernel/engine constraints (§2.3) are serving-stack-specific.
- **Delete stock GLM-5.2** once a REAP variant is chosen and downloaded.
- **Distributed collection**: 504B likely needs multi-device even for calibration; the
  2-Mac JACCL layer-split infra is the local analog of rented multi-GPU sharding.

## 6. MLX CUDA backend — the serving-target fork, resolved (researched 2026-07-08)

MLX has a **CUDA backend** (`pip install mlx[cuda12]`, NVIDIA SM≥7.5, driver≥550, CUDA≥12,
Linux). Design model: **author/test on Mac, run on a Linux/NVIDIA server** — exactly the
"develop local, run on rented GPU" workflow. So it is NOT "MLX vs CUDA": **MLX runs on
NVIDIA.** What matters is which *parts* of KEEP port.

**Portable to CUDA (runs on rented NVIDIA via mlx[cuda]):**
- Dense BF16 teacher forward (imatrix collection, eval).
- The KD-recovery **P-step surrogate** — pure MLX ops (`mx.take/sum/exp`), no custom kernel.
- MLX-native quantized MoE inference: `QuantizedMatmul` (landed ~Jan–Apr 2026), quantized
  **GEMV** (#3180, Mar 2026), and critically **GatherQMM** — the MoE routed op — whose
  `NO_GPU` marker was removed and matrix-matrix sm80/naive + gather_qmv paths landed
  (#3321 Apr 6, #3417 Apr 18 2026). MoE routed inference on CUDA now works for AFFINE quants.

**NOT portable (Apple-silicon only):**
- KEEP's **custom E8P VQ Metal kernels** (`gather_vqmm` + E8P lattice decode) — `mx.fast.metal_kernel`
  is Metal source; there is no CUDA execution path for a CustomKernel. Would need a CUDA reimpl.
- **NAX prefill** — Apple Neural Accelerator hardware; irrelevant on NVIDIA (which uses its
  own tensor cores via CUTLASS/cuBLAS in MLX's native quant kernels).

**Consequence — the real fork is codebook, not framework:**
- **(A) Serve on Apple silicon (MLX/Metal):** keep E8P VQ + NAX; the current pipeline IS the target.
- **(B) Serve on rented NVIDIA (MLX/CUDA):** express the RC in **MLX-native affine quant**
  (int4/int8 group-quant, which now has working CUDA GatherQMM) and apply KEEP's *methodology*
  (importance-driven mixed-precision allocation + KD-recovery sidecar) on top. Drop/defer the
  E8P custom kernels. The pipeline, method, and recovery all run on CUDA; only the specific
  E8P lattice codebook is left behind.
- **(C) GGUF/llama.cpp** (the REAP-GGUF link): a *different* quant ecosystem — out of scope for
  the MLX-based KEEP pipeline. Use the REAP **safetensors** variants (176B/504B) as the base, not GGUF.

**Open quality question (cheaply testable on Air, Metal, before renting):** KEEP's 2-bit edge
came from E8P VQ beating affine. Does the allocation+recovery methodology still clear the wow
bar on **MLX-native affine quant** (the CUDA-portable representation)? Measure this on Air first
— it decides whether path (B) is viable or whether NVIDIA serving needs a CUDA E8P kernel port.

---

## GLM-5.2-REAP implementation gaps

Scope checked on 2026-07-09 for `0xSero/glm-5.2-reap-504B-v2`, cached snapshot
`6c9241aa05fb243a0edb7c804c213ec1cf5c920d`.

### What exists today

- Model-policy validation keeps stock GLM52 at 256 routed experts by default, but
  now accepts the REAP model's 168 experts only when the caller supplies the exact
  `glm52-reap-504b-v2` profile, Hugging Face model ID, and pinned revision. The
  profile-aware path retains all family, IndexShare, and required-field checks and
  cross-checks the profile against the loaded config; it does not widen the stock
  GLM52 contract.
- Installed upstream MLX-LM has `mlx_lm.models.glm_moe_dsa` and complete basic
  `ModelArgs`, but not the IndexShare runtime surface KEEP tracks:
  `installed_glm52_support` checks `ModelArgs` plus runtime source for
  `prev_topk_indices`, `indexer_types`, and IndexShare cache behavior
  (`src/mlx_vq/models/glm52_policy.py:77-114`). Local check returned
  `available=True`, `missing_model_args=()`, `indexshare_available=False`, with
  missing `ModelArgs.indexer_types`, `ModelArgs.index_topk_pattern`,
  `ModelArgs.index_topk_freq`, `ModelArgs.index_skip_topk_offset`,
  `runtime.prev_topk_indices`, `runtime.indexer_types_schedule`, and
  `runtime.indexshare_cache`.
- KEEP has a GLM52 VQ runtime adapter. It constructs sparse GLM52 MoE layers with
  `Glm52VQMoE` and requires a bound `switch_mlp` before execution
  (`src/mlx_vq/models/glm52_vq_adapter.py:283-323`), uses IndexShare cache
  scheduling in `GLM52VQModel.make_cache`
  (`src/mlx_vq/models/glm52_vq_adapter.py:423-430`), binds per-layer VQ
  gate/up/down artifacts from `layer-000NN-{gate,up,down}_proj.safetensors`
  (`src/mlx_vq/models/glm52_vq_adapter.py`), and strictly binds one decoder
  layer's non-VQ weights while excluding neighboring layers, routed experts,
  and MTP tensors. The non-VQ binder validates every selected shape and the
  complete missing/unexpected set before mutating the layer; it also splits the
  source `kv_b_proj.weight` into the runtime's `embed_q` and `unembed_out`
  parameters.
- Stream conversion now supports the pinned native ModelOpt NVFP4 source directly.
  It validates the producer contract, resolves each packed weight plus independently
  indexed block/global-scale companions, decodes one expert projection at a time to
  finite C-contiguous `float32`, and feeds the existing E8/E8P quantizer without a
  dense checkpoint. Plans, group artifacts, resume records, manifests, and CLI JSON
  identify `source_weight_encoding=modelopt_nvfp4` and
  `source_decoder=modelopt_nvfp4_v1`; `skip_existing` validates the existing
  tensor shapes/dtypes, quantization settings, embedded codebook tensor, scale
  estimator, and source lineage before counting a group ready.
- The pinned-snapshot plan is green for exactly 225 groups over layers 3-77:
  37,800 routed projection bundles, 53 cross-shard bundles, 61 required main-model
  source shards present, and zero layer-78 MTP groups. Peak decoded expert working
  size is 50,331,648 bytes. Synthetic integration tests prove E8 and E8P outputs are
  byte-identical between serial and explicit two-worker conversion.
- Thirteen registered GLM52 build ops now exist. `glm52-source-audit` validates the
  pinned profile/config/index/NVFP4 mapping and `glm52-source-payload-audit`
  validates every selected bundle member's local safetensors header, dtype, shape,
  byte length, and physical file extent without decoding payloads. Both emit JSON
  and append-only JSONL evidence before any nonzero exit.
  `glm52-moe-materialize-groups` consumes the payload-audit evidence and converts
  an explicit bounded or full-plan selection directly into separate gate/up/down
  E8 or E8P artifacts. `glm52-moe-artifact-audit` reconstructs the selected
  canonical plan and independently verifies the complete artifact tree, tensor
  schema, codebook, hashes, physical extents, pinned lineage, resume identity,
  and exact routed/accounting figures. `glm52-vq-validate` runs `scripts/validate_glm52_vq.py`
  against an already-existing source/index and artifact. The parked low-level recipe
  exercises only layer-local validation for layers 3 and 77
  (`recipes/glm52_504b_vq_validation_probe_20260702.yaml:20-51`).
  `glm52-non-vq-pack` raw-range repacks and source-relative audits the complete
  main-model BF16/F32 non-routed inventory into a deterministic lean package.
  `glm52-layer-forward-probe` authenticates that production package plus a
  passing bounded routed-artifact audit, strictly binds one sparse decoder
  layer, and records a finite layer-local forward with explicit negative
  whole-model, generation, tokenizer, long-context, and speed claims. The
  checked low-level materialization recipe now exercises all six ordered steps.
  Four additional noncacheable runtime-preflight ops run the pinned IndexShare
  static plus tiny synthetic contract, the immutable local tokenizer/chat
  payload audit, bounded tiny-fixture `generate_step` compatibility, and the
  header-only full-bind preflight. They are wired in that order by
  `recipes/glm52_reap_504b_runtime_preflight_20260709.yaml`; the terminal
  full-bind step deliberately exits nonzero while routed coverage is incomplete.
  Two more ops freeze the source-relative family policy and exact local-tokenized
  report/selection/holdout prompt pack before teacher or candidate metrics.

### What is missing versus the Qwen family chain

The Qwen high-level chain exists as source audit -> source payload audit ->
materialize groups -> artifact audit -> bind probes -> runtime probes ->
tokenizer/eval/benchmark probes -> family gate
(`src/mlx_vq/build/highlevel.py:429-767`). Its backing ops are registered in
`src/mlx_vq/build/ops.py:826-1490`.

GLM52 now has the source-audit, source-payload-audit, materialize-groups,
artifact-audit, lean non-routed-package, bounded layer bind/forward, IndexShare
static/synthetic runtime, tokenizer readiness, tiny-fixture `generate_step`, and
header-only full-bind-preflight equivalents, plus a frozen family policy and eval
prompt authority. The full-bind preflight is a real
fail-loud inventory gate, but it is currently **BLOCKED** on 219 missing routed
groups and does not construct or bind the production model. The remaining gaps
are production whole-model bind/generation, production long-context IndexShare,
dequantized-source teacher-cache payloads, contextual eval rows, route/math
comparisons, resident benchmark rows, and the family gate
corresponding to the Qwen family registrations
(`src/mlx_vq/build/ops.py:1077-1490`).

### Template decision

Do not add a compiling GLM52 high-level recipe yet. A real template would have to
reference missing ops or silently reuse Qwen/Air ops against incompatible artifact
contracts. The current `glm52_vq_groups` high-level template is therefore a guard
that fails with a clear message and points back to this section. Remove that guard
only after all 225 routed groups pass audit, the production model binds and
generates without dense routed materialization, and the teacher/eval/benchmark/
family-gate chain is implemented and tested. The checked runtime-preflight recipe
is diagnostic evidence, not the missing high-level release recipe.

**Design principle for P2:** build the one-command pipeline so the **quantized representation is
pluggable** (E8P-VQ vs MLX-native-affine) behind the same recipe/gates. Then the *same* pipeline
targets Apple (E8P) or rented NVIDIA (affine) by swapping the representation — no fork in the driver.

## 8. CRITICAL SOURCE FINDING (2026-07-09): the pinned REAP source is native ModelOpt NVFP4

`0xSero/glm-5.2-reap-504B-v2` is downloaded at immutable revision
`6c9241aa05fb243a0edb7c804c213ec1cf5c920d`: 63/63 snapshot shards, about
308.8 GB. Its `quantization_config` is native **ModelOpt NVFP4 W4A4**, not
canonical compressed-tensors. The supported weight contract is static float 4-bit,
group size 16. `input_scale` belongs to activation quantization metadata and is not
part of weight reconstruction. There is no BF16 REAP teacher in this pinned repo.

**Consequence:** the FP4 model IS the source. KEEP's GLM-5.2 job is deterministic
FP4 dequant -> teacher/reference -> uniform E8 routed compression plus a lean repack
of the source-precision non-routed tensors, with KD recovery against the dequantized
model. Header-derived accounting projects 98,433,923,808 tensor-payload bytes and
1.5934433 bpw over the exact 494,194,805,304-parameter main model; the user accepted
that revised headline on 2026-07-09. The non-routed component now physically exists
and passes audit, but the whole remains a projection until all 225 routed groups
physically exist and pass audit. Quality ceiling = the FP4
model (which is what the community runs of REAP-504B anyway).

### Decoder checkpoint complete

The converter now reconstructs producer-matched weights as
`E2M1 x (E4M3FN block scale x float32 global scale)`. The low nibble maps to the
even logical column, the high nibble to the odd column, and E2M1 code 8 is
canonicalized to positive zero. Bundle resolution never assumes companion shard
co-location, and malformed encodings, dtypes, shapes, payload lengths, signed/NaN
block scales, non-finite/non-positive global scales, or finite inputs that would
overflow the decoded float32 output fail loud.

Fresh checkpoint evidence on 2026-07-09:

- focused decoder/planner/schema/GLM52 acceptance suite: `130 passed`, with only
  the existing SWIG deprecation warnings;
- focused `compileall` and `git diff --check`: exit 0;
- real pinned plan: 225 groups, no missing groups, layers 3-77 only, 37,800
  bundles, 53 cross-shard bundles, and all 61 main-model source shards present;
- decoded layer 10 / expert 0 SHA-256: gate
  `19638efebbca55205804574337d71a87dd0217ef6322e6a399444ecf0d79bfbd`, up
  `b673bd500f468b16eab2d04eefab9deca16191aef3bc63f2a18d72a0c6a6df13`, down
  `5e9ce06d0db45f464464f10fcab8bd424ee81f58115e9bea693c87bea3f59dc6`;
- decoded cross-shard layer 29 / expert 58 gate SHA-256:
  `0119f166fe1fb59f47faa5b0580538b7030d1cbaa691206a7f98568f4ec7a117`.

This closes the decoder prerequisite; by itself it does not prove materialization
or runtime acceptance. Bounded materialization evidence is recorded below. The
FP4 source remains the teacher/reference ceiling; exact W4A4 forward parity would
additionally require the source activation-quantization behavior.

### Source and source-payload audit checkpoint complete

The source boundary is now profile-bound and fail-loud. `glm52-source-audit` requires
local config/index/profile inputs plus the exact profile/model/revision identity and
the pinned raw config/index SHA-256 values. The profile file itself participates in
the build-step key. Unsupported encodings, incomplete group plans, missing
packed-weight or scale companions, unexpected routed tensors, and pinned
shard/cross-shard count drift make the audit non-ready.
`glm52-source-payload-audit` first requires that full source contract to pass, then
reads each required shard header once and validates all selected packed weights,
block scales, and global scales without decoding model data. Missing files,
directories, unreadable headers, wrong names/dtypes/shapes/byte lengths, overlapping
or gapped layouts, invalid offsets, truncated payload extents, or over-large bounded
selections block materialization. Bounded readiness is reported separately from
full-payload readiness, and the live payload op is noncacheable so resume cannot hide
a removed or changed shard; executing it also force-runs downstream consumers.
Layer 78 is inventoried as excluded MTP evidence and never enters a conversion
group.

Fresh pinned-snapshot evidence on 2026-07-09:

- source audit: `154433` indexed tensors across `63` snapshot shards; exact profile,
  model ID, revision, raw config hash
  `5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b`, raw
  index hash `bb5b4fa9782aea5ffc66f9145d6e630f1045d385c30437c531bebe422c075f3f`,
  and native ModelOpt NVFP4 contract; `225/225` groups over
  layers 3-77; `37800/37800` bundles; `113400/113400` required bundle members;
  `61` required main-model shards; `53` cross-shard bundles; `37800` separately
  reported activation-scale tensors; and `2039` layer-78 tensors explicitly
  excluded;
- local payload audit: all `61/61` required shards present, all `113400` bundle
  member headers and physical extents checked, zero missing/invalid shards or tensor
  headers, all 225 groups ready; `0.98 s` wall time, 230,817,792-byte peak RSS, and
  no weight decode/model load;
- focused source/decoder/planner/profile/build suite: `259 passed`, with only the
  existing SWIG deprecation warnings;
- durable local evidence:
  `artifacts/quality/glm52-reap-source-audit-20260709.json` and
  `artifacts/quality/glm52-reap-source-payload-audit-20260709.json` (plus JSONL
  companions).

This closes backlog wave 1 only. Wave 2 below now supplies the registered,
deterministic, resumable GLM52 materialize-groups op. The high-level GLM52 guard
remains in place.

### Bounded materialize-groups checkpoint complete

`glm52-moe-materialize-groups` is now a real registered build op around the direct
ModelOpt NVFP4 stream converter. The checked-in bounded recipe wires
`source audit -> source payload audit -> materialize groups` with actual step
references, so materialization cannot run from a merely supplied path that did not
pass the pinned profile/config/index/payload contract. The materializer requires an
explicit group selection (or explicit `--all-groups`), enforces the canonical
225-group plan over layers 3-77, rejects extra group artifacts and lineage drift,
and writes group artifacts and the canonical manifest atomically. Normal build
resume preserves complete group checkpoints, revalidates their byte lengths and
SHA-256 values, and reads no source payload for already-valid groups. Explicit
worker parallelism remains opt-in; the default working-set policy is one decoded
50,331,648-byte expert per worker. No dense checkpoint is written.

Fresh pinned-snapshot evidence on 2026-07-09:

- serial E8 layer-10 gate: `52.1376 s`, 1,464,156,160-byte process peak RSS,
  272,500,893 artifact bytes, artifact SHA-256
  `3de6e7fa06dbf4556c5d54227ab6d793a6732a2d273b7b887241f00f562c68c7`,
  and manifest SHA-256
  `8ce34fa0900f6252dbc37deee983efb2db128ca61839c77fea663b4d6a3a7d64`;
- explicit two-worker E8 layer-10 gate: `27.0859 s` wrapper time (`27.24 s`
  external wall), 1,079,558,144-byte process peak RSS, two decoded experts at
  most in flight, and the byte-identical 272,500,893-byte artifact with the same
  `3de6e7fa...562c68c7` SHA-256 as the serial run;
- serial E8P layer-29 gate, including the expert-58 cross-shard bundle over source
  shards 16 and 17: `2163.1296 s`, 2,281,144,320-byte process peak RSS,
  536,742,048 artifact bytes, and artifact SHA-256
  `c2971c8a0fa708ed74fad92ffa57f75986593d8875727029fb066a3de93c8189`;
- serial E8 layers 3-4, all gate/up/down projections: six groups in `306.4544 s`,
  1,466,318,848-byte process peak RSS, 1,635,005,348 artifact bytes, group
  SHA-256 values
  `92a5d87cd08b01383d94f5d668ef0e0aa4757405c5ad1268e5842749e0f00f29`,
  `2f3d4f852b9d4919c4590af4fee19dcca988648176f84bbc93dc3a4a1a1742fd`,
  `6315c3684876b9426b7185e600335506f5409be57795d89892cbb45e903cf898`,
  `06b5b161a1281b7cec08463f5e1e6b2266e46601951b26a2a4e4b34d1c5e5da5`,
  `76959112019bce0f879b36b98e5fef975d76575d908421b39060f800409d62fa`,
  and `c24ec5ad4cbe395090ca1fcb8910a6688516998affdce8178ac11d6a8694ea62`,
  with manifest SHA-256
  `4f72f8ea5e1d8a0f9aafb0982e2509ef40e7dfcc7354392b1046e4395566514b`;
- exact rerun of that six-group selection: `1.2317 s`, zero converted groups,
  six validated existing groups, zero source bundle/member reads, and unchanged
  group and manifest SHA-256 values.

Durable evidence is under
`artifacts/quality/glm52-wave2-materialization-20260709/`, including the per-run
JSON records and append-only `materialization.jsonl`.

This closes bounded backlog wave 2 only. It does not prove the full 225-group
artifact, whole-model bytes/bpw, binding, runtime forward or generation, source-
relative quality, speed acceptance, or RC readiness. Wave 3 below supplies the
strict artifact audit; the high-level guard remains active.

### Strict routed-artifact audit checkpoint complete

`glm52-moe-artifact-audit` is now a real, noncacheable build op and the bounded
recipe is a four-step live chain:

`source audit -> payload audit -> materialize groups -> artifact audit`.

The audit rebuilds the exact selected plan from the pinned profile/config/index
instead of trusting manifest counts. It allowlists the entire artifact tree and
requires exact filenames, tensor names, dtypes, shapes, E8/E8P budget, group size,
embedded codebook values and metadata, scale estimator, decoder/encoding/model/
revision/config/index/profile lineage, file byte lengths and SHA-256 values, and
contiguous physical safetensors extents. Any extra file type, dense routed tensor,
partial, symlink, wrong selection, or layer-78 payload fails loud. Source accounting
likewise validates every indexed tensor descriptor and the complete physical extent
of all 63 shards before emitting retained-weight or projection figures; the index's
`metadata.total_size` is not used as physical truth.

Fresh pinned evidence on 2026-07-09:

- first real four-step `keep build` under the final content-addressed profile completed
  in `115.05 s` with 1,461,125,120-byte peak RSS and zero swaps. The two-group E8
  materializer itself took `110.3719 s`,
  wrote 545,001,786 physical artifact bytes, and the audit measured exactly
  `1.03125` semantic routed bpw (`1.03125834465` including files/headers). The
  canonical manifest SHA-256 is
  `7440d7cea6d1b6f5bfab281f25f5679afe140c9abf4be4755c20f56ed75f6c24`;
- an immediate normal rerun completed the four-step chain in `4.66 s` with
  325,599,232-byte peak RSS and zero swaps. Materialization took `0.89445 s`,
  converted zero groups, validated two existing groups, and read zero source
  bundles/members. The audit consumed both append-only run records and reported
  `resume_verified=true`, `byte_identity_verified=true`, and stable group-set
  SHA-256 `8bacd1177f1dca3395855487a35670021660b7f9df161f8e3d5ead14a08fccbe`;
- the six-group layers-3-4 artifact passes with two complete layer trios,
  12,683,575,296 actual routed parameters, 1,634,992,128 code+scale bytes,
  exactly `1.03125` routed bpw, 1,635,005,348 physical group-file bytes, and
  resume/byte-identity proof. Group-set SHA-256 is
  `3f2479608cea3cab3cbe221770b6853a1adabe583158fc1754a72d5068038cf6`;
- the cross-shard layer-29 E8P artifact passes at exactly `2.03125` routed bpw
  (`2.03125835600` physical), 536,742,048 physical bytes, and verified resume;
- asking the auditor to accept the E8 layer-10 artifact as E8P exits nonzero only
  after writing failure evidence: `manifest code bits budget must be 16, found 8`.

The same header pass establishes exact main-model accounting: 475,634,073,600
routed parameters, 18,560,731,704 retained non-routed parameters in 1,194 tensors,
6,630,491,048 excluded layer-78 parameters, and a 494,194,805,304-parameter main
denominator. Retained non-routed payload is 37,121,488,608 bytes. A uniform full-E8
layout projects 61,312,435,200 routed tensor bytes including repeated codebooks,
so the accepted source-precision composite projection is 98,433,923,808 bytes /
1.5934433 bpw. At this Wave 3 checkpoint actual whole-model artifact bytes/bpw
remained `null`; the lean package checkpoint below now makes the non-routed
component actual, but the full routed component is still missing.

Durable evidence is under
`artifacts/quality/glm52-wave3-artifact-audit-20260709/` and the four-step build
root under `artifacts/build/glm52_reap_504b_materialization_probe_20260709/`.
This closes bounded routed-artifact audit only.

### Lean non-routed package checkpoint complete

`glm52-non-vq-pack` is now a registered, resumable artifact op in the checked-in
recipe (originally its fifth and now followed by the layer-forward sixth step).
Production mode requires the canonical profile, raw config/index
hashes, and exact 63-shard Hugging Face LFS blob-identity inventory. It validates
every source shard's indexed header and physical extent, excludes the complete
`.mlp.experts.` source namespace, pre-existing `.mlp.switch_mlp.` runtime artifacts,
and all layers at or above 78, then preserves every retained BF16/F32 tensor name,
shape, dtype, and payload byte exactly. The writer uses at most a 1 MiB copy buffer;
it never calls MLX or NumPy tensor materialization. Lexical greedy sharding, canonical
safetensors headers/index/manifest, atomic publication, per-tensor and per-shard
hashes, strict-tree audit, and source-relative byte comparison make both first write
and interrupted resume fail loud.

Fresh pinned evidence on 2026-07-09:

- the actual package contains 1,194 tensors and 18,560,731,704 parameters:
  1,119 BF16 tensors plus 75 F32 router-correction tensors;
- nine shards contain exactly 37,121,488,608 tensor-payload bytes. The complete
  artifact tree is 37,122,403,011 bytes, so headers plus canonical index and manifest
  add exactly 914,403 bytes;
- manifest SHA-256 is
  `5113750fdaf009b2174a2772dd2e3bccf490cfff354f082e5e0ad8af8771f8c8`,
  package-set SHA-256 is
  `2719f13a66313b5b8acc4c053914cdbfa10fdbf103a62b924495eecddbcea2ed`,
  and package-index SHA-256 is
  `ff7def155c88006cda458344b5cb8e8a20303049d20405cd4e9b28e7885df097`;
- the source LFS identity inventory SHA-256 is
  `ace08e87dcbce3a22249e54196a27c0045992f8d5b8ca07f342899fa7a53fc8d`;
- the first five-step build took `134.13 s`, peaked at 323,944,448 bytes, and had
  zero swaps. An immediate normal rerun took `205.30 s`, peaked at 328,138,752
  bytes, rewrote zero shards, source-verified/reused all nine, and retained every
  hash. The rerun performs two complete source-relative audits and is therefore an
  integrity proof, not a repack speed measurement;
- focused pack/build/resume verification passed, and independent review found no
  remaining Critical or Important issue.

The accepted 98,433,923,808-byte / 1.5934433-bpw target now has an actual audited
37,121,488,608-byte non-routed component. Whole-model bytes/bpw remain a projection
until all 225 routed E8 groups exist and pass audit. The next ordered stage is real
IndexShare/tokenizer/full-bind preflight; full materialization, whole-model runtime,
generation, quality, speed, and RC acceptance remain open.

### Bounded layer bind/forward checkpoint complete

`glm52-layer-forward-probe` is now a registered noncacheable evidence op, and the
checked low-level recipe is a six-step source-audit -> source-payload-audit ->
materialize-groups -> non-VQ-pack -> artifact-audit -> layer-forward-probe chain.
The probe authenticates the profile, model/revision, config/index, 63-blob source
inventory, production non-VQ evidence, selected package shard bytes, and bounded
routed audit before constructing or mutating the runtime layer. It then validates
all selected non-VQ arrays before one strict update and binds the complete E8
gate/up/down trio.

Fresh pinned evidence on 2026-07-09:

- layer 3 strictly loaded 15 non-VQ runtime parameters with zero missing or
  unexpected tensors. The source `model.layers.3.self_attn.kv_b_proj.weight`
  was split into the runtime `embed_q` and `unembed_out` surfaces;
- the VQ trio bound 168 experts at top-k 8 with consistent 8-bit, group-size-512
  projection metadata. No dense routed parameter was present;
- deterministic BF16 input shape `[1, 1, 6144]` produced a finite BF16 output
  shape `[1, 1, 6144]`. The final direct durable probe spent `1.839849500 s`
  validating evidence, `0.091113000 s` binding non-VQ, `0.005727875 s` binding
  VQ, and `0.016422291 s` in forward (`0.157069291 s` measured runtime), with
  1,254,490,792 MLX peak bytes, 2,006,499,328 RSS bytes, zero pageout delta, and
  zero swapout delta;
- the only selected non-VQ shard was authenticated as
  `model-00004-of-00009.safetensors`, 4,132,565,384 bytes, SHA-256
  `68c79e4d0bb50301a2277e7c3f3867a35a5134c182e1badfbe0d764265ac8767`;
- before the first six-step build, `README.md`, `tokenizer_config.json`,
  `generation_config.json`, and `chat_template.jinja` were fetched into the same
  pinned snapshot. Because the build hashes `source_dir` as the complete
  directory, that expanded metadata surface correctly produced the new non-VQ
  key `a678bc3c` and a fresh package publication. The `256.59 s` build retained
  the exact package-set, manifest, and index hashes above. This was not caused
  merely by adding the downstream graph step;
- the unchanged six-step rerun took `213.45 s`; non-VQ wrote zero shards,
  source-verified/reused all nine, reported resume green, and retained all
  package hashes. External maximum RSS was 2,005,729,280 bytes and swapouts were
  zero;
- staged independent reviews found four Important evidence-integrity issues:
  three in the first probe review and one in the final whole-wave review. All
  four were fixed under regressions, focused verification was rerun, and final
  re-review approved the wave with no remaining Critical or Important finding.

Durable direct evidence is
`artifacts/quality/glm52-wave4-layer3-forward-20260709.json`. Checked-build
evidence is under
`artifacts/build/glm52_reap_504b_materialization_probe_20260709/steps/layer_forward_probe-7a9c4ae1/evidence/evidence_json.json`,
with the republished package and evidence under sibling step
`non_vq_pack-a678bc3c/`. The routed authority remains
`artifacts/quality/glm52-wave3-artifact-audit-20260709/e8-layers3-4-audit.json`.

This layer-forward checkpoint is deliberately bounded:
`full_routed_artifact_ready=false`, `whole_model_runtime_proven=false`,
`generation_proven=false`, `tokenizer_proven=false`,
`long_context_indexshare_proven=false`, and `speed_claim=false`. Layer 3 uses the
shared indexer with one token and no previous top-k input, so this specific probe
does not prove sparse IndexShare selection. At the time of Wave 4 the snapshot
still lacked `tokenizer.json`; the runtime checkpoint below supersedes that
tokenizer/IndexShare readiness statement, but does not widen this probe into
whole-model evidence. The accepted 98,433,923,808-byte / 1.5934433-bpw whole-main
target remains projected until all 225 routed groups exist and pass audit.

### Runtime capability checkpoint partial; full-bind preflight blocked

The pinned runtime-preflight wave adds exact, durable capability evidence without
claiming a production model run:

- `artifacts/quality/glm52-indexshare-runtime-20260709.json` authenticates the
  pinned config/index hashes and `glm52-reap-504b-v2` profile, then proves the
  exact 21-full/57-shared production schedule and 105 main plus five layer-78 MTP
  indexer tensors. A deterministic tiny full -> shared chain at `index_topk=2`
  reuses prefill/decode top-k indices exactly, advances full/shared cache offsets
  from `[3,3]`/`[3]` to `[4,4]`/`[4]`, stays finite, retains no dense routed
  weights, and raises the missing-top-k guard before cache mutation (`0 -> 0`).
  `production_long_context_proven`, `whole_model_runtime_proven`,
  `full_model_bind_proven`, and `generation_proven` remain false. Evidence
  SHA-256 is
  `94785cf0884ce5ac955a09b535c0b2ef00ad3cb742da7769bf504a4c9143ad25`.
- `artifacts/quality/glm52-tokenizer-readiness-20260709.json` proves the exact
  five-file pinned tokenizer/chat payload, including the 20,217,442-byte
  `tokenizer.json`, with `local_files_only=true`, `trust_remote_code=false`,
  complete EOS handling, exact prompt round-trip, and thinking-on/off chat
  renders. It records `tokenizer_payload_ready=true` and `production_ready=true`,
  while whole-model bind and generation remain false. Evidence SHA-256 is
  `27a372e67a59b264c73939311097f9f273971c7401c4f74763e3dfb4c2220f6b`.
- `artifacts/quality/glm52-synthetic-generation-contract-20260709.json` proves
  only `whole_model_scope=tiny_fixture`: a two-layer in-memory VQ model emits
  finite `[1,3,32]` direct logits, then exactly two repeatable greedy
  `generate_step` tokens with finite `[2,32]` log-probabilities and explicit
  full/shared cache offsets `[4,4]`/`[4]` then `[5,5]`/`[5]`. It records
  `synthetic_runtime_contract=true`, `production_artifact_used=false`,
  `tokenizer_used=false`, and `production_generation_proven=false`. Evidence
  SHA-256 is
  `baa9c8214553c6af391c6a671a5b0185b8d7eb1746c3e6c52586f512536662a9`.
- `artifacts/quality/glm52-wave5-full-bind-preflight-20260709.json` is header-only:
  `tensor_payloads_read=false`, `payload_hashes_verified=false`, and
  `full_model_constructed=false`. It validates all 1,194 non-VQ source tensors,
  1,272 runtime targets, and the 78 `kv_b_proj` sources mapping to 156 runtime
  targets, but sees only six of 225 routed groups. It therefore records
  `glm52_full_bind_preflight_blocked`, `preflight_pass=false`, and exactly 219
  missing groups; production binding and generation remain false. Evidence
  SHA-256 is
  `457d037dd3343c9276837261c781837753cd9af18c9bfaf0410c4ff89dc74eb3`.

The checked four-step runtime recipe validates and dry-runs as
`indexshare_runtime -> tokenizer_readiness -> synthetic_generation ->
full_bind_preflight`. All four ops are noncacheable. The checked normal build
completes the first three ready probes, then the final preflight exits `2`;
overall build exit is `1`, with blocker JSON preserved. `/usr/bin/time -l env
UV_CACHE_DIR=/tmp/keep-uv-cache uv run keep build
recipes/glm52_reap_504b_runtime_preflight_20260709.yaml` took `4.76 s` real,
reached 606,306,304-byte maximum RSS, and recorded zero swaps. This is the
intended fail-loud behavior, not a successful runtime build. Fresh recipe
verification was `ok: ... (4 steps)`, dry-run exit 0, and `2 passed` for
`tests/test_glm52_runtime_build_integration.py`.

The accepted 98,433,923,808-byte / 1.5934433-bpw composite remains a projection:
37,121,488,608 non-routed payload bytes are actual, but the 61,312,435,200 routed
bytes are projected until the remaining 219 groups are materialized and all 225
pass audit. The next runtime gate is therefore full routed materialization and
audit, followed by a rerun of the header preflight, production binding, and a
production token-generation smoke.

### Frozen family policy and prompt authority complete

Before any recovery tuning or candidate-quality row exists, KEEP now freezes a
GLM52-specific source-relative contract. The checked
`recipes/glm52_reap_504b_eval_contract_20260709.yaml` recipe revalidates the
pinned tokenizer, writes the family policy, and then emits the exact prompt pack.
Validation is `ok: ... (3 steps)`; dry-run is green; the live build took `6.33 s`,
reached 609,812,480-byte maximum RSS, and recorded zero swaps. Focused tests are
`15 passed`.

The prompt authority has 22 rows per report, selection, and holdout split (66
total). Every split has eight route, seven math, and seven instruction rows. Only
selection is tuning-eligible and holdout tuning is forbidden. The prompt-text
SHA-256 is
`3c39f6bb133907cf2fd93260a3cc548c0b186a264230568678cd076f1638eb6a`.
The policy requires full-vocabulary teacher and candidate logits and rejects
compact/top-k-renormalized KLD. Frozen gates are mean KLD `<=0.30`, p999 KLD
`<=3.0`, global top1 `>=0.85`, every-domain top1 `>=0.80`, and mean PPL ratio
`<=1.05`. Benchmark gates require three clean same-machine repetitions of
`prefill_1k` and `decode_128` against
`same_machine_pinned_fp4_source_streaming_control`, with ratio `<=1.15` and zero
pageout/swapout deltas. Policy contract SHA-256 is
`57ae812222de79555774296129e7110750f492794b1fe21bac91ed5cc986f206`.

The teacher identity is explicitly deterministic dequantization of the pinned
ModelOpt NVFP4 weights. Activation quantization is not emulated, exact W4A4
runtime parity is not claimed, and this is not a BF16 teacher. Durable evidence is
`artifacts/quality/glm52-family-policy-20260709-v2.json` and
`artifacts/quality/glm52-family-eval-prompts-20260709-v2.json`. Production
validation recomputes the complete policy body, authenticates the five exact
tokenizer files and readiness claims, pins the corpus hash, binds final provenance
into the prompt-pack digest, and rejects output paths that alias any input or the
tokenizer snapshot. These artifacts freeze
contracts only: they do not contain a teacher cache or candidate metrics and do
not prove quality, route/math parity, resident speed, or a family-gate pass. The
high-level guard remains active; its message now names the real remaining gates.

The two `keep get` raw-HF-id bugs are fixed and now have real isolated-cache
proof. With `HF_HUB_OFFLINE=1`, the raw ID resolves symbolic `main` to immutable
revision `6c9241aa05fb243a0edb7c804c213ec1cf5c920d`; a reuse-only complete view
reports 63/63 shards, 308,829,060,264 bytes, `ready: yes`, and exit 0. A second
view exposing the same config/index but only one shard reports 1/63,
`ready: no`, and exit 1. No payload was copied, downloaded, or loaded. Exact
observation commands, cache-view construction commands, manifests, and hashes
are recorded in
`artifacts/quality/glm52-raw-hf-get-check-only-20260710.json` (SHA-256
`82ac2559e013952313ef182fd21c2940458ae05e666c852806cf015f0b5f2424`). This closes raw-ID
cache discovery and check-only exit semantics, not the still-missing end-to-end
high-level build.

### Authenticated teacher metadata and fail-loud family gate complete

KEEP now has a real `glm52-family-eval-teacher-metadata` build operation and a
terminal `glm52-family-gate-check`. The metadata operation does not generate or
pretend to contain a teacher cache. It revalidates the pinned source index and
all 61 required safetensors headers, binds the five exact tokenizer files plus
the frozen policy/prompt identities, and emits 66 metadata rows. Its explicit
negative claims are `tensor_payloads_read=false`, `full_model_constructed=false`,
`candidate_artifact_used=false`, `teacher_cache_payload_present=false`,
`teacher_cache_ready=false`, and `teacher_logits_generated=false`. The intended
future cache remains exact full-vocabulary logits from deterministic ModelOpt
NVFP4 weight dequantization; activation quantization is not emulated and neither
BF16-teacher nor exact-W4A4-runtime parity is claimed.

Teacher preparation cross-authenticates the prompt pack's policy and tokenizer
readiness SHA-256 values and paths against the supplied files, not merely the
prompt pack's freely recomputable final digest. It also validates the prompt
content contract, exact token IDs, source/payload audits, and JSONL contents.
Only the known additive source-audit schema evolution—legacy omission of
`requested_groups` versus the current empty list for an unfiltered full
audit—is normalized; every other field remains exact. Rehashed provenance
changes, altered payload counts, input aliases, source-tree outputs, and
summary/metadata-output aliases all fail before destructive writes.

The terminal family gate accepts evidence files, not boolean override flags. It
distinguishes malformed provenance (exit `1`) from valid but incomplete release
evidence (exit `2`). Gate schema v1 records `release_pass_enabled=false` and
`raw_release_evidence_validators_ready=false`; it intentionally has no green
path until raw release-evidence validators and common artifact identity are implemented.
Optional eval, benchmark, and production summaries are therefore rejected
rather than trusted; an all-true rehashed payload cannot create a pass, and
regating cannot promote a terminal rc-2 step to completed. No-run or failed-run
regates return 1, and accepted-nonzero/passing-gate contradictions become
terminal failures rather than completed steps. Current durable evidence
`artifacts/quality/glm52-family-gate-preflight-20260709.json` is intentionally
blocked: 6/225 routed groups are present and 219 are missing. Static IndexShare,
synthetic generation, frozen policy/prompt, teacher-source metadata, and
no-dense-routed-expert checks pass. The six release blockers are
`dequantized_source_teacher_cache_payload`, `full_225_group_artifact`,
`production_model_bind_and_generation`,
`full_vocabulary_source_relative_family_eval`, `route_math_diagnostics`, and
`same_machine_pinned_fp4_benchmark`. The evidence must not be described as a
family-gate pass.

Direct local evidence is:

- `artifacts/quality/glm52-teacher-metadata-20260709.json`, SHA-256
  `621a013eb37f617409ec568340976e769610b3b6c8cae346639762d816a7fd7a`;
- `artifacts/quality/glm52-wave7-teacher-metadata-20260709/glm52_teacher_source_metadata.jsonl`,
  SHA-256 `b43a2f44938679b1e64541a0367333d7993eacfc3890a4f9aed5fbfca894a683`;
- `artifacts/quality/glm52-family-gate-preflight-20260709.json`, SHA-256
  `371441fea4f37cee6bb66b45cb3a193b4c8e470ca113edf7f44fc0f4324363ab`.

The checked
`recipes/glm52_reap_504b_family_preflight_20260709.yaml` recipe is
`source_audit -> source_payload_audit -> indexshare_runtime ->
tokenizer_readiness -> synthetic_generation -> full_bind_evidence ->
family_policy -> eval_prompts -> teacher_metadata -> family_gate`. The
`full_bind_evidence` step recomputes the honest 6-present/219-missing report
against strict hashed non-routed and routed artifact inputs. Validation reports
ten steps and the strict hashed dry-run exits zero; `--no-hash` is rejected.
The normal build completes the first nine steps, preserves the final blocker
JSON, then records subprocess exit `2` as structured `gate_failed` with all six
input hashes; overall build exit is `1`. It took `14.68 s` real, reached
609,583,104-byte maximum RSS, and recorded zero swaps. The path-bound build gate
evidence SHA-256 is
`e3ec9e51a92ea1a0bf5af8dd5c59e7f813712d1800508c4d8887a3606d80caa6`.
Focused contract/build verification is `173 passed`; the broader GLM52 and
high-level suite is `215 passed`; compileall and `git diff --check` pass. Ruff is
not installed in this checkout.

Full E8 materialization has started separately with the resumable, safe
single-worker command (`--all-groups --code-bits 8 --expert-workers 1
--skip-existing`). At the 22:31 PDT checkpoint it was healthy at 61/225 group
files and 15 GB under
`artifacts/quality/glm52-wave6-full-materialization-20260709/e8-full-w1`, with
about 97.9% CPU and 0.94 GiB RSS. The accepted 98,433,923,808-byte / 1.5934433-bpw
target therefore remains projected until this run finishes, the strict
225-group audit passes, and production bind/generation evidence exists.

### Full composite generation and warm-residency checkpoint complete

This checkpoint supersedes the in-progress materialization and projected-byte
statements immediately above. The safe single-worker E8 run completed all 225
groups for layers 3--77, wrote no dense routed checkpoint and no layer-78 MTP
payload, and then passed a deterministic zero-write resume. The routed manifest
SHA-256 is
`ba1d3135ef8901f1a69ead28b5f9d330ef40d015ac31fdb4d2dcfe678c41f0a4`;
the strict composite audit records a 61,312,435,200-byte routed tensor payload,
37,121,488,608 non-routed tensor bytes, and the accepted exact whole-main payload
of 98,433,923,808 bytes / `1.593443277857996` bpw. The common authenticated
artifact identity is
`ef9d2e49d4a9d113a13d8b8e6c6ce7ebe60a7e9c7fb7b1b7784357d3efee5067`.
`artifacts/quality/glm52-wave6-full-artifact-audit-20260709.json` and
`artifacts/quality/glm52-wave6-full-bind-preflight-20260709.json` both pass; the
preflight sees all 225 groups, all 1,272 runtime non-VQ targets, nine non-VQ
files, zero dense routed parameters, and zero layer-78 groups.

Production generation now authenticates the complete source/artifact/evidence
set, constructs and strictly binds the complete model, eagerly realizes every
parameter, clears allocator cache to an asserted exact zero, and requires a
15-second zero-pageout/zero-swapout quiet window. It records an upstream
`mlx_lm.generate_step` sacrificial warmup, drains its one-token generator through
cleanup, synchronizes MLX-LM's exact generation stream, clears cache to zero
again, and requires a second quiet window. The measured token then mirrors
MLX-LM's bounded prefill without its unnecessary generated-token lookahead:
prompt tokens 1--10 populate and materialize the explicit cache, prefill
temporaries are cleared, and token 11 drives exactly one final greedy decode.
The measured token ID is `785`, decoded as `The`, with finite full-vocabulary
log-probabilities and an exact match to the upstream warmup token.

Durable evidence is
`artifacts/quality/glm52-wave6-production-generation-bounded-prefill-warm-resident-20260710.json`,
file SHA-256
`758b5bbebbb365cdf69998a034cf3791a6210d2723d5bec6c022336eceadcc82`.
The measured post-warmup interval starts at 98,433,923,832 active MLX bytes and
zero allocator-cache bytes, peaks at 103,257,924,340 bytes, ends with
5,489,963,346 cache bytes, and records exact integer pageout/swapout deltas
`0/0`. It therefore records `warm_residency_proven=true` under the normal Darwin
`iogpu.wired_limit_mb=0` policy with no MLX wired-limit environment override.
Cold first-touch and the sacrificial upstream warmup are still dirty on this
loaded 128 GB host, so `cold_residency_memory_clean=false`,
`generation_warmup_memory_clean=false`, and
`production_residency_proven=false` remain fail-loud. This is a real complete
bind/generation and clean warm steady-state proof, not a clean cold-start or
publication-speed claim.

The exact child command was run while the caller held the repository's
exclusive `.keep-heavy-job.lock`:

```bash
env -u GLM_MLX_WIRED_LIMIT_GB -u GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB \
  .venv/bin/python benchmarks/probe_glm52_production_generation.py \
  --profile-path models/glm52-reap-504b-v2.yaml \
  --config-path "$HOME/.cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2/snapshots/6c9241aa05fb243a0edb7c804c213ec1cf5c920d/config.json" \
  --source-index-path "$HOME/.cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2/snapshots/6c9241aa05fb243a0edb7c804c213ec1cf5c920d/model.safetensors.index.json" \
  --tokenizer-dir "$HOME/.cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2/snapshots/6c9241aa05fb243a0edb7c804c213ec1cf5c920d" \
  --tokenizer-readiness-json artifacts/quality/glm52-tokenizer-readiness-20260709.json \
  --family-policy-json artifacts/quality/glm52-family-policy-20260709-v2.json \
  --non-vq-artifact-dir artifacts/build/glm52_reap_504b_materialization_probe_20260709/steps/non_vq_pack-a678bc3c/out \
  --non-vq-evidence-json artifacts/build/glm52_reap_504b_materialization_probe_20260709/steps/non_vq_pack-a678bc3c/evidence/evidence_json.json \
  --routed-artifact-dir artifacts/quality/glm52-wave6-full-materialization-20260709/e8-full-w1 \
  --composite-audit-json artifacts/quality/glm52-wave6-full-artifact-audit-20260709.json \
  --materialization-runs-jsonl artifacts/quality/glm52-wave6-full-materialization-20260709/materialization.jsonl \
  --full-bind-preflight-json artifacts/quality/glm52-wave6-full-bind-preflight-20260709.json \
  --model-id 0xSero/glm-5.2-reap-504B-v2 \
  --revision 6c9241aa05fb243a0edb7c804c213ec1cf5c920d \
  --prompt "The capital of France is" \
  --output-json artifacts/quality/glm52-wave6-production-generation-bounded-prefill-warm-resident-20260710.json
```

Focused GLM52 plus memory-metrics verification is `263 passed`; compileall and
`git diff --check` pass. Independent review found and closed stream-cleanup,
phase-accounting, peak-epoch, and dirty-warmup claim gaps, then approved the
bounded-prefill path with no remaining P0--P2 finding. Quality caches and
source-relative metrics, route/math diagnostics, same-machine speed rows, raw
family-gate evidence validators, the high-level recipe, chat/report/model-card
surfaces, and a clean cold-start residency claim remain open. The high-level
GLM52 guard must remain active.

### Intentional interruption and bounded restart checkpoint

The full 225-group run above proves completed-artifact zero-group-payload-write
idempotence, not an actual interrupted attempt. A separate current-REAP proof
now exercises that missing case without touching the accepted artifact. A fresh
two-group E8 run (`10:gate_proj`, `10:up_proj`) used the same pinned ModelOpt
source, single-worker policy, `--skip-existing`, and repository heavy-job lock.
After `10:gate_proj` was atomically published, PTY `Ctrl-C`/`SIGINT` interrupted
the process during the second group; the process exited `130` with
`KeyboardInterrupt`. The stopped directory contained exactly the gate group,
with SHA-256
`3de6e7fa06dbf4556c5d54227ab6d793a6732a2d273b7b887241f00f562c68c7`,
and contained no up group, canonical manifest, or partial file.

Restarting with identical artifact-affecting arguments validated and retained
that exact inode/mtime checkpoint, converted only `10:up_proj`, and reported one
existing plus one converted group, 168 source bundles / 504 bundle members read,
51.494747 seconds, and 1,464,041,472-byte peak RSS. The recovered up-group hash
is `a397ff9ad6d7a2baf7a55ac4a44dd8bff258338d3aa9ae9673ae65fe34d7c863`;
both recovered hashes exactly equal the corresponding groups in the full
production artifact. A third zero-group-payload-write run reported two existing
groups, zero source reads, the same
`b552d0821207591ee7497731f4f8db6c6c255cb25c749018072f13224024eb2d`
manifest, 0.871923 seconds, and 188,628,992-byte peak RSS. The materializer still
rewrites that canonical manifest atomically by design. The strict bounded audit
passes and reports `byte_identity_verified=true` with no blockers.

The audit's older `resume_verified` field is intentionally still false: its
contract requires both an all-converted record and an all-existing record, while
this proof is the stronger mixed recovery case. The durable packet is
`artifacts/quality/glm52-interrupted-resume-proof-20260710.json`, SHA-256
`cecc236576529edfd2f1715cb4354ac79bc2cfe0dc07cffcd218379a942b3b6e`.
Literal argv arrays, the PTY trigger, and the audit command are in
`artifacts/quality/glm52-wave7-interrupted-resume-20260710/commands.json`; the
checked-in PTY transcript is `interrupted-attempt-terminal.txt` in that same
directory. This demonstrates interruption/retry for the actual GLM52
materializer and complements the full-225 group-payload-idempotence proof. It
does not yet demonstrate an interrupted end-to-end high-level `keep build`;
that product-level acceptance remains open until the guarded chain is real.

The exact historical reproduction protocol is below. For another fresh
interruption proof, change `OUT` to a new immutable directory rather than
deleting or overwriting this evidence.

```zsh
SNAPSHOT="$HOME/.cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2/snapshots/6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
OUT="artifacts/quality/glm52-wave7-interrupted-resume-20260710"
BASE=(
  env UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python
  benchmarks/materialize_glm52_reap_groups.py
  --profile-path models/glm52-reap-504b-v2.yaml
  --model-id 0xSero/glm-5.2-reap-504B-v2
  --revision 6c9241aa05fb243a0edb7c804c213ec1cf5c920d
  --config-path "$SNAPSHOT/config.json"
  --index-path "$SNAPSHOT/model.safetensors.index.json"
  --source-dir "$SNAPSHOT"
  --output-dir "$OUT/e8-l10-gate-up"
  --groups 10:gate_proj,10:up_proj
  --code-bits 8 --group-size 512 --scale-estimator max_abs
  --expert-workers 1 --skip-existing
)

# Fresh attempt: after layer-00010-gate_proj.safetensors appears, send Ctrl-C.
/usr/bin/lockf -kn .keep-heavy-job.lock "${BASE[@]}"

# Resume the same artifact-affecting command.
/usr/bin/lockf -kn .keep-heavy-job.lock "${BASE[@]}" \
  --output-json "$OUT/resume-completion.json" \
  --append-jsonl "$OUT/materialization.jsonl"

# Verify zero group conversions and zero group-payload writes.
/usr/bin/lockf -kn .keep-heavy-job.lock "${BASE[@]}" \
  --output-json "$OUT/zero-write-verification.json" \
  --append-jsonl "$OUT/materialization.jsonl"

env UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python \
  benchmarks/audit_glm52_reap_materialization.py \
  --artifact-dir "$OUT/e8-l10-gate-up" --source-dir "$SNAPSHOT" \
  --profile-path models/glm52-reap-504b-v2.yaml \
  --config-path "$SNAPSHOT/config.json" \
  --index-path "$SNAPSHOT/model.safetensors.index.json" \
  --model-id 0xSero/glm-5.2-reap-504B-v2 \
  --revision 6c9241aa05fb243a0edb7c804c213ec1cf5c920d \
  --groups 10:gate_proj,10:up_proj \
  --code-bits 8 --group-size 512 --scale-estimator max_abs \
  --materialization-runs-jsonl "$OUT/materialization.jsonl" \
  --output-json "$OUT/bounded-artifact-audit.json"
```

### Authenticated schema-v2 raw family-gate checkpoint blocked

The family-gate JSON authority loader now rejects duplicate object names at
every nesting depth and rejects the non-standard `NaN`, `Infinity`, and
`-Infinity` constants before validation. It still reads each authority file
once, parses those captured bytes, and computes the recorded file SHA-256 from
the same bytes. After that hardening, this exact command returned the required
exit `2` and emitted a valid blocked checkpoint:

```bash
set +e
UV_CACHE_DIR=/tmp/keep-uv-cache uv run python \
  benchmarks/check_glm52_family_gate.py \
  --family-policy-json artifacts/quality/glm52-family-policy-20260709-v2.json \
  --eval-prompt-pack-json artifacts/quality/glm52-family-eval-prompts-20260709-v2.json \
  --teacher-metadata-json artifacts/quality/glm52-teacher-metadata-20260709.json \
  --full-bind-preflight-json artifacts/quality/glm52-wave6-full-bind-preflight-20260709.json \
  --indexshare-runtime-json artifacts/quality/glm52-indexshare-runtime-20260709.json \
  --synthetic-generation-json artifacts/quality/glm52-synthetic-generation-contract-20260709.json \
  --non-vq-evidence-json artifacts/build/glm52_reap_504b_materialization_probe_20260709/steps/non_vq_pack-a678bc3c/evidence/evidence_json.json \
  --composite-artifact-audit-json artifacts/quality/glm52-wave6-full-artifact-audit-20260709.json \
  --production-generation-json artifacts/quality/glm52-wave6-production-generation-bounded-prefill-warm-resident-20260710.json \
  --output-json artifacts/quality/glm52-family-gate-raw-evidence-20260710.json
test "$?" -eq 2
```

The durable output is
`artifacts/quality/glm52-family-gate-raw-evidence-20260710.json`, file SHA-256
`5f8918ed1c7a554bc15ca747b6cbc643df06f3386126b10be3d4ca1c1c3ea9ad`.
Independent field assertions confirm gate schema v2, blocked status, the exact
common artifact identity
`ef9d2e49d4a9d113a13d8b8e6c6ce7ebe60a7e9c7fb7b1b7784357d3efee5067`,
and complete `225/225` routed coverage with zero missing groups. The identity
authenticates the accepted exact whole-main payload of 98,433,923,808 bytes /
`1.593443277857996` bpw.

The promoted checks are exactly
`full_routed_coverage_ready=true`,
`full_225_group_artifact_ready=true`,
`accepted_artifact_bytes_and_bpw_ready=true`,
`production_binding_and_generation_ready=true`, and
`no_dense_routed_experts=true`. Granular raw-validator readiness is true only
for `artifact` and `production`; `family_eval` and `family_benchmark` remain
false. The corresponding unpromoted checks remain
`teacher_cache_full_vocabulary_ready=false`,
`full_vocabulary_source_relative_family_eval_ready=false`,
`route_math_diagnostics_ready=false`, and
`same_machine_pinned_fp4_benchmark_ready=false`.

Exactly four blockers remain:

- `dequantized_source_teacher_cache_payload`;
- `full_vocabulary_source_relative_family_eval`;
- `route_math_diagnostics`;
- `same_machine_pinned_fp4_benchmark`.

This is not a quality, speed, or release claim:
`teacher_cache_payload_present=false`, `full_vocabulary_eval_proven=false`,
`same_machine_benchmark_proven=false`,
`raw_release_evidence_validators_ready=false`,
`release_pass_enabled=false`, and `family_gate_pass=false`. The authenticated
production input proves a clean warm steady-state interval
(`warm_residency_proven=true`), while cold first-touch and the sacrificial
warmup remain dirty (`cold_residency_memory_clean=false`,
`generation_warmup_memory_clean=false`, and
`production_residency_proven=false`). Therefore the checkpoint does not claim
clean cold-start residency or publication speed. The high-level GLM52 guard
must remain active until the four remaining evidence lanes close. Fresh broad
verification is `154 passed`; focused compileall and `git diff --check` also
exit zero.

### Current-state completion audit and guarded-compiler wording correction

A fresh 2026-07-10 recomposition of the schema-v2 family gate from the exact
current inputs returned exit `2` and was byte-for-byte identical to
`artifacts/quality/glm52-family-gate-raw-evidence-20260710.json` (SHA-256
`5f8918ed1c7a554bc15ca747b6cbc643df06f3386126b10be3d4ca1c1c3ea9ad`).
The full artifact still has exactly 225 routed group files, and the accepted
98,433,923,808-byte / `1.593443277857996`-bpw identity remains unchanged. A
fresh no-model structural regression across the GLM52 artifact, runtime,
production, family-policy, raw-gate, high-level, model-profile, and raw-HF
surfaces passed `453` tests with only the two existing SWIG warnings.

That audit found stale user-facing wording outside this readiness authority.
The high-level compiler guard, `examples/glm52-reap.yaml`, and the GLM52 model
profile still claimed the full 225-group artifact and production
bind/generation were missing. The guard and example now describe the
authenticated current frontier while remaining fail-loud on exactly four
evidence blockers: teacher cache, full-vocabulary family eval, route/math, and
same-machine pinned-FP4 benchmark. Those four keep the family gate
non-release-capable. TDD observed the expected RED failure before that
correction. An attempted profile-note correction was rejected by the broad
suite because the profile YAML is part of the authenticated artifact identity;
the strict family-preflight recipe correctly detected its hash change. That edit
was reverted, preserving profile SHA-256
`ae8b852139ac26e63846b0475064e5b955a00f1ff75f628682a3d8494474114d`.
The stale profile note is therefore frozen-contract drift until a deliberate
identity reissue regenerates all dependent hashes. This is a correctness and
onboarding fix, not new teacher, quality, benchmark, cold-residency, or release
evidence.

### Public objective and Quickstart authority correction

The root `codex-goal-community-wow-keep-rc.md` file is now explicitly a
superseded historical Air record rather than a competing active objective.
`docs/QUICKSTART.md` lists the registered `glm52-reap-504b-v2` profile and uses
only verified current commands: `keep models show`, offline `keep get
--check-only`, and the deliberately guarded high-level compile. It states the
actual 63/63-shard source, 225-group accepted composite, warm-only residency
boundary, and exact four release-evidence blockers. It does not present the
guarded `keep build` or the still-absent `doctor`, `report`, and `chat` surfaces
as working GLM52 release commands.

The Quickstart also identifies the model profile's embedded narrative as frozen
at an earlier authenticated checkpoint. The profile file itself remains
unchanged at SHA-256
`ae8b852139ac26e63846b0475064e5b955a00f1ff75f628682a3d8494474114d`;
the current readiness document, not that frozen prose, is the live status
authority. This improves public truthfulness only. It does not add a runnable
high-level build, teacher cache, eval, route/math evidence, benchmark, clean
cold-start proof, report, chat, doctor, or release pass.

### Strict family-evidence recipe live-build and regate proof

The checked one-step
`recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml` recipe has now been
executed through the real build executor rather than only validated and
exercised through a dry run. The following commands use the recipe's exact nine
pinned evidence authorities and perform no model or tensor-payload work:

```bash
UV_CACHE_DIR=/tmp/keep-uv-cache uv run keep validate \
  recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml
UV_CACHE_DIR=/tmp/keep-uv-cache uv run keep build \
  recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml --dry-run
UV_CACHE_DIR=/tmp/keep-uv-cache uv run keep build \
  recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml
UV_CACHE_DIR=/tmp/keep-uv-cache uv run keep build \
  recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml --regate
```

Validation and the strict hashed dry run each exited `0`; the dry run resolved
step key `48146c10b53be9db`. The live build took 1.38 seconds, reached
105,562,112-byte maximum RSS, recorded zero swaps, and returned the required
overall exit `1`:
the family-gate subprocess returned `2`, so the executor retained its evidence
and wrote terminal `status=gate_failed` rather than mislabeling the blocked gate
as complete. The `family_gate-48146c10/evidence/evidence_json.json` output under
the recipe build root is byte-for-byte identical to the checked family-gate
authority and has SHA-256
`5f8918ed1c7a554bc15ca747b6cbc643df06f3386126b10be3d4ca1c1c3ea9ad`.
The exact commands, top-level exits, input hashes, `/usr/bin/time -l`
observations, and pre/post-regate stat are retained in
`artifacts/quality/glm52-family-evidence-v2-build-regate-observations-20260710.json`
(SHA-256
`c4f324beac207a5e8fe64ff981ba8e9f2382f84bd1c9e92f897a957be59bb806`).

The follow-up regate exited the expected `1`, took 0.74 seconds, reached
101,744,640-byte maximum RSS, recorded zero swaps, appended a `regated` ledger
event, and did not rerun or rewrite the evidence. The retained observation
records identical pre/post inode, modification time, 5,675-byte size, and hash.
The recomposed checkpoint still authenticates common artifact identity
`ef9d2e49d4a9d113a13d8b8e6c6ce7ebe60a7e9c7fb7b1b7784357d3efee5067`,
225/225 routed coverage, 98,433,923,808 accepted tensor-payload bytes, production
binding/generation, and no dense routed experts. It remains honestly blocked on
exactly four items: dequantized-source teacher cache, full-vocabulary
source-relative family eval, route/math diagnostics, and same-machine pinned-FP4
benchmark. This proof advances reproducibility and executor recovery only; it
does not make the high-level build runnable or the release gate passable.

### Approved FP32 source-teacher-cache architecture and synthetic pipeline

On 2026-07-10, the user gave FULL APPROVAL to the FP32 source-teacher-cache
architecture. The committed design authority is
`docs/superpowers/specs/2026-07-10-glm52-source-teacher-cache-design.md` at
`6863815c`.

The synthetic TDD implementation committed at `bf88c7f0` adds the strict
`glm52_teacher_cache_fp32_v1` contract writer and auditor in
`src/mlx_vq/quality/glm52_teacher_cache.py`. It also adds the layer-major
selected-expert streaming source runner in
`src/mlx_vq/models/glm52_source_teacher.py`, including the 78-layer boundary,
TOCTOU-safe chained-checkpoint resume, and the strict contract that source
compute is BF16 while published full-vocabulary logits are F32.

Producer orchestration and build/gate schema v3 committed at `dff8a628` add:

- a locked producer whose prompt-at-a-time shard sink publishes durable work;
- a byte-level non-VQ package audit before model binding;
- the `teacher_cache_artifact` build input kind;
- streamed directory-content hashing;
- the `glm52-teacher-cache-audit` and `glm52-teacher-cache-produce` operations;
- family-gate schema v3, reachable only from a strict raw cache re-audit rather
  than from self-authenticated summary evidence.

The nested heavy-lock deadlock found during adversarial review was fixed at
`52bba157`: the teacher-cache producer operation now acquires and owns the
heavy-job lock directly instead of running beneath the executor-held copy. The
operation version and step keys remain unchanged. Across three adversarial
review rounds, 12 findings were raised and all 12 were closed with pinned
negative tests.

Final host verification passed 429 focused tests. `keep validate` exited `0`.
The strict dry run retained step key `48146c10b53be9db`. The no-cache family
checker exited the required `2` and remained byte-for-byte identical to
`artifacts/quality/glm52-family-gate-raw-evidence-20260710.json`.

These results prove the approved architecture, cache contract, synthetic
runner behavior, orchestration, locking, build integration, and fail-closed
gate transition. They do **not** prove a real teacher execution: no real-source
teacher forward has run, and the cache does not exist yet. The current family
gate therefore remains the schema-v2 blocked authority with exactly four
blockers:

- `dequantized_source_teacher_cache_payload`;
- `full_vocabulary_source_relative_family_eval`;
- `route_math_diagnostics`;
- `same_machine_pinned_fp4_benchmark`.

Before the cache blocker can clear, the remaining sequence is bounded
real-source proof-ladder steps 10-12, then the full 66-prompt production run
under the heavy-job lock, then the strict raw cache audit. Schema v3 becomes
authoritative only after that raw re-audit succeeds. No teacher-quality,
full-vocabulary-eval, benchmark, release, or family-gate pass is claimed here.

### Evening source-teacher production and first real full-vocabulary verdict

On 2026-07-10, five real source-teacher-cache runs were produced. Four runs
completed, and every completed cache (v1, v2, v3, and v5) is byte-identical for
all 66/66 shards. Strict payload audits pass. The production path now includes
source-blob authentication, `F_NOCACHE`, and bounded MLX cache handling.

This does **not** make the teacher cache release-eligible. Every run was
memory-dirty during source load and layer forward. The best layer phase still
recorded 6,649 pageouts, and instrumented sampling showed free memory falling
to approximately 15 MB mid-loop. The frozen release rule remains zero pageouts
and zero swapouts, so `release_eligible=false`. The completed caches prove real,
deterministic source production and strict payload validity, not release-clean
execution. A clean retry likely requires a quiet machine with other
applications closed.

Release-class source-route traces were also published in the same pass at
`artifacts/quality/glm52-source-route-traces-v5-20260710`.

The first real full-vocabulary candidate comparison is retained as a
diagnostic-only chain in
`artifacts/quality/glm52-candidate-eval-comparison-20260710.json`. Its frozen
wow-gate metrics are:

- mean KLD 1.6340, versus a maximum of 0.30;
- p999 KLD 13.072, versus a maximum of 3.0;
- global top-1 agreement 0.4786, versus a minimum of 0.85;
- mean perplexity ratio 4.0930, versus a maximum of 1.05.

Every domain fails. `benchmarks/diag_glm52_candidate_capture_consistency.py`
proves capture faithfulness: the batched capture is byte-identical, with equal
SHA-256, to a plain single-prompt forward. For the checked true token, teacher
log probability is -5.807 and candidate log probability is -7.907. The
accepted 98.434 GB / 1.5934-bpw artifact therefore genuinely fails the frozen
wow quality gate. The failure is real compression loss at routed semantic
1.03125 bpw, not an evaluation defect.

The four family-gate blockers remain. Quality recovery or artifact redesign is
now a user-level decision. Gate ladder v4-v6, the benchmark harness, `keep
doctor`, report, chat, and model-card surfaces, the troubleshooting guide, and
the real 23-step DAG replacing the high-level guard also landed on 2026-07-10.
Those additions improve execution and diagnosis, but do not change the failed
quality verdict, the frozen memory-cleanliness rule, the four-blocker status,
or release eligibility.

### Generated recovery-campaign handoff

`keep recovery campaign handoff` now renders one deterministic, evidence-bound
view of the configured recovery campaign, verified ledger, current read-only
observation, and exact Git branch/commit. Terminal output is the default;
`--json` emits the same canonical facts as JSON. The command accepts `--config`
and `--ledger` overrides. Every nonempty ledger also requires the paired
`--ledger-head-sha256` and `--ledger-event-count` values from an independently
retained external anchor; missing, partial, mismatched, or rolled-back anchors
fail before projection. A missing canonical ledger is reported as honest
absence without anchors, while corrupt or contradictory evidence fails loudly
or returns the documented contradiction status; missing values are never
rendered as zero.

`--out artifacts/quality/glm52-recovery-campaign-handoff-20260711.md` publishes
the Markdown rendering transactionally to the ignored quality-artifact area and
prints an authenticated path, SHA-256, and byte-count receipt. Existing files,
unsafe paths, symlinks, and the protected root handoffs are rejected rather than
overwritten.

The handoff accepts a quality row only when a single-use release evidence record
binds the exact canonical row SHA, purpose, manifest/candidate/baseline
identities, producer invocation, and named verification result. Diagnostic,
unrelated, reused, or contradictory evidence cannot produce a positive row.
Ledger approval events alone cannot establish machine verification or
independent review because they do not authenticate the external profile result
or parser-sealed review record. The human release boundary therefore always
remains explicit. Generating a handoff is not release approval, does not satisfy
the family gate, and makes no RC quality claim.
