# Inspiration-repo analysis for the DeepSeek-V4-Flash phase

**Date:** 2026-08-11
**Repos (cloned to `.repos/`, gitignored):**

| Repo | HEAD | License | One line |
| --- | --- | --- | --- |
| [jundot/omlx](https://github.com/jundot/omlx) | `2450a53` (2026-08-11) | Apache 2.0 | Mac LLM inference server: continuous batching, tiered RAM/SSD KV cache, menu-bar app, and a complete DeepSeek-V4 MLX runtime |
| [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) | `9ca0028` (2026-06-11) | Apache 2.0 | DFlash block-diffusion speculative decoding on stock MLX; 2-4.4x lossless speedups |
| [youssofal/mtplx](https://github.com/youssofal/mtplx) | `ed1c8ee` (2026-08-07) | Apache 2.0 | MTP-first runtime: uses models' built-in multi-token-prediction heads with exact rejection sampling; Forge trains MTP adapters |

Context: KEEP pivoted to `deepseek-ai/DeepSeek-V4-Flash-0731` on 2026-08-11
(`docs/deepseek-v4-flash/2026-08-11-pivot-decision.md`). This document mines the three
repos for quantization, recovery, inference, and integration leverage for that
family, then lays out a recommended path.

---

## 1. oMLX — the most load-bearing of the three

### What it is

585 Python files plus a Swift menu-bar app. An OpenAI/Anthropic-compatible
server over MLX with continuous batching, a paged KV cache with a hot RAM tier
and cold SSD tier (`omlx/cache/paged_ssd_cache.py`), block-aware prefix cache,
a cache-corruption recovery manager (`omlx/cache/recovery.py` —
`CacheRecoveryManager` lets the scheduler survive cache failures instead of
dying), speculative prefill (`omlx/specprefill/`), and a `patches/` plugin
architecture that grafts model families onto pinned mlx-lm.

### The DeepSeek-V4 runtime nobody has to write again

`omlx/patches/deepseek_v4/` is a complete, working V4 family runtime
(~7,700 lines): model, switch layers, chat template, tool parser, tokenizer
patch, cache handlers, verify kernels. Provenance is explicit: the loader
patch is "copied verbatim from PR 1192 head `5c10538`" against pinned mlx-lm
`0.31.3 (ed1fca4)` — meaning **mlx-lm PR 1192 is the upstream DeepSeek-V4
implementation** and should be tracked for merge.

Key facts KEEP needs:

1. **The FP8 problem is already solved for inference.** V4-Flash's
   `e4m3 weight + e8m0 block scale` maps onto MLX's **native `mxfp8` quant
   mode after sanitization** — no BF16 dequant materialization. The
   safetensors `F8_E8M0` dtype (which `mx.load` rejects) is handled by a
   header-rewrite trick: patch the header to advertise `U8`, load, restore
   (`utils_patch.py:_load_safetensors`).
2. **The reference per-module quant map** (`deepseek_v4_model.py:
   make_quantization_config`): routed experts → `mxfp4` (gs32), shared
   experts → `mxfp8`, attention + indexer projections → `mxfp8`, MTP fusion
   projections (`e_proj`/`h_proj`/`main_proj`) → `mxfp8`, everything else →
   affine q8/gs64. **The community's default V4-Flash on a Mac is mxfp4
   routed experts.** That is KEEP's real comparison baseline.
3. **V4 structural quirks encoded in the model:** per-layer
   `compress_ratios` restricted to {0, 4, 128} (DSpark pooled/sparse
   attention), hyper-connections (`hyper_connection.py`), `LimitedSwiGLU`,
   an embedded DSpark verify path with fixed-point `decode_consistency`, and
   Metal `verify_qmv` kernels with inline fp8 decode
   (`omlx_fp8_weight`/`omlx_fp8_scale`).
4. **`oq.py` (8,354 lines): mixed-precision recipe engine.** Combines GGUF
   K-quant layer positioning, unsloth-Dynamic-2.0-style selective
   non-quantization, and MSE-optimal clipping. Levels oQ2–oQ8; fractional
   levels (2.5/2.7/3.5) add **routed-expert protection**: "Super Weights"
   floors that hold routed `down_proj` above base bits, expert-count- and
   hidden-size-conditional rules (e.g. different treatment when
   `num_experts >= 512 and hidden_size >= 4096`), and GLM indexer
   projections pinned to Q8. Two ideas transfer directly:
   - **Sensitivity proxy:** when the source model exceeds RAM, oQ auto-builds
     a uniform 4-bit proxy to measure layer sensitivity against. KEEP faces
     exactly this with a 284 GB source on a 128 GB Mac.
   - Its per-path predicate structure is the same shape as
     `classify_deepseek_v4_parameter` in KEEP's new policy module —
     convergent design, richer rules worth cribbing.

### KEEP integration surface (the stretch goal)

oMLX's `patches/` directory is a de-facto plugin system: each family patches
the pinned mlx-lm loader and registers module-level quant configs. A forked
oMLX serving KEEP artifacts is therefore not an app rewrite; it is a
`patches/keep_vq/` package that (a) recognizes a KEEP artifact manifest,
(b) swaps `SwitchGLU` for `QuantizedVQSwitchGLU`, and (c) returns a
`make_quantization_config`-style dict for the non-expert tensors. Continuous
batching, tiered KV cache, prefix cache, dashboards come for free. Filed as
side priority per Jack's directive — but the integration point is now known
and small.

---

## 2. dflash-mlx — verify-shaped kernels and disciplined rollback

### What it is

DFlash (arXiv 2602.06036) speculative decoding: a ~1B block-diffusion draft
emits 16 tokens in one pass, the target verifies all 16 in one forward pass,
greedy acceptance keeps the verified prefix. Lossless. Measured 2.2–4.4x on
M5-class hardware at 84–91% acceptance.

### What transfers to KEEP/RAMP

1. **`verify_qmm.py` (1,167 lines): verify-specialized small-M quantized
   matmul kernels.** Decode kernels are M=1; verification is M=8–17. dflash
   ships shape-dispatched Metal variants (`mma2big`, `mma2big_pipe`,
   k-split, NAX M=16 tensor-unit path on `applegpu_g17*`, steel-MMA
   fallbacks for older chips) selected by (K, N, bits). **KEEP's
   `gather_vqmm` kernel family is decode-shaped today; a V4-Flash runtime
   that exploits the MTP head needs wide-M VQ verify variants, and this
   file is the reference for how to structure the dispatch.**
2. **Tape-replay rollback.** Instead of snapshotting recurrent state before
   speculation, record an "innovation tape" during verify and replay only
   accepted steps through a Metal kernel. Cheap rollback that preserves
   numerical coherence over long generations. The discipline — never
   snapshot what you can replay — applies to RAMP's KV/recurrent state
   management and to KEEP's crash-resumable runtime work.
3. **Prefix cache L1 (RAM) + L2 (SSD)** with byte/entry budgets and eviction
   (credits oMLX's tiered design). Snapshots include captured hidden states
   and last logits, so hits skip prefill entirely.
4. **Positional sparse prefill:** prefill the target on a *subset* of tokens
   placed at their true RoPE positions (contiguous runs stay bitwise-exact).
   Interacts with oMLX specprefill. Relevant to teacher-cache generation:
   the same machinery could produce supervised positions for a selected
   subset of a long session without paying full prefill.

---

## 3. mtplx — the MTP playbook V4-Flash needs

### What it is

MTP-first runtime: uses the MTP heads models already ship (V4-Flash has
`num_nextn_predict_layers: 1`), drafts D tokens ahead, verifies in one
batched pass, accepts via **exact Leviathan–Chen rejection sampling with
residual correction** — output distribution provably identical at real
sampling settings (temperature 0.6 / top_p 0.95), 1.6–2.24x measured.

### What transfers to KEEP

1. **Direct DeepSeek-V4 support already exists:**
   - `deepseek_mtp_patch.py` — runtime MTP injection for
     `deepseek_v3/v32/glm_moe_dsa` configs (reads
     `num_nextn_predict_layers`).
   - `deepseek_v4_adaptive_width.py` — preregistered max-K3 speculation
     policy with margin thresholds (D1 0.25, D2 10.0) and a sealed
     construction fingerprint.
   - `deepseek_v4_attention_island.py` — compiles only the post-attention
     dependency chain (HC-post, FFN HC-pre/RMSNorm, router, gather-QMM MoE,
     route reduction, HC-post) into fixed-shape "islands"; attention stays
     eager with the exact compressed slice. Notes the production V4
     checkpoint has **three structural layer layouts** (hash/gs32,
     score/gs32, score/gs64) × physical M2/M3/M4 = nine module tapes, with
     layer weights passed as array inputs so 43 layers don't create 129
     compiled closures.
   - `deepseek_v4_attn_proj_wide_m3.py` — physical-M3 Q4 wq_b projection
     lane reusing each packed weight word across three verifier rows.
2. **Forge = the MTP recovery-training pipeline KEEP will want.**
   `mtp_adapters.py` implements LoRA install → freeze-for-training →
   collect/save/merge for MTP heads, and Forge verifies the result is
   *actually faster and still exact* before publishing ("Depth 1 is
   fastest: 227.1 → 296.1, 1.30x") — with the explicit refusal to attach a
   sidecar MTP head to an arbitrary trunk because provenance can't be
   proven. That evidence discipline is KEEP's house style already.
3. **`proj_quant.py` reframes the decode budget:** decode throughput is
   bytes-read-per-token, and trunk residents (attention projections, dense
   + shared-expert MLPs) dominate the *always-read* set on MoE models.
   Measured: q4 residents cut per-token reads 11.9 → 7.9 GB, decode
   33.9 → 43.1 tok/s, pass@1 statistically unchanged (McNemar p=1.0).
   KEEP quantizes routed experts (93–98% of params) — but on V4-Flash the
   trunk is only hidden-4096 and the *per-token active set* is 13B, so
   resident precision and expert precision both matter to tok/s.
4. **KV quantization ladder:** in-tree paged q8/q4 KV with guaranteed
   MLX-SDPA dequant fallback (`kv_quant.py`), TurboQuant vLLM-Metal kernels
   with FWHT rotation and a Lloyd-Max 3-bit table for the adventurous
   (`turboquant.py`). At 1M-token contexts, KV precision is a first-class
   quality lever V4-Flash forces onto KEEP's roadmap.

---

## 4. Synthesis — recommended path for KEEP × DeepSeek-V4-Flash

### Immediate implications for in-flight work

- **FP8 strategy (pivot doc open item) is resolved:** runtime keeps weights
  in MLX-native `mxfp8`/`mxfp4`; KEEP's `keep.convert.fp8_block` dequant is
  for **codebook fitting and teacher parity only**, not inference. The
  cutover plan's Task 7 stands, with that scope note.
- **The headline baseline changes.** KEEP's comparison target for V4-Flash
  is not BF16 and not affine-q4 — it is **mxfp4 routed experts (oMLX
  default)**. KEEP's pitch: VQ E8/E8P beats mxfp4 quality at equal ~4 bpw,
  and reaches 2–3 bpw where mxfp4 cannot go, with oQ-style Super-Weights
  floors (down_proj protection) as the mixed-precision ladder.
- **Adapter effort collapses.** Between mlx-lm PR 1192 (track for merge)
  and oMLX's Apache-2.0 `patches/deepseek_v4/`, KEEP's V4 adapter is a
  bind-VQ-experts layer over an existing model implementation, exactly as
  the GLM-5.2 adapter subclassed `deepseek_v32`. The follow-on adapter plan
  should vendor or depend on that implementation, not rewrite it.

### The compounding thesis (research ideation)

The three repos multiply rather than add:

```
tok/s ≈ (bytes moved per token)⁻¹ × (accepted tokens per verify pass)
        └── KEEP VQ shrinks this        └── MTP/DFlash grows this
```

- **VQ × MTP compounding:** V4-Flash ships an MTP head. Speculative decode
  multiplies whatever per-token byte reduction VQ achieves — a 2.5-bpw VQ
  artifact at MTP depth 1–2 could plausibly deliver 3–5x aggregate over the
  bf16 AR baseline on a Mac. Nobody has published VQ-expert + MTP-verify on
  Apple Silicon. That is the wow candidate for this family.
- **Acceptance rate as a quality gate.** MTP acceptance is a sensitive,
  cheap, task-relevant quality metric for quantization: if VQ degrades the
  trunk, acceptance drops before benchmark deltas are resolvable. Add
  "acceptance rate at fixed depth vs source" to the family gate alongside
  NLL/KLD — mtplx's tune/verify loop is the harness pattern.
- **MTP head recovery as a KEEP lever.** After VQ'ing experts, fine-tune
  the (single, dense, small) MTP block with mtplx-style LoRA against the
  teacher to restore acceptance. Cheap to train, directly monetizes as
  tok/s, and slots into KEEP's existing sidecar-trainer machinery.
- **Wide-M VQ verify kernels.** Port the dflash `verify_qmm` dispatch shape
  (M=2–17, K-split, NAX-M16-when-available, steel fallback) onto
  `gather_vqmm`. Without this, MTP verify on VQ weights falls back to slow
  paths and the compounding above dies. This is the main new Metal work and
  aligns with the standing MLX/Metal-standardization directive.
- **Teacher campaign becomes tractable.** 284 GB FP8 fits a single
  8xH100 p5.48xlarge's 640 GB HBM entirely on-GPU (GLM-5.2 at ~1 TB never
  did) — no host-RAM streaming, likely no multi-day run. The retained teich
  pack (257 sessions, 2.5M supervised tokens) reuses as-is. Capacity
  blocks, not on-demand retries.
- **KV quantization enters scope.** 1M-token context makes KV the dominant
  memory at long context; mtplx's q8/q4-with-fallback ladder and TurboQuant
  FWHT approach define the design space. Candidate later lever: VQ-coded KV
  cache (E8 codebooks on KV blocks) — same math, new tensor.
- **Sparse-attention distillation question (novel, worth a note):** V4's
  DSpark layers (compress ratios {0,4,128}) mean teacher logits differ
  between sparse and full attention modes. Which mode should the teacher
  cache record for distilling a VQ student that will run sparse? Likely
  answer: match the deployment mode (sparse), but measure the divergence
  on a teich subset first. No literature covers this yet.

### Sequencing recommendation

1. Land the decoupling plan (already written:
   `docs/superpowers/plans/2026-08-11-keep-deepseek-v4-flash-cutover.md`).
2. Weight download + measurement pass (parameter count, layer layouts,
   `first_k_dense_replace`, MTP tensor names, per-layer compress ratios).
3. V4 adapter plan: bind VQ experts over the PR-1192/oMLX model; mxfp4
   baseline artifact for comparison; family gate with acceptance-rate row.
4. Wide-M `gather_vqmm` verify kernels (Metal), then MTP-on-VQ end-to-end.
5. MTP-head LoRA recovery lever; teacher campaign on capacity-block H100s.
6. Stretch, unblocking nothing: oMLX fork with a `patches/keep_vq/` package.

---

## 5. Determined FP4 routed-expert format (Wave 2, revision `7872f01b`)

Measured against the local shards at `~/models/DeepSeek-V4-Flash-0731` and
implemented in `src/keep/convert/fp4_expert.py` (tests:
`tests/test_fp4_expert_dequant.py`). This is the format every VQ codebook fit
has to read, so the facts are pinned with citations rather than inferred.

| Fact | Value | Authority |
| --- | --- | --- |
| Storage | `I8 [O, K/2]` for a logical `[O, K]`; two FP4 per byte, packed along `K` | `inference/model.py:137-143`; `inference/convert.py:135` (bare `.view()`) |
| Shapes | `w1`/`w3` `I8 [2048, 2048]` → `[2048, 4096]`; `w2` `I8 [4096, 1024]` → `[4096, 2048]` | shard headers, layer 0 expert 0 |
| **Nibble order** | **low nibble = first logical element**, high nibble = second | `inference/convert.py:30-33` + `:42`; independently, `omlx/.../deepseek_v4_model.py:2219-2226` hands the bytes to MLX `mxfp4` unreordered |
| **Value table** | OCP e2m1, no NaN/inf: `0, .5, 1, 1.5, 2, 3, 4, 6` and negatives | `inference/convert.py:11-14` (`FP4_TABLE`) |
| Scales | `F8_E8M0 [O, K/32]` — one bias-127 exponent per **32 logical** inputs (`4096/128 == 2048/64 == 32`) | `inference/model.py:139-143`; `inference/convert.py:26,28` |
| **Scale rule** | `logical[o, g*32:(g+1)*32] = e2m1(codes) * 2**(scale[o,g] - 127)` — plain per-group multiply, no offset | derived from `inference/convert.py:44-52` (the FP4→FP8 recast's `offset * s_blk` telescopes to `scale`); matches `inference/kernel.py:498-509` |

Two deliberate divergences, both pinned by tests:

- **Code `0x8`** is e2m1 negative zero. DeepSeek's `FP4_TABLE` writes it as a
  bare `0.0`; MLX's `mxfp4` kernel and the OCP spec give `-0.0`. We follow
  OCP/MLX so the sign bit survives, consistent with `fp8_block.decode_e4m3`.
  Numerically identical either way.
- **E8M0 byte `0xFF`** decodes to NaN (OCP / `ml_dtypes.float8_e8m0fnu`, and
  the choice `fp8_block.decode_ue8m0` already made, reused here). MLX's
  `mxfp4` yields `+inf`. Real shards contain no `0xFF` scale bytes, so this is
  documentation, not a hazard.

`mx.dequantize(..., group_size=32, bits=4, mode="mxfp4")` agrees with the
numpy decode **bit-for-bit**, on random bytes and on a full real `w1`. Since
oMLX runs the model by passing the shipped bytes into exactly that path, MLX's
unpack order *is* the release's unpack order — this is what pins the nibble
order on real data, because the statistical gate cannot: a nibble swap is a
permutation *within* each 32-wide scale group, leaving absmax, rms, sign
balance and the e2m1 grid all invariant.

Measured sanity (layer 0 expert 0, all three projections): `absmax`
0.125–0.25, `rms` ≈ 0.0246 (spread across `w1`/`w2`/`w3` < 1.01×), mean ≈ 1e-5,
43.5 % negative, 13 % exactly zero, scale exponents `2**-8 … 2**-4`
(`absmax = 6.0 * 2**-5` ✓).
