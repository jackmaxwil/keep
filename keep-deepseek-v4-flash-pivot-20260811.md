# KEEP pivot: GLM-5.2 → DeepSeek-V4-Flash-0731

**Date:** 2026-08-11
**Decision owner:** Jack Mazac
**Status:** Decided and executed. GLM-5.2 is retired as KEEP's primary target.

## Decision

KEEP's primary model target moves from `GLM-5.2-REAP-504B` to
`deepseek-ai/DeepSeek-V4-Flash-0731`. The reason is intelligence per unit of
model size: DeepSeek-V4-Flash reaches effectively the same measured
intelligence as GLM-5.2 with roughly a third of the active parameters and a
third of the total parameters, at a tenth of the hosted price.

| Metric | GLM-5.2 (max) | DeepSeek-V4-Flash-0731 |
| --- | ---: | ---: |
| Artificial Analysis Intelligence Index | 53 | 52 |
| Total parameters | 753 B | 284 B |
| Active parameters per token | 40 B | 13 B |
| Output speed (AA) | 137.7 tok/s | 130.9 tok/s |
| Hosted price, input / output per 1M | $1.40 / $4.40 | $0.14 / $0.28 |
| License | MIT | MIT |

Sources: `https://artificialanalysis.ai/models/glm-5-2`,
`https://artificialanalysis.ai/models/deepseek-v4-flash`. One point of
intelligence index for 2.65x fewer total parameters and 3.1x fewer active
parameters is the whole argument. KEEP exists to compress routed MoE experts;
its value is highest on the model that already has the best
intelligence-to-size ratio, because the compressed artifact inherits that
ratio.

Note one unresolved inconsistency in the public numbers: the Hugging Face model
card states "304B params" while Artificial Analysis reports 284 B total. Treat
the parameter count as approximately 284–304 B until it is measured directly
from the downloaded shard index, and record the measured value in the family
profile rather than either published figure.

## Target model facts

From `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731` and its
`config.json`:

| Field | Value |
| --- | --- |
| `model_type` | `deepseek_v4` |
| `architectures` | `["DeepseekV4ForCausalLM"]` |
| `num_hidden_layers` | 43 |
| `hidden_size` | 4096 |
| `vocab_size` | 129280 |
| `n_routed_experts` | 256 |
| `n_shared_experts` | 1 |
| `num_experts_per_tok` | 6 |
| `moe_intermediate_size` | 2048 |
| `q_lora_rank` | 1024 |
| `qk_rope_head_dim` | 64 |
| `max_position_embeddings` | 1048576 |
| `torch_dtype` | `bfloat16` |
| `num_nextn_predict_layers` | 1 |
| `index_topk` / `index_n_heads` / `index_head_dim` | 512 / 64 / 128 |
| `quantization_config` | fp8, `e4m3`, scale `ue8m0`, `weight_block_size [128,128]`, dynamic activation scheme — **residents only** |
| `expert_dtype` | `fp4` (added to this table 2026-08-11; routed experts are *not* covered by `quantization_config` above) |

Released tensor types include `BF16, I64, F32, F8_E4M3, I8`, and the repository
ships a speculative decoding module. Maximum recommended output length is 384K
tokens at high and max reasoning effort. The technical report is arXiv
2606.19348.

### What this changes for KEEP, concretely

1. **The source weights are already FP8, not BF16.** This is the single
   largest new engineering item. Every existing KEEP loader path assumes
   BF16/FP16 source shards. DeepSeek-V4-Flash ships block-quantized FP8
   (`e4m3`, 128x128 weight blocks, `ue8m0` scales). Expert tensors must be
   dequantized from FP8 blocks to BF16 before VQ codebook fitting, or the
   quantizer must learn to consume FP8 blocks directly. KEEP has no FP8 path
   today — `src/mlx_vq/convert/nvfp4.py` is NVFP4 and is adjacent machinery,
   not a substitute.

   > **Correction, 2026-08-11 (measured from the real shards, revision
   > `7872f01b1d1fe23eabc4c98b48bffcef5a386062`).** "FP8 throughout" was read
   > off `quantization_config` alone and is wrong. `config.json` also carries
   > `expert_dtype: "fp4"`, and the shards show two formats:
   > - **Residents — attention *and* shared experts:** `F8_E4M3` codes with
   >   128x128 block scales, exactly as `quantization_config` describes
   >   (e.g. `layers.N.attn.wkv.weight` `F8_E4M3 [512,4096]`,
   >   `layers.N.ffn.shared_experts.w1.weight` `F8_E4M3 [2048,4096]` with
   >   `.scale` `F8_E8M0 [16,32]`). This is what `keep.convert.fp8_block`
   >   decodes.
   > - **Routed experts:** FP4 packed two-per-byte in `I8` —
   >   `layers.N.ffn.experts.E.w1.weight` `I8 [2048,2048]` for a logical
   >   `[2048,4096]` — with `F8_E8M0` **group-32** scales `[2048,128]`
   >   (`w2`: `I8 [4096,1024]`, scale `[4096,64]`). `fp8_block` cannot read
   >   these; they need a separate FP4/E8M0 decoder.
   >
   > So the largest engineering item is *two* decoders, not one, and the
   > routed-expert half — the part VQ actually consumes — is the one still
   > unbuilt. On-disk size is **~172 GB across 48 shards**, not 284 GB; the
   > 284 GB figure assumed FP8 experts. BF16-dequantized is still ~568 GB.
2. **The DSA attention path is already DeepSeek-shaped.** GLM-5.2's adapter
   subclasses `mlx_lm.models.deepseek_v32.DeepseekV32Attention`
   (`src/mlx_vq/models/glm52_vq_adapter.py:313`). The `ModelArgs` field set of
   mlx-lm 0.31.3's `deepseek_v32` matches DeepSeek-V4-Flash's `config.json`
   field-for-field on every attention and MoE field that matters. mlx-lm has
   no `deepseek_v4` module yet, so the adapter must supply the model class, but
   it inherits far more from upstream than the GLM-5.2 adapter did.
3. **Shallower and narrower.** 43 layers against GLM-5.2's 78, hidden 4096
   against 6144. Per-layer artifacts drop accordingly, and the long-context
   chunked-attention fixes from July stay relevant but operate on smaller
   tensors.
4. **Far more experts, smaller each.** 256 routed experts at top-6 against
   GLM-5.2's 168 at top-8, same `moe_intermediate_size` of 2048. More
   projections per layer, each cheaper. Group-size policy and codebook
   selection must be re-derived, not inherited.
5. **The GPU requirement drops sharply.** GLM-5.2-REAP-504B in BF16 is roughly
   1.0 TB, which is why teacher generation demanded a 2 TB-host-RAM
   `p5.48xlarge` and host-side layer streaming. DeepSeek-V4-Flash in native FP8
   is roughly 284 GB and fits inside a single 8xH100 node's 640 GB of HBM with
   room for KV cache, so teacher generation can run fully on-GPU. Dequantized
   to BF16 it is roughly 568 GB, which still fits 640 GB but leaves little
   headroom; prefer keeping weights FP8 on H100/H200 and dequantizing per-tensor
   on demand. A100-based `p4de.24xlarge` has the same 640 GB but no native FP8
   tensor cores, so it is a fallback, not a peer.
6. **1M context and an MTP layer.** `max_position_embeddings` is 1048576 and
   `num_nextn_predict_layers` is 1. The MTP layer must be explicitly skipped or
   intentionally loaded, exactly as the new-model-family template requires. The
   1M context makes the teacher-cache corpus design a capacity question rather
   than a truncation question.

## GLM-5.2 shutdown, executed 2026-08-11

### Why the GLM-5.2 run was already dead

The final managed job never obtained a GPU. CloudTrail shows 50
`RunInstances` calls between 18:17 and 19:28 PDT on 2026-08-04, every one
failing with `Server.InsufficientInstanceCapacity` for `p5.48xlarge` across all
four `us-west-2` availability zones. SkyPilot recorded job 4 as
`FAILED_CONTROLLER` after 8h 12m. The must-start deadline of
`2026-08-05T06:18:52Z` passed with no GPU, and the watchdog Lambda had been
alarming every ten minutes since. Quotas were never the constraint —
`Running On-Demand P instances` is 768 vCPU, enough for four concurrent
`p5.48xlarge`. Total GPU spend on the campaign: **$0.00**.

### What was killed

| Item | Action | Evidence |
| --- | --- | --- |
| Managed job `glm52-sky-20260724-cache-seed` | Terminal, `FAILED_CONTROLLER` (job 4); jobs 1–3 terminal earlier | `sky jobs queue --refresh -a` |
| SkyPilot jobs controller `sky-jobs-controller-9d9f31a9` | Cluster terminated (`sky down`) | No KEEP EC2 instances remain in `us-west-2` |
| EventBridge rule `keep-glm52-sky-watchdog` | `DISABLED` (was `rate(10 minutes)`, alarming) | `aws events list-rules` |
| EventBridge rule `keep-glm52-sky-must-start-cancel` | Already `DISABLED`, left disabled | `aws events list-rules` |
| GLM-5.2 weight objects in S3 | 764 object versions, 1297.1 GB permanently deleted | see below |
| Orphaned local SkyPilot executor processes holding port 50012 | Killed; API server restartable again | `lsof -nP -iTCP:50012` returns nothing |

S3 deletion covered every version, including non-current versions, under:

- `keep-glm52-models-246813579024-us-west-2`: `source-hub/` (435.8 GB),
  `source-snapshot/` (617.7 GB), `non-vq-package/` (74.2 GB), and
  `.safetensors`/`.npz`/`.bin`/`.gguf` shards under `training-baseline/`
  (122.6 GB)
- `keep-glm52-models-246813579024-us-east-1`: `source-hub/` (9.6 GB),
  `non-vq-package/` (37.1 GB)

All deleted content is either re-downloadable from Hugging Face or rebuildable
from the retained conversion manifests.

### What was deliberately retained

Evidence and anything reusable by the DeepSeek line:

| Prefix | Objects | Size | Why kept |
| --- | ---: | ---: | --- |
| `campaigns/` | 2693 | 178.0 MB | Full campaign lineage, launch claims, submission intents, watchdog history |
| `task13/` | 57 | 21.8 MB | Gates, approvals, templates, activation records |
| `teich-pack/` | 1 | 94.6 MB | The 257-session teich coding-agent corpus — directly reusable for DeepSeek distillation |
| `validation/` | 68 | 4.3 MB | Validation packs and reference selections |
| `campaign-audits/` | 17 | 0.9 MB | Artifact audits |
| `training-baseline/` | 4 | 0.9 MB | JSON evidence only, shards removed |
| `repo/`, `lambda/`, `quality/`, `profile/`, `reviews/` | 10 | 4.1 MB | Staged repository tarballs, packaged Lambdas, small reports |

The `keep-glm52-gpu` CloudFormation stack was **kept deployed**. Its resources
are model-agnostic GPU control plane — VPC, four subnets, gateway endpoint,
SkyPilot controller and worker IAM roles and instance profiles, GPU budget,
cost anomaly monitor and subscription, watchdog and must-start-cancel Lambdas,
their alarms and dead-letter queues. Only the stack name and the watchdog's
hardcoded S3 paths are GLM-specific. Deleting and rebuilding it would re-do
every IAM, VPC, and budget resource and risk another drift-settlement cycle
like 2026-08-03/04. It will be generalized in place instead.

One piece of intentional drift now exists: `SkyWatchdogRule` is hardcoded
`State: ENABLED` in `aws/glm52-gpu/cfn/gpu-teacher-stack.yaml:2153`, and the
deployed rule is now `DISABLED`. The generalization work must introduce an
explicit watchdog-state parameter so the next stack update settles this rather
than silently re-enabling a watchdog pointed at a dead campaign.

## Local toolchain incident, resolved

Every uv-managed CPython on this Mac was chmod'ed to mode `600` — readable only
by owner, no execute bit — at 2026-08-08 21:32–21:33 local time. That silently
broke `uv run`, `pytest`, and the SkyPilot CLI, whose shim shebang points at one
of those interpreters, which is why no work landed between 2026-08-04 and
2026-08-11. `mtime` was untouched, so the change is only visible via `ctime` and
the mode itself. System `/usr/bin/python3` and the `*-config` shell scripts were
unaffected.

Resolved by `uv python install 3.11 --reinstall` and
`uv python install 3.12 --reinstall`, which fetched fresh interpreters with
correct modes rather than re-permissioning files an endpoint agent had stripped.
The freshly installed interpreters were not re-stripped, which suggests a
one-time scan rather than continuous enforcement. StepSecurity and CrowdStrike
agents are both present on this machine; if the modes are stripped again, it is
an IT question, not a repo question.

After the fix, `tests/test_glm52_sky_submission_integration.py` passes 313
tests, including the two uncommitted submission fixes described below.

## Uncommitted work in the tree at pivot time

Two GLM-5.2 submission fixes and one new SkyPilot task file were in the working
tree, all tested and passing:

- `aws/glm52-gpu/scripts/submit_sky_campaign.py` — `sky jobs launch --async`
  corrected to `--detach-run` (verified against the pinned SkyPilot 0.13.0
  `sky/client/cli/command.py`; `--async` is not the flag that command takes),
  and a `--waive-cache-seed-launch-capability` flag that waives only the
  deployed-capability proof while keeping the conditional one-shot launch claim
  intact.
- `tests/test_glm52_sky_submission_integration.py` — coverage for both.
- `aws/glm52-gpu/skypilot/glm52-cache-seed.yaml` — untracked cache-seed task.

The `--detach-run` correction is a real bug fix that survives the pivot: the
generalized submission path needs it regardless of model family. The waiver flag
and the cache-seed task file are GLM-5.2-shaped and should be generalized rather
than carried forward verbatim.

## Current coupling to GLM-5.2

`grep -rli 'glm52|glm-5.2'` counts, by area: `src/` 816 files, `tests/` 939,
`aws/` 154, `benchmarks/` 105, `docs/` 41, `recipes/` 7, `scripts/` 4. Those
counts include `__pycache__` artifacts and incidental mentions, but the
structural coupling is real and lives in four places:

1. **`src/glm52_enforcement/`** — 120 Python modules of campaign, submission,
   fence, watchdog, and spend-enforcement logic, named for the model family it
   happens to have been written against.
2. **`src/mlx_vq/models/`** — five GLM-5.2 modules totalling 4,312 lines
   (`glm52_vq_adapter.py`, `glm52_composite_loader.py`,
   `glm52_source_teacher.py`, `glm52_long_context_attention.py`,
   `glm52_policy.py`), alongside per-family GLM-4.5-Air and Qwen equivalents.
3. **`src/mlx_vq/models/__init__.py`** — a hand-maintained export map with
   family-specific symbol names (`bind_glm52_vq_experts`,
   `bind_glm45_air_vq_experts`, `bind_qwen_*`). There is no family-agnostic
   adapter protocol; each family invents its own binder names, so adding a
   family means editing shared files rather than registering a new one.
4. **`src/mlx_vq/models/profiles.py`** — `ConverterKind` is a closed
   `Literal["glm_stream", "qwen_moe_groups", "glm52_vq_groups"]` with a matching
   `ALLOWED_CONVERTERS` frozenset. A new family cannot be expressed without
   editing this type.

The package-level cutover was already planned and is documented in
`docs/KEEP_NEW_MODEL_FAMILY_TEMPLATE.md`, which states that `mlx_vq.*` imports
"remain supported as the Air-ladder compatibility shim until the GLM-5.2
cutover". `src/keep/` and `src/ramp/` already exist as thin alias packages over
`mlx_vq` via `src/keep/_alias.py`. The pivot is the trigger to finish that
cutover: make `keep`/`ramp` canonical, make family support a registration rather
than an edit to shared files, and add DeepSeek-V4-Flash as the first family
onboarded through the generalized path.

The implementation plan for that work is
`docs/superpowers/plans/2026-08-11-keep-deepseek-v4-flash-cutover.md`.

## Open items

- **Measure the true parameter count** from the downloaded shard index and
  record it in the family profile, resolving the 284 B / 304 B discrepancy.
- ~~**Decide the FP8 strategy**~~: dequantize expert blocks to BF16 before VQ
  fitting, or fit codebooks directly against FP8 blocks. This determines
  whether KEEP's compression ratio is quoted against FP8 or BF16 source, which
  changes every headline number the project publishes.

  > **Resolved by measurement, 2026-08-11** (revision
  > `7872f01b1d1fe23eabc4c98b48bffcef5a386062`). The question was
  > mis-premised: the experts are not FP8. `expert_dtype: "fp4"` — routed
  > experts are FP4-in-`I8` with `F8_E8M0` group-32 scales, residents are
  > `F8_E4M3` with 128x128 blocks. That settles the strategy without a
  > judgement call:
  > - **Resident path: shipped.** `keep.convert.fp8_block` decodes the
  >   `F8_E4M3`/128x128 residents (attention + shared experts).
  > - **Expert path: a new FP4/E8M0 decoder is required**, and it is the
  >   blocker for VQ fitting. Decode-to-BF16-then-fit is the only option on
  >   the table, because there is no FP8 expert block to fit against.
  > - **Ratio denominator: BF16.** Quoting against a 4-bit source would make
  >   KEEP's own ~2.5–3 bpw look like a rounding error rather than a win, and
  >   the honest comparison target is the community mxfp4 artifact, not the
  >   source bytes. Headline numbers stay BF16-relative (~568 GB).
- **Re-derive group-size policy and codebook family** for 256 experts at
  `moe_intermediate_size` 2048. GLM-5.2's 512/512/512 policy is not
  transferable on inspection alone.
- **Do not buy GPU capacity for GLM-5.2.** A `p5.48xlarge` 24-hour Capacity
  Block was available at $996.67 upfront ($41.53/h, cheaper than the $55.04/h
  on-demand rate the campaign priced) and was the unblock for the capacity
  wall. It is moot now. When DeepSeek teacher generation is ready, re-query
  `describe-capacity-block-offerings` — reserved capacity, not on-demand
  retries, is the answer to H100 scarcity in `us-west-2`, and spot is not
  (placement score 1 of 10 in all four AZs).
