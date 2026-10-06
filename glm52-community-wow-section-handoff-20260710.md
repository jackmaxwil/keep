# GLM-5.2-REAP-KEEP-504B comprehensive continuation handoff

## Resume point

Work in `/Users/jack.mazac/Developer/keep` on branch
`keep-glm52-pipeline-and-p1-lock` at `4f5c5902` (`docs: clarify GLM52
dry-run step key`). The persistent Codex goal is thread
`019f4928-4ea1-7211-8095-31aca8257fe8` with the objective:

> Ship `GLM-5.2-REAP-KEEP-504B` as a community-wow KEEP release candidate:
> source-relative quality against the pinned FP4 REAP model, fast and resident
> on Apple Silicon through RAMP, reproducible from a raw Hugging Face model ID
> with one approachable `keep build` command, and accompanied by legible
> quality, speed, memory, provenance, chat, and publication-ready artifacts.

The goal is **not complete**. It was marked `blocked` only because the active
`superpowers:brainstorming` workflow requires explicit user approval before the
reviewed FP32 teacher-cache design may be written to the repository or
implemented. This is a process/authority gate, not a technical failure. The
exact unblock is the user replying **approve** to the architecture already
presented. Do not treat this handoff request itself as approval.

Two user decisions are already authoritative:

1. The user accepts the actual candidate at **98,433,923,808 tensor-payload
   bytes / 1.593443277857996 whole-main bpw**, commonly rounded to
   **98.434 GB / 1.5934 bpw**. Do not reopen the original 1.3--1.4 bpw / ~90 GB
   target unless the user explicitly asks.
2. The full-vocabulary source-teacher cache must store **FP32 logits**.

The proposed compute contract is BF16 forward compute with FP32 cache storage.
It has been independently reviewed and checked against the live MLX runtime,
but it has not received explicit user approval and has not been committed.

No repository files were changed after `4f5c5902` during the design/handoff
section. Live `git status --short --branch` is:

```text
## keep-glm52-pipeline-and-p1-lock
?? .keep-heavy-job.lock
?? runs/
```

Both untracked paths predate this handoff and are protected. Never stage,
delete, reset, clean, stash, overwrite, or absorb them.

## Authority order

Read these completely before changing code or judging readiness:

1. `/Users/jack.mazac/.codex/attachments/84d57f15-d4ef-4d84-9279-b7743f10b297/goal-objective.md`
2. `/Users/jack.mazac/Developer/keep/AGENTS.md`
3. `/Users/jack.mazac/Developer/keep/docs/COMMUNITY_PRODUCT_PLAN.md`
4. `/Users/jack.mazac/Developer/keep/docs/GLM52_PIPELINE_READINESS.md`
5. `/Users/jack.mazac/Developer/keep/NAMING.md`
6. `/Users/jack.mazac/Developer/keep/WORK_LOG.md`
7. `/Users/jack.mazac/Developer/keep/DISCOVERY.md`
8. Current Git status and diff, including untracked implementation files.
9. The Qwen reference chain the objective names explicitly:
   `src/mlx_vq/build/highlevel.py`, `src/mlx_vq/build/ops.py`,
   `src/mlx_vq/convert/qwen_moe.py`, and
   `recipes/qwen36_35b_a3b_materialization_probe_20260702.yaml`.

Use the live checkout and the latest sections of
`docs/GLM52_PIPELINE_READINESS.md` as current authority when earlier sections
or authenticated prose have drifted. In particular,
`models/glm52-reap-504b-v2.yaml` contains stale checkpoint-era narrative but is
part of the authenticated composite identity. A prior attempt to update that
prose correctly failed the strict family-preflight hash contract and was
reverted. Do not edit the model profile merely to fix prose; a deliberate
identity reissue would have to regenerate every dependent authority.

Two other historical surfaces also contain superseded checkpoint statements:
the top of `docs/COMMUNITY_PRODUCT_PLAN.md` still describes projected/six-group
state, and the top of `DISCOVERY.md` retains older six-group/full-bind blockers.
Do not use those older paragraphs to override the later readiness sections or
raw 225-group evidence.

Public naming remains:

- **KEEP**: compression method and product.
- **RAMP**: routed Apple-Silicon runtime.
- `mlx_vq`: legacy compatibility namespace, not the public identity.

## Hard guards and authority boundaries

- No push, merge, publish, Hugging Face upload, rented hardware, paid service,
  or other external spend without explicit approval.
- No more GLM-4.5-Air quality work.
- Do not touch peer networking, Thunderbolt/RDMA/JACCL, or MAXBOOK for this
  lane. No peer work was used in the current section.
- Leave `GLM_MLX_WIRED_LIMIT_GB` and
  `GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB` unset. A custom MLX or Darwin wired
  limit is a separate, explicitly approved diagnostic.
- No dense routed-expert checkpoint or dense intermediate checkpoint.
- No layer-78 MTP materialization or accounting in the main model.
- No GGUF/llama.cpp detour.
- No canonical compressed-tensors adapter unless the pinned ModelOpt path
  genuinely requires it.
- No fake or stub build operations in a compiling recipe.
- Do not remove the GLM52 high-level compiler guard until the entire real
  source-to-release chain exists and is tested.
- Do not infer companion tensor shard co-location; ModelOpt weight, block
  scale, and global scale may live in different shards.
- Do not claim a BF16 source teacher or exact W4A4 runtime parity. The source
  reference is deterministic dequantization of pinned ModelOpt NVFP4 weights;
  activation quantization through `input_scale` is not emulated.
- Preserve the frozen report/selection/holdout split. Only selection may guide
  recovery; holdout tuning is forbidden.
- Use `UV_CACHE_DIR=/tmp/keep-uv-cache` for `uv` commands.
- The supported test entry point is `uv run --group dev python -m pytest`, not
  `uv run pytest`.
- Heavy artifact/model operations must hold the established advisory
  `.keep-heavy-job.lock`. Build operations that produce artifacts already do
  so through `src/mlx_vq/build/executor.py`.

## Pinned source identity

| Field | Authority |
|---|---|
| Model | `0xSero/glm-5.2-reap-504B-v2` |
| Revision | `6c9241aa05fb243a0edb7c804c213ec1cf5c920d` |
| Snapshot | `$HOME/.cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2/snapshots/6c9241aa05fb243a0edb7c804c213ec1cf5c920d` |
| Source payload | 63/63 shards, 308,829,060,264 bytes |
| Config SHA-256 | `5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b` |
| Index SHA-256 | `bb5b4fa9782aea5ffc66f9145d6e630f1045d385c30437c531bebe422c075f3f` |
| Profile SHA-256 | `ae8b852139ac26e63846b0475064e5b955a00f1ff75f628682a3d8494474114d` |
| Profile contract SHA-256 | `28d95f2f2e411ff886c38b0c773f335c34a1b044e401573e90b399c93db3dcb8` |
| Encoding | Native ModelOpt NVFP4 W4A4, not BF16 and not canonical compressed-tensors |
| Decoder | E2M1 values multiplied by E4M3FN block scale and FP32 global scale |
| Nibble order | Low nibble is even logical column; high nibble is odd logical column |
| Activation metadata | `input_scale` is not part of weight reconstruction |

Pinned decoded-byte SHA-256 oracles remain:

- Layer 10 expert 0 gate:
  `19638efebbca55205804574337d71a87dd0217ef6322e6a399444ecf0d79bfbd`
- Layer 10 expert 0 up:
  `b673bd500f468b16eab2d04eefab9deca16191aef3bc63f2a18d72a0c6a6df13`
- Layer 10 expert 0 down:
  `5e9ce06d0db45f464464f10fcab8bd424ee81f58115e9bea693c87bea3f59dc6`
- Cross-shard layer 29 expert 58 gate:
  `0119f166fe1fb59f47faa5b0580538b7030d1cbaa691206a7f98568f4ec7a117`

## What is genuinely complete

The implementation spine since baseline `75f654f5` is real and committed. The
important wave commits are:

| Wave | Commit | Proven result |
|---|---|---|
| ModelOpt decoder | `f1d039bc` | Producer-matched NVFP4 reconstruction, source safetensors reads, deterministic stream-converter integration |
| Source/payload audit | `3e054050` | Pinned REAP config and 225 routed projection groups, 61 required source shards, no layer-78 groups |
| Materialization | `0670749a` | Real resumable expert-at-a-time E8/E8P group production |
| Routed artifact audit | `fff3e47b` | Strict tensor/tree/lineage/accounting/hash validation |
| Non-routed package | `371be17a` | Audited 37.121 GB lean source-precision non-VQ package |
| Layer-local binding | `76687b4f` | Profile-aware 168-expert bind and finite BF16 layer forward |
| Runtime preflight | `988ff8c6` | Static/runtime IndexShare contract, tokenizer readiness, tiny generation surface |
| Frozen eval authority | `2ade170d` | Immutable 66-prompt policy and report/selection/holdout contract |
| Teacher metadata/gate v1 | `241ec05a` | Honest dequantized-source metadata and fail-loud terminal family gate |
| Composite artifact | `f4de4143` | Common non-VQ+routed identity and exact whole-model accounting |
| Production probe | `729715ed`, `81008501` | Complete bind and real warm-resident one-token generation |
| Raw-HF resolution | `77d37d6f` | Symbolic `main` resolves before cache lookup; incomplete `--check-only` exits nonzero |
| Raw gate v2 | `decb8912` through `f5949d8d` | Raw artifact/production validators and authenticated schema-v2 blocked checkpoint |
| Evidence hardening | `61661c93`, `95715c8e` | Duplicate-key/non-finite rejection and evidence-chain closure |
| Resume/raw-HF/docs | `0a8f00e4`, `f9a7e320`, `ebe6e0d5`, `b2a2b045` | Real interruption recovery, offline raw-ID proof, truthful guard and Quickstart |
| Executor proof | `9558b772`, `4f5c5902` | Real family-evidence build/regate behavior and exact dry-run step key |

Use `git log --oneline 75f654f5..HEAD` for the complete commit spine. Do not
rewrite or squash it as part of the teacher-cache work.

### Source, artifact, and runtime facts now proven

- The real planner produces exactly 225 groups: layers 3--77 times
  gate/up/down. All 37,800 routed expert projection bundles resolve, including
  53 cross-shard bundles. Layer 78 is excluded.
- Full safe single-worker E8 materialization completed all 225 groups. It wrote
  no dense routed checkpoint and passed deterministic zero-write resume.
- The non-routed package contains exactly 1,194 tensors,
  18,560,731,704 parameters, 1,119 BF16 tensors, 75 F32 router-correction
  tensors, and 37,121,488,608 tensor-payload bytes across nine shards.
- Full composite bind sees all 1,272 runtime non-VQ targets, all 225 routed
  groups, no unbound routed experts, and no layer-78 payload.
- A complete authenticated model generated token ID `785` (`The`) and matched
  the upstream MLX-LM warmup token exactly.
- Raw-HF offline readiness is proven from the unpinned model ID: a complete
  isolated cache view resolved `main` to the immutable revision, reported
  63/63 shards, and exited 0; an intentionally incomplete 1/63 view reported
  `ready: no` and exited 1 without downloading or mutating the source.
- A real bounded two-group materialization was interrupted with SIGINT after
  the first atomic group publication, resumed by converting only the missing
  group, matched production hashes, and then replayed with zero source reads
  and zero group-payload writes.

## Accepted artifact accounting

| Quantity | Exact value |
|---|---:|
| Main-model parameters, excluding MTP | 494,194,805,304 |
| Routed parameters | 475,634,073,600 |
| Non-routed parameters | 18,560,731,704 |
| Routed codes | 59,454,259,200 bytes |
| Routed scales | 1,857,945,600 bytes |
| Routed codes + scales | 61,312,204,800 bytes |
| Routed codebook | 230,400 bytes |
| Routed tensor payload including codebook | 61,312,435,200 bytes |
| Non-routed tensor payload | 37,121,488,608 bytes |
| Whole-main tensor payload | **98,433,923,808 bytes** |
| Whole-main tensor-payload bpw | **1.593443277857996** |
| Whole artifact file bytes | 98,435,103,723 bytes |
| Whole artifact file bpw | 1.593462378261114 |
| Whole artifact tree bytes | 98,435,323,829 bytes |
| Whole artifact tree/physical bpw | 1.593465941325681 |
| Routed semantic bpw | 1.03125 |

The strict audit is
`artifacts/quality/glm52-wave6-full-artifact-audit-20260709.json`, SHA-256
`8026322a533606ed451d13b029d83836fefbf1a938b33529de535f3fc778591a`.
It records `audit_pass=true`, `artifact_integrity_pass=true`,
`full_routed_artifact_ready=true`, `resume_verified=true`,
`byte_identity_verified=true`, and `dense_routed_experts=false`.

The common authenticated composite identity is:

```text
ef9d2e49d4a9d113a13d8b8e6c6ce7ebe60a7e9c7fb7b1b7784357d3efee5067
```

### Residency truth boundary

Production evidence is
`artifacts/quality/glm52-wave6-production-generation-bounded-prefill-warm-resident-20260710.json`,
SHA-256
`758b5bbebbb365cdf69998a034cf3791a6210d2723d5bec6c022336eceadcc82`.

The measured post-warmup interval started at 98,433,923,832 active MLX bytes,
peaked at 103,257,924,340 bytes, and recorded exact pageout/swapout deltas
`0/0` under Darwin `iogpu.wired_limit_mb=0` with no MLX wired-limit override.
This proves complete warm residency and production bind/generation. It does
**not** prove clean cold start or clean sacrificial warmup:

- `warm_residency_proven=true`
- `pre_generation_memory_clean=true`
- `steady_state_generation_memory_clean=true`
- `cold_residency_memory_clean=false`
- `generation_warmup_memory_clean=false`
- `production_residency_proven=false`

Do not convert the warm proof into a clean-cold-residency or publication-speed
claim.

## Current family-gate truth

The direct authority is
`artifacts/quality/glm52-family-gate-raw-evidence-20260710.json`, SHA-256
`5f8918ed1c7a554bc15ca747b6cbc643df06f3386126b10be3d4ca1c1c3ea9ad`.

It is a valid schema-v2 blocked checkpoint:

- `gate_schema_version=2`
- `gate_status=glm52_family_gate_blocked`
- `family_gate_pass=false`
- `release_pass_enabled=false`
- 225/225 routed groups, zero missing
- accepted artifact bytes/bpw ready
- production binding/generation ready
- no dense routed experts

Exactly four release-evidence blockers remain:

1. `dequantized_source_teacher_cache_payload`
2. `full_vocabulary_source_relative_family_eval`
3. `route_math_diagnostics`
4. `same_machine_pinned_fp4_benchmark`

The frozen raw quality gate is exact full-vocabulary comparison, not compact
top-k KLD: mean KLD `<= 0.30`, p999 KLD `<= 3.0`, global top1 agreement
`>= 0.85`, every-domain top1 agreement `>= 0.80`, and mean PPL ratio
`<= 1.05`. The frozen benchmark gate requires `prefill_1k` and `decode_128`,
three clean repetitions per candidate/control scenario on the same machine,
the pinned-FP4 streaming source as control, candidate/reference latency ratio
`<= 1.15`, and exact zero pageout/swapout deltas.

The current one-step checked recipe is
`recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml`. Its observed
semantics are important:

- `keep validate`: exit 0.
- `keep build --dry-run`: exit 0, step key `48146c10b53be9db`.
- Normal `keep build`: top-level exit 1 because the terminal family-gate child
  correctly returns 2; the executor records `status=gate_failed` and preserves
  the valid blocked evidence.
- `keep build --regate`: expected top-level exit 1, appends a `regated` ledger
  event, does not rerun or rewrite the evidence.

The complete executor observation is
`artifacts/quality/glm52-family-evidence-v2-build-regate-observations-20260710.json`,
SHA-256
`c4f324beac207a5e8fe64ff981ba8e9f2382f84bd1c9e92f897a957be59bb806`.
Do not misread the expected top-level exit 1 as corrupt evidence, and do not
mislabel the blocked gate as a passing build.

## FP32 source-teacher/cache section

### Decision state

The user selected FP32 storage. The following architecture is the reviewed
recommendation, not yet an approved spec:

- Deterministically dequantize pinned ModelOpt NVFP4 routed weights to FP32.
- Cast each decoded projection to BF16 immediately before MLX matmul.
- Use the existing BF16 non-routed model path and BF16 LM-head operands.
- Preserve the current router's BF16 gate matmul and FP32 sigmoid, correction,
  top-k selection, normalization, and scaling.
- Store full-vocabulary logits as FP32.
- Continue to declare `activation_quantization_emulated=false`,
  `exact_w4a4_runtime_parity_claimed=false`, and
  `bf16_teacher_claimed=false`.

This is **BF16 compute with FP32 cache storage**, not FP32 inference and not a
BF16 source-checkpoint claim.

### Frozen cache dimensions

The frozen prompt authority is
`artifacts/quality/glm52-family-eval-prompts-20260709-v2.json`, SHA-256
`697677a4949f4ee7e370ac5e9a55e9631b0a1385f95a07dfd3a3f2b87d8edf31`.

| Quantity | Exact value |
|---|---:|
| Prompts | 66 |
| Source tokens | 810 |
| Causal predictor positions | 744 |
| Model vocabulary | 154,880 |
| FP32 logit values | 115,230,720 |
| Raw FP32 tensor bytes | **460,922,880** |
| Report positions / bytes | 238 / 147,445,760 |
| Selection positions / bytes | 255 / 157,977,600 |
| Holdout positions / bytes | 251 / 155,499,520 |
| Longest prompt | 19 tokens |
| Longest predictor input | 18 tokens |

Each prompt feeds `encoded_token_ids[:-1]`; row `i` of its output predicts
`encoded_token_ids[i + 1]`. Right-pad predictor inputs to `[66, 18]` and carry
an authenticated valid-position mask.

### Recommended source-forward architecture

Use layer-major, all-prompt, selected-expert streaming. Reuse the real
`GLM52VQModel` adapter and all architecture-sensitive components from
`src/mlx_vq/models/glm52_vq_adapter.py`: embedding, attention, IndexShare,
norms, dense MLPs, router, shared expert, final norm, and LM head.

Do not call `Glm52VQMoE.__call__` for the source teacher because it requires a
bound VQ `switch_mlp`. Manually execute the sparse MoE portion while reusing the
layer's existing attention, norms, gate, and shared expert:

1. Bind the authenticated 37,121,488,608-byte non-VQ package once.
2. Embed the right-padded `[66, 18]` predictor batch.
3. Use MLX's explicit causal mask with per-row right padding. Live inspection
   confirmed `create_causal_mask(..., right_padding=...)` produces the required
   batched boolean mask. The argument is a per-row **pad count**, so pass
   `18 - (token_count - 1)`, never the valid length itself.
4. Before every sparse MoE, gather only the 744 valid hidden rows. Never let
   padded rows enter router accounting.
5. Run the existing router to produce `[744, 8]` expert indices and FP32 route
   scores. There must be exactly `744 * 8 = 5,952` valid route assignments per
   sparse layer and zero padded assignments.
6. Iterate unique selected experts. Resolve each gate/up/down bundle with the
   completed ModelOpt resolver; decode one projection at a time with
   `read_modelopt_nvfp4_weight`; cast the C-contiguous FP32 weight to BF16 for
   its matmul; evaluate; then release both FP32 and BF16 copies before reading
   the next projection.
7. Fill a BF16 route-output tensor `[744, 8, 6144]` (about 73 MB), preserving
   original top-k route-slot order. Apply scores and reduce in route-rank order
   exactly like the adapter. Expert-major direct accumulation would alter
   floating-point addition order and weaken byte identity.
8. Add the existing shared-expert result and scatter valid outputs back into
   `[66, 18, 6144]`.
9. Continue through layers 0--77, never layer 78.
10. Apply final norm, run the BF16 LM head one prompt at a time, immediately
    convert to FP32, drop no valid predictor rows, and atomically write the
    prompt shard.

It is cheap and goal-aligned to capture source router IDs/scores during this
same pass as a separately authenticated sibling diagnostic. Keep it outside the
canonical one-tensor logit shards. Source route traces alone do not clear the
route/math blocker; a raw candidate comparison and validator are still needed.

The frozen length 18 is below `index_topk=2048`. Live upstream inspection
confirmed that the Indexer returns `None` in this case. The checkpoint schema
may allow optional IndexShare state for future workloads, but this frozen run
must record it as absent.

### Cache artifact contract

Use a GLM52-specific contract, provisionally
`glm52_teacher_cache_fp32_v1`:

```text
<cache-root>/
  glm52-teacher-cache-fp32-manifest.json
  teacher_logits/
    <canonical-prompt-id>.safetensors   # exactly 66 files
```

Each shard contains exactly one tensor:

```text
name: logits
dtype: F32
shape: [token_count - 1, 154880]
layout: contiguous row-major, vocabulary last
semantic: row i predicts encoded_token_ids[i + 1]
```

The manifest must bind the pinned model/revision, profile, config/index,
source encoding/decoder, source and payload audits, non-VQ package, policy,
prompt pack, prompt-content hash, tokenizer identities, storage dtype,
compute/accumulation precision, ordered 66-row inventory, file SHA-256, raw
tensor SHA-256, exact per-split/global counts, and stable content identities.

Do not reuse the generic Air teacher-cache contract unchanged:

- `src/mlx_vq/quality/teacher_cache.py` currently excludes the required
  dequantized ModelOpt source kind.
- Its full-logit writer downcasts to FP16.
- It requires top-k/logprob side tensors that are not part of this canonical
  cache.
- It permits multiple floating dtypes and non-exact vocabulary width.
- Its default validation is not the exact tree, physical extent, hash, and
  all-value scan required for this release trust boundary.

Add a dedicated module such as
`src/mlx_vq/quality/glm52_teacher_cache.py` rather than weakening Air
compatibility.

### Atomicity, resume, and memory evidence

- Hold the global heavy-job lock and a cache/run-specific exclusive lock
  outside the final exact artifact tree.
- Keep resumable state outside the final cache root.
- Write immutable per-layer BF16 checkpoints
  `checkpoints/layer-00000.safetensors` through
  `checkpoints/layer-00077.safetensors`, each carrying hidden
  `[66,18,6144]`, optional IndexShare state, layer identity, valid-mask hash,
  prompt/source/config/precision identities, and previous-checkpoint hash.
- Write a unique sibling temporary file, flush and `fsync` it, atomic-rename,
  then `fsync` the directory before appending the ledger record.
- Resume only from the highest contiguous strictly validated checkpoint chain.
  Orphan temporary files are non-authoritative. A malformed/hash-invalid final
  checkpoint fails loudly; do not silently fall back.
- Write every prompt shard through the same durable temp/fsync/rename protocol.
  Publish the final cache manifest last. Its presence is the completion marker.
- Reuse an existing canonical shard only when the authenticated ledger,
  identity, header, bytes, and hashes all validate. Invalid or unledgered final
  shards fail loudly and are not overwritten.

Because the forward is layer-major, pageout/swapout observations are honestly
attributable to producer phases, not individual prompt shards. The manifest
should record ordered producer phases and let rows reference them. Keep these
separate:

- `payload_integrity_pass`
- `all_producer_memory_counters_known`
- `all_producer_memory_clean`
- `system_wired_default`
- `release_eligible`

Unknown or boolean counters must never collapse to integer zero. Structural
validity may remain diagnostic after a dirty run, but release eligibility
requires known zero pageout/swapout deltas for every contributing phase under
the default wired-memory policy.

### Strict cache auditor

The auditor must distrust the manifest and recompute from the raw root:

- Reject symlinked roots/files/directories, path escapes, absolute paths,
  backslashes, `.`/`..`, special files, missing files, and extras.
- Reject duplicate JSON keys and `NaN`/`Infinity` in both manifest and
  safetensors headers.
- Require exact schema field inventories, exactly one `logits` tensor, no
  uncontrolled metadata, exact F32 dtype/shape/offsets, no gaps/overlap, exact
  physical extent, and no trailing bytes.
- Recompute file and raw tensor SHA-256 values.
- Stream-scan all 115,230,720 values for finiteness and reject constant causal
  rows.
- Zip manifest rows to the validated frozen prompt order and exact token hashes.
- Reconcile row, split, global element, and byte totals to the frozen constants.
- Use no-follow opens plus before/after file identity checks to detect mutation
  during audit, then re-read and verify the manifest identity after traversal.

### Build and family-gate integration

1. Add a new external input kind such as `teacher_cache_artifact` in
   `src/mlx_vq/build/recipe.py`. Preserve legacy `teacher_cache`, which currently
   means a metadata file.
2. Add a strong directory-content hash in `src/mlx_vq/build/hashing.py` that
   streams the manifest and actual shard bytes or incorporates independently
   recomputed shard hashes. Manifest plus names/sizes is insufficient against a
   same-size shard mutation.
3. Register producer/auditor operations in `src/mlx_vq/build/ops.py`. An
   artifact-producing op automatically enters the heavy-job lock path.
4. Preserve the current negative teacher-metadata authority unchanged; it
   honestly says no teacher cache exists.
5. Extend `benchmarks/check_glm52_family_gate.py` with raw cache root/manifest
   inputs. It must call the strict validator directly and never trust a prepared
   `audit_pass` boolean or self-rehashed summary.
6. Add gate schema v3 in `src/mlx_vq/quality/glm52_family.py` while preserving
   historical v1/v2 behavior and requiring the complete current v2 raw trio.
7. Extend `src/mlx_vq/build/gates.py`; it currently recognizes the exact v1/v2
   blocked contracts and requires the teacher-cache check to remain false.
8. Cross-bind the resulting cache identity into later raw candidate-eval rows.

Expected gate exits after that integration:

- No cache: current schema-v2 valid blocked result, direct checker exit 2.
- Supplied corrupt, unauthenticated, or memory-dirty release cache: invalid
  evidence, exit 1.
- Fully audited clean cache: schema-v3 blocked checkpoint, exit 2, with only
  full-vocabulary eval, route/math diagnostics, and same-machine benchmark
  remaining.

Do not remove the high-level compiler guard after the cache alone.

### Required proof ladder

Use TDD and progress in this order:

1. Tiny synthetic ModelOpt bundle: decoder result against an independent dense
   oracle.
2. Tiny MoE: selected-expert streaming against a dense all-expert reference,
   including exact route IDs and route-rank reduction.
3. Batched right-padding forward against individual-prompt forwards.
4. Exactly 5,952 valid assignments per sparse layer and zero padded
   assignments.
5. Precision contract: decoder F32, streamed weights/hidden/output BF16,
   selected scores F32, stored logits F32.
6. Interrupted layer checkpoint plus resume; final cache byte-identical to an
   uninterrupted run.
7. Orphan temporary recovery and tampered-checkpoint fail-loud behavior.
8. Adversarial cache writer/auditor tests using a tiny vocabulary fixture.
9. Existing real pinned decoded-byte oracle.
10. One real expert gate/up/down bundle.
11. One complete real sparse layer.
12. One frozen prompt through all 78 main layers, compared with an independent
    non-checkpointed execution.
13. All 66 prompts: exact 744 positions, 115,230,720 FP32 values, and
    460,922,880 raw bytes.
14. Independent full raw cache audit and zero pageout/swapout deltas under the
    system-default wired policy.

Do cheap structural and synthetic gates before any full source pass.

### Rejected alternatives

- **Prompt-major sequential source forward:** simplest prompt-level resume, but
  may reread most of the routed source up to 66 times. It is unacceptable for
  the full cache.
- **Dense decoded routed layer/model:** roughly 25.37 GB of decoded routed
  weights per sparse layer if all 168 experts are resident, dangerous copy
  peaks, and contrary to the no-dense-checkpoint guard.
- **External vLLM/TP8 native W4A4:** requires unavailable/unapproved external
  hardware and spend, and changes the reference semantics toward activation-
  quantized W4A4 rather than the chosen deterministic dequantized-weight
  reference.

## Exact next-agent sequence

The current design workflow must be resumed faithfully:

1. Ask for or recognize an explicit user **approve** response to the reviewed
   architecture. This handoff request is not approval.
2. After approval, write the design spec at a path such as
   `docs/superpowers/specs/2026-07-10-glm52-source-teacher-cache-design.md`.
   Include architecture, data flow, claim boundaries, cache schema, failure
   semantics, security/trust boundaries, resume, memory evidence, testing, and
   the rejected alternatives above.
3. Self-review the spec against the live code and this handoff, run
   `git diff --check`, and create a focused local commit. Do not stage
   `.keep-heavy-job.lock`, `runs/`, generated model payloads, or unrelated
   changes.
4. Ask the user to review the committed spec. Do not begin implementation until
   that spec review is approved.
5. Only after spec approval, invoke `superpowers:writing-plans` and produce a
   concrete TDD implementation plan.
6. Execute the plan in synthetic/bounded waves, independently review meaningful
   waves, update readiness/ledgers with exact evidence, and create cohesive
   local checkpoint commits.

Likely implementation surfaces, subject to the approved spec/plan:

- New source runner near `src/mlx_vq/models/glm52_source_teacher.py`
- New family cache contract near
  `src/mlx_vq/quality/glm52_teacher_cache.py`
- New producer/validator CLIs under `benchmarks/`
- New focused source-streaming and cache-auditor tests
- `src/mlx_vq/build/recipe.py`
- `src/mlx_vq/build/hashing.py`
- `src/mlx_vq/build/ops.py`
- `benchmarks/check_glm52_family_gate.py`
- `src/mlx_vq/quality/glm52_family.py`
- `src/mlx_vq/build/gates.py`
- A schema-v3 family-evidence recipe and readiness/ledger updates

## Safe commands and expected outcomes

Start every resumed session with:

```zsh
cd /Users/jack.mazac/Developer/keep
git status --short --branch
git log -8 --oneline --decorate
SNAPSHOT="$HOME/.cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2/snapshots/6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
test -f "$SNAPSHOT/config.json"
test -f "$SNAPSHOT/model.safetensors.index.json"
```

Cheap current family-evidence checks:

```zsh
UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONDONTWRITEBYTECODE=1 \
  uv run keep validate \
  recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml

UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONDONTWRITEBYTECODE=1 \
  uv run keep build \
  recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml --dry-run
```

Expected: both exit 0; dry-run step key `48146c10b53be9db`.

The normal build/regate commands are no-model evidence composition, but they
return expected top-level exit 1 because the valid gate child returns 2:

```zsh
set +e
UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONDONTWRITEBYTECODE=1 \
  uv run keep build \
  recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml
build_rc=$?

UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONDONTWRITEBYTECODE=1 \
  uv run keep build \
  recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml --regate
regate_rc=$?
set -e
test "$build_rc" -eq 1
test "$regate_rc" -eq 1
```

Directly recompose the current gate into `/tmp` without replacing the checked
authority:

```zsh
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
  --output-json /tmp/glm52-family-gate-handoff.json
rc=$?
set -e
test "$rc" -eq 2
cmp -s /tmp/glm52-family-gate-handoff.json \
  artifacts/quality/glm52-family-gate-raw-evidence-20260710.json
```

Raw-ID readiness without network/model loading:

```zsh
HF_HUB_OFFLINE=1 UV_CACHE_DIR=/tmp/keep-uv-cache \
  uv run keep get 0xSero/glm-5.2-reap-504B-v2 --check-only
```

Focused tests should use:

```zsh
UV_CACHE_DIR=/tmp/keep-uv-cache \
  uv run --group dev python -m pytest <exact-test-files> -q
```

Do not launch the full source-teacher run from this handoff alone. It requires
design approval, committed spec approval, implementation, synthetic parity,
bounded real-source proof, and the heavy-job lock first.

## Evidence index

| Evidence | SHA-256 | Meaning |
|---|---|---|
| `artifacts/quality/glm52-family-policy-20260709-v2.json` | `0975f7dc1117c5fba7532e9166f4767546fd691a6cb52520a40f09874f972ce2` | Frozen quality/benchmark/artifact policy |
| `artifacts/quality/glm52-family-eval-prompts-20260709-v2.json` | `697677a4949f4ee7e370ac5e9a55e9631b0a1385f95a07dfd3a3f2b87d8edf31` | Frozen 66-prompt authority |
| `artifacts/quality/glm52-teacher-metadata-20260709.json` | `621a013eb37f617409ec568340976e769610b3b6c8cae346639762d816a7fd7a` | Honest source metadata; explicitly no cache |
| `artifacts/quality/glm52-wave6-full-artifact-audit-20260709.json` | `8026322a533606ed451d13b029d83836fefbf1a938b33529de535f3fc778591a` | Full 225-group composite audit |
| `artifacts/quality/glm52-wave6-production-generation-bounded-prefill-warm-resident-20260710.json` | `758b5bbebbb365cdf69998a034cf3791a6210d2723d5bec6c022336eceadcc82` | Complete bind and warm-resident generation |
| `artifacts/quality/glm52-family-gate-raw-evidence-20260710.json` | `5f8918ed1c7a554bc15ca747b6cbc643df06f3386126b10be3d4ca1c1c3ea9ad` | Current schema-v2 blocked gate |
| `artifacts/quality/glm52-family-evidence-v2-build-regate-observations-20260710.json` | `c4f324beac207a5e8fe64ff981ba8e9f2382f84bd1c9e92f897a957be59bb806` | Real executor build/regate proof |
| `artifacts/quality/glm52-interrupted-resume-proof-20260710.json` | `cecc236576529edfd2f1715cb4354ac79bc2cfe0dc07cffcd218379a942b3b6e` | Actual SIGINT/restart/zero-write materializer proof |
| `artifacts/quality/glm52-raw-hf-get-check-only-20260710.json` | `82ac2559e013952313ef182fd21c2940458ae05e666c852806cf015f0b5f2424` | Offline raw-ID and incomplete-cache exit semantics |

## What still prevents goal completion

Even after a successful teacher cache, the full objective remains open. Do not
mark the goal complete until current evidence proves every item below:

- FP32 dequantized-source cache exists and passes raw/cache/memory audit.
- Candidate full-vocabulary logits exist for the same 66 prompts and are bound
  to the accepted artifact identity.
- Exact source-relative mean KLD, p999 KLD, global/domain top1, and PPL-ratio
  gates pass without compact surrogates, fallback rows, dirty evidence, or
  holdout tuning.
- Route/math diagnostics exist and pass.
- Three clean same-machine repetitions of `prefill_1k` and `decode_128` compare
  the candidate to the pinned-FP4 streaming control and meet the frozen ratio.
- Cold/residency claim boundaries are resolved honestly; current evidence is
  warm-only.
- The high-level guard is replaced only after every referenced real op exists.
- A raw HF ID drives the complete high-level `keep build` chain end to end,
  including product-level interruption/resume.
- `keep doctor`, readable terminal/HTML report, `keep chat`, quickstart,
  troubleshooting, immutable provenance, and publication-ready model card are
  real and tested.
- Independent review, proportional tests, artifact audits, and exact
  reproduction commands are green.
- Publishing artifacts may be prepared, but no Hugging Face upload occurs
  without explicit approval.

The immediate highest-risk unfinished acceptance condition remains the
dequantized-source FP32 teacher cache. Resume there after the explicit approval
sequence; do not detour into Air, dense conversion, peer recovery, or public
surface polish first.
