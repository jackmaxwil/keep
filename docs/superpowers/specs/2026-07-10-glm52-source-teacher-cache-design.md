# GLM-5.2 Source Teacher Cache Design Specification

**Status:** Approved architecture; design-only specification

**Date:** 2026-07-10

**Target:** `GLM-5.2-REAP-KEEP-504B`

**Canonical cache contract:** `glm52_teacher_cache_fp32_v1`

This specification turns the approved FP32 source-teacher/cache architecture
into an implementation contract. It does not authorize a model run, release,
publish, external hardware use, or removal of the guarded high-level GLM-5.2
compiler path. Implementation and heavy execution remain separate, later
approval and proof steps.

## 1. Objective and claim boundaries

The objective is to produce a deterministic, full-vocabulary source-teacher
cache for the frozen 66-prompt GLM-5.2 family evaluation. The producer shall:

- deterministically dequantize the pinned ModelOpt NVFP4 routed weights to
  FP32;
- cast each decoded routed projection to BF16 immediately before its MLX
  matrix multiplication;
- use the existing BF16 non-routed path and BF16 LM-head operands;
- preserve the current router's BF16 gate matrix multiplication and FP32
  sigmoid, correction, top-k selection, normalization, and scaling;
- store every full-vocabulary logit as FP32.

The resulting claim is **BF16 forward compute with FP32 cache storage**. It is
not FP32 inference, a BF16 source-checkpoint claim, or an emulation of the
pinned model's activation-quantized runtime. The cache manifest, producer
evidence, auditor output, family gate, and every downstream evaluation shall
therefore keep these fields exactly false:

```text
activation_quantization_emulated=false
exact_w4a4_runtime_parity_claimed=false
bf16_teacher_claimed=false
```

The source reference is deterministic dequantization of the pinned ModelOpt
NVFP4 weights. `input_scale` is not emulated and is not part of weight
reconstruction. A structurally valid cache is not automatically
release-eligible: complete raw audit, known clean producer memory evidence, and
the system-default wired-memory policy are all independent requirements.

## 2. Frozen inputs and exact dimensions

### 2.1 Source and prompt authority

| Field | Frozen value |
|---|---|
| Model | `0xSero/glm-5.2-reap-504B-v2` |
| Revision | `6c9241aa05fb243a0edb7c804c213ec1cf5c920d` |
| Profile | `glm52-reap-504b-v2` |
| Profile SHA-256 | `ae8b852139ac26e63846b0475064e5b955a00f1ff75f628682a3d8494474114d` |
| Profile contract SHA-256 | `28d95f2f2e411ff886c38b0c773f335c34a1b044e401573e90b399c93db3dcb8` |
| Config SHA-256 | `5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b` |
| Index SHA-256 | `bb5b4fa9782aea5ffc66f9145d6e630f1045d385c30437c531bebe422c075f3f` |
| Prompt pack | `artifacts/quality/glm52-family-eval-prompts-20260709-v2.json` |
| Prompt pack file SHA-256 | `697677a4949f4ee7e370ac5e9a55e9631b0a1385f95a07dfd3a3f2b87d8edf31` |

The producer and auditor shall consume the frozen prompt rows in their
authenticated order. They shall not re-tokenize prompt text to create cache
inputs. Each prompt feeds `encoded_token_ids[:-1]`; output row `i` predicts
`encoded_token_ids[i + 1]`. Selection is the only tuning-eligible split, and
holdout tuning remains forbidden.

### 2.2 Dimensions and byte accounting

| Quantity | Exact value |
|---|---:|
| Prompts | 66 |
| Source tokens | 810 |
| Causal predictor positions | 744 |
| Model vocabulary | 154,880 |
| FP32 logit values | 115,230,720 |
| Raw FP32 tensor bytes | **460,922,880** |
| Report positions | 238 |
| Report raw bytes | 147,445,760 |
| Selection positions | 255 |
| Selection raw bytes | 157,977,600 |
| Holdout positions | 251 |
| Holdout raw bytes | 155,499,520 |
| Longest prompt | 19 tokens |
| Longest predictor input | 18 tokens |

These totals are identities, not estimates:

```text
744 * 154880 = 115230720 values
115230720 * 4 = 460922880 bytes
238 + 255 + 251 = 744 positions
```

Any count, dimension, split, vocabulary width, element total, or raw byte total
that differs from this table invalidates the cache.

## 3. Source-forward architecture

The producer shall use **layer-major, all-prompt, selected-expert streaming**.
It shall reuse `GLM52VQModel` and the architecture-sensitive embedding,
attention, IndexShare, norms, dense MLPs, router, shared expert, final norm, and
LM head. It shall not call `Glm52VQMoE.__call__` for a sparse source-teacher
layer because that path requires a bound VQ `switch_mlp`. Instead, it shall
execute only the sparse MoE portion manually while preserving the surrounding
adapter behavior.

The normative flow is:

1. Bind the authenticated 37,121,488,608-byte non-VQ package once.
2. Embed the right-padded `[66, 18]` predictor batch.
3. Use MLX's explicit causal mask with per-row right padding. The
   `right_padding` argument is a per-row **pad count**, so pass
   `18 - (token_count - 1)`, never the valid length itself.
4. Before every sparse MoE, gather only the 744 valid hidden rows. Never let
   padded rows enter router accounting.
5. Run the existing router to produce `[744, 8]` expert indices and FP32 route
   scores. There shall be exactly `744 * 8 = 5,952` valid route assignments per
   sparse layer and zero padded assignments.
6. Iterate unique selected experts. Resolve each gate/up/down bundle with the
   completed ModelOpt resolver; decode one projection at a time with
   `read_modelopt_nvfp4_weight`; cast the C-contiguous FP32 weight to BF16 for
   its matrix multiplication; evaluate; then release both FP32 and BF16 copies
   before reading the next projection.
7. Fill a BF16 route-output tensor `[744, 8, 6144]` (about 73 MB), preserving
   original top-k route-slot order. Apply scores and reduce in route-rank order
   exactly like the adapter. Expert-major direct accumulation is forbidden
   because it changes floating-point addition order and weakens byte identity.
8. Add the existing shared-expert result and scatter valid outputs back into
   `[66, 18, 6144]`.
9. Continue through main layers 0--77, never layer 78.
10. Apply final norm, run the BF16 LM head one prompt at a time, immediately
    convert to FP32, drop no valid predictor rows, and atomically write the
    prompt shard.

Layer checkpoints and final shards must preserve this exact computation order.
The implementation must not replace route-rank reduction with expert-major
accumulation, even if a tolerance-based test appears to pass.

The producer may capture source router IDs and scores during the same pass, but
only as a separately authenticated sibling diagnostic outside the canonical
cache tree. Source traces alone do not clear the route/math family-gate blocker;
a raw candidate comparison and validator are still required.

For this frozen run, predictor length 18 is below `index_topk=2048`; the live
Indexer returns `None` when key length is at most `index_topk`. The checkpoint
schema may reserve optional IndexShare state for future workloads, but every
checkpoint and the final manifest for this run shall record IndexShare state as
absent. Absence here is an authenticated fact, not an omitted or unknown field.

## 4. Cache artifact contract: `glm52_teacher_cache_fp32_v1`

### 4.1 Exact final tree

The completed cache root shall contain exactly this tree and nothing else:

```text
<cache-root>/
  glm52-teacher-cache-fp32-manifest.json
  teacher_logits/
    <canonical-prompt-id>.safetensors   # exactly 66 files
```

Locks, ledgers, temporary files, checkpoints, logs, route diagnostics, and
resume state shall remain outside `<cache-root>`. The 66 shard basenames shall
be the 66 canonical prompt IDs in frozen prompt-pack order, with the literal
`.safetensors` suffix. No aliases, nested prompt directories, alternate suffixes,
or extra metadata files are allowed.

### 4.2 Shard contract

Each shard shall contain exactly one tensor and no uncontrolled safetensors
metadata:

```text
name: logits
dtype: F32
shape: [token_count - 1, 154880]
layout: contiguous row-major, vocabulary last
semantic: row i predicts encoded_token_ids[i + 1]
```

The raw tensor payload begins at the declared safetensors data offset, occupies
exactly `(token_count - 1) * 154880 * 4` bytes, has no gaps or overlaps, and
ends at the physical end of the file. The file shall contain no trailing bytes.

### 4.3 Manifest contract and binding inventory

`glm52-teacher-cache-fp32-manifest.json` is a duplicate-key-free, finite-value
JSON object. Its schema shall freeze exact key inventories for the manifest and
all nested records. It shall contain and authenticate all of the following;
none may be inferred from filenames or ambient state:

The exact top-level field inventory is:

```text
schema_version
record_type
created_at
producer
source
source_evidence
non_vq_package
policy
prompt_authority
tokenizer
precision
architecture
totals
producer_phases
shards
payload_integrity_pass
all_producer_memory_counters_known
all_producer_memory_clean
system_wired_default
release_eligible
cache_content_sha256
manifest_body_sha256
```

The exact nested inventories are:

- `producer`: `implementation`, `version`, `mlx_version`, `mlx_lm_version`.
- `source`: `model_id`, `revision`, `profile`, `profile_sha256`,
  `profile_contract_sha256`, `config_sha256`, `index_sha256`,
  `weight_encoding`, `decoder`, `decoded_weight_oracle_contract`.
- `source_evidence`: `source_audit_path`, `source_audit_file_sha256`,
  `source_audit_record_type`, `source_payload_audit_path`,
  `source_payload_audit_file_sha256`, `source_payload_audit_record_type`,
  `routed_group_count`, `routed_projection_bundle_count`,
  `required_payload_shard_count`.
- `non_vq_package`: `evidence_path`, `evidence_file_sha256`,
  `manifest_sha256`, `package_set_sha256`, `retained_tensor_count`,
  `tensor_payload_bytes`, `bound_package_identity`.
- `policy`: `path`, `file_sha256`, `contract_sha256`,
  `activation_quantization_emulated`, `exact_w4a4_runtime_parity_claimed`,
  `bf16_teacher_claimed`, `holdout_tuning_forbidden`,
  `full_vocabulary_logits_required`.
- `prompt_authority`: `path`, `file_sha256`, `contract_sha256`,
  `content_contract_sha256`, `prompt_text_sha256`, `ordered_prompt_ids`.
- `tokenizer`: `base_vocab_size`, `tokenizer_length`, `model_vocab_size`,
  `eos_token_ids`, `files`; each `files` value contains exactly `size_bytes`
  and `sha256`.
- `precision`: `decoded_weight_dtype`, `routed_matmul_weight_dtype`,
  `routed_hidden_dtype`, `routed_output_dtype`, `route_score_dtype`,
  `non_routed_operand_dtype`, `lm_head_operand_dtype`, `stored_logits_dtype`,
  `routed_matmul_accumulation_contract`,
  `route_reduction_accumulation_contract`, `lm_head_accumulation_contract`,
  `route_reduction_order`, `padding_mode`, `right_padding_formula`.
- `architecture`: `predictor_batch_shape`, `hidden_size`,
  `experts_per_token`, `valid_route_assignments_per_sparse_layer`,
  `first_main_layer`, `last_main_layer`, `excluded_mtp_layer`,
  `indexshare_state`.
- `totals`: `prompt_count`, `source_token_count`, `predictor_position_count`,
  `vocab_size`, `fp32_value_count`, `raw_tensor_bytes`, `splits`; each
  `splits` value contains exactly `position_count` and `raw_tensor_bytes`.
- Each `producer_phases` record: `phase_id`, `ordinal`, `start_boundary`,
  `end_boundary`, `contribution_range`, `pageouts_delta`, `swapouts_delta`,
  `memory_counters_known`, `memory_clean`, `system_wired_default`,
  `input_identity_sha256`, `output_identity_sha256`.
- Each `shards` record: `prompt_id`, `split`, `domain`, `tuning_eligible`,
  `token_count`, `token_ids_sha256`, `relative_path`, `tensor_name`, `dtype`,
  `shape`, `element_count`, `raw_tensor_bytes`, `file_sha256`,
  `raw_tensor_sha256`, `producer_phase_ids`.

The numbered requirements below define the required values and relationships
within that exact inventory. All manifest path fields are normalized POSIX
repository-relative logical authority paths, except `shards[].relative_path`,
which is cache-root-relative. Absolute paths and empty, `.` or `..` components
are forbidden.

1. **Contract identity:** schema version 1, record type
   `glm52_teacher_cache_fp32_v1`, producer implementation/version, creation
   timestamp used only as provenance, and a canonical manifest-body SHA-256.
2. **Pinned source identity:** model ID, immutable revision, profile name,
   profile SHA-256, profile contract SHA-256, config SHA-256, index SHA-256,
   source encoding `modelopt_nvfp4`, decoder `modelopt_nvfp4_v1`, and the
   decoded-weight byte-oracle contract.
3. **Source evidence:** normalized repository-relative logical authority paths,
   exact record types, and file SHA-256 values for the source audit and
   source-payload audit; exact 225 routed groups, 37,800 routed projection
   bundles, and 61 required payload shards.
4. **Non-VQ package:** evidence path and SHA-256, package manifest SHA-256,
   package-set/content identity, exact 1,194 tensors, exact
   37,121,488,608 tensor-payload bytes, and the identity proving it is the
   package actually bound by the producer.
5. **Frozen policy:** family-policy path, file SHA-256, embedded policy contract
   SHA-256, the three false claim-boundary booleans, holdout-tuning prohibition,
   and full-vocabulary requirement.
6. **Frozen prompt authority:** prompt-pack path, file SHA-256
   `697677a4949f4ee7e370ac5e9a55e9631b0a1385f95a07dfd3a3f2b87d8edf31`,
   embedded prompt-pack contract SHA-256, prompt-content contract SHA-256,
   prompt-text SHA-256, and the ordered 66-row inventory.
7. **Tokenizer identity:** tokenizer base vocabulary, tokenizer length, model
   vocabulary 154,880, EOS IDs, and the exact filename/size/SHA-256 inventory
   authenticated by the prompt pack and policy.
8. **Precision and execution identity:** decoded weight dtype F32, routed
   matmul weight/hidden/output dtype BF16, route-score dtype F32, non-routed and
   LM-head operand dtype BF16, and stored-logit dtype F32. Matmul accumulation
   is the MLX-native accumulation contract for BF16 operands, bound to exact MLX
   and MLX-LM versions. Route reduction is bound to the adapter expression
   `BF16 route output * FP32 score`, sum on route axis `-2`, then cast to BF16;
   its order is `route_rank`. The padding mode is `right_padding_pad_count` and
   the pad-count formula is `18-(token_count-1)`.
9. **Architecture identity:** batch shape `[66,18]`, hidden size 6,144, eight
   selected experts per token, 5,952 valid assignments per sparse layer, main
   layer range 0--77, layer 78 excluded, and IndexShare state explicitly absent.
10. **Global and split totals:** all constants in Section 2.2, including exact
    element and raw-byte totals for report, selection, holdout, and global data.
11. **Producer phases:** an ordered phase inventory with stable phase IDs,
    start/end boundaries, contribution range, pageout/swapout observations,
    counter-known state, clean state, wired-memory-policy state, and hashes of
    the inputs and outputs attributable to that phase.
12. **Ordered shard inventory:** exactly 66 records zipped to frozen prompt
    order. Each record binds canonical prompt ID, split, domain, tuning
    eligibility, token count, token-ID SHA-256, relative shard path, tensor name,
    dtype, exact shape, element count, raw byte count, whole-file SHA-256, raw
    tensor SHA-256, and contributing producer-phase IDs.
13. **Completion and release state:** `payload_integrity_pass`,
    `all_producer_memory_counters_known`, `all_producer_memory_clean`,
    `system_wired_default`, and `release_eligible` as five separate booleans.
14. **Stable content identities:** a cache-content SHA-256 derived from the
    ordered pinned identities, execution/precision contract, totals, producer
    phase identities, and each ordered shard's whole-file and raw-tensor
    SHA-256; and a manifest-body SHA-256 computed over the canonical manifest
    with only that self-digest field omitted.

The manifest's presence is the completion marker, not proof that its claims are
true. Only the independent strict auditor may establish validity.

## 5. Why the Air `teacher_cache.py` contract is not reused

The generic Air contract shall remain compatible and unchanged. It is not the
canonical GLM-5.2 cache contract because the live implementation has all of
these disqualifying properties:

1. `ALLOWED_TEACHER_KINDS` excludes the required deterministically dequantized
   ModelOpt source kind.
2. Its full-logit writer converts selected logits to FP16.
3. Its row schema and validator require `topk_ids`, `topk_logprobs`,
   `target_logprobs`, and `teacher_top1_ids`; those side tensors are not part of
   the canonical one-tensor GLM-5.2 shard.
4. It accepts multiple floating dtypes and treats any positive second logits
   dimension as a usable vocabulary width instead of requiring exactly F32 and
   154,880.
5. Its default validation checks headers with `check_values=false`; it does not
   establish the exact tree, exact physical extent, file/raw hashes, or full
   115,230,720-value scan required by this release trust boundary.

A new module, `src/mlx_vq/quality/glm52_teacher_cache.py`, shall own the exact
GLM-5.2 writer/reader/auditor contract. Implementation shall not weaken or
silently reinterpret `src/mlx_vq/quality/teacher_cache.py`.

## 6. Atomicity, resume, and memory-evidence design

### 6.1 Locking and state separation

The producer shall hold both:

- the repository's established global `.keep-heavy-job.lock`; and
- a cache/run-specific exclusive lock outside the final exact artifact tree.

No source load, checkpoint publication, shard publication, or manifest
publication may occur without both locks. Resumable state, ledger, checkpoints,
temporary files, logs, and locks shall be outside `<cache-root>` so the strict
final-tree contract remains exact.

### 6.2 Layer checkpoints and chain validation

The producer shall write immutable BF16 checkpoints
`checkpoints/layer-00000.safetensors` through
`checkpoints/layer-00077.safetensors`. Each checkpoint represents the hidden
state after its numbered main decoder layer and carries:

- hidden tensor `[66,18,6144]` in BF16;
- optional IndexShare state, explicitly recorded absent for this frozen run;
- layer identity and exact layer number;
- valid-position-mask hash;
- prompt, source, profile, config, policy, and precision identities;
- the previous checkpoint hash, with layer 0 bound to an authenticated genesis
  identity covering the embedded input state.

Resume is permitted only from the highest contiguous checkpoint whose complete
hash chain and identities validate. Orphan temporary files are non-authoritative.
A malformed, identity-mismatched, or hash-invalid final checkpoint fails
loudly; the producer shall not silently fall back to an earlier checkpoint or
overwrite the bad checkpoint.

### 6.3 Durable publication protocol

Every checkpoint and prompt shard shall use the same protocol:

1. create a unique sibling temporary file;
2. write the complete bytes;
3. flush userspace buffers and `fsync` the file;
4. atomically rename to the final filename;
5. `fsync` the containing directory;
6. only then append and durably publish the corresponding authenticated ledger
   record.

An existing canonical shard may be reused only when its authenticated ledger
record, all bound identities, safetensors header, physical bytes, whole-file
hash, and raw-tensor hash validate. Invalid or unledgered final shards fail
loudly and are not overwritten. The final manifest is written last through the
same durable protocol; its presence is the only completion marker.

### 6.4 Producer-phase memory evidence

Because computation is layer-major, pageout/swapout observations are attributable
to ordered producer phases, not individual prompt shards. The manifest shall
record the phases and let each prompt row reference every contributing phase.
These booleans remain independent:

```text
payload_integrity_pass
all_producer_memory_counters_known
all_producer_memory_clean
system_wired_default
release_eligible
```

Unknown, unavailable, malformed, or boolean counters shall never collapse to
integer zero. Structural payload validity may remain useful diagnostic evidence
after a memory-dirty run, but `release_eligible=true` requires known integer-zero
pageout and swapout deltas for every contributing phase under the system-default
wired-memory policy, in addition to payload integrity.

## 7. Strict cache auditor requirements

The auditor shall treat the cache root, manifest, shard names, headers, metadata,
hashes, counters, and booleans as untrusted and recompute truth from the raw
root. It shall:

- reject symlinked roots, files, or directories; path escapes; absolute paths;
  backslashes; `.` or `..` components; special files; missing files; and extras;
- reject duplicate JSON keys and `NaN`/`Infinity` in both the manifest and
  safetensors headers;
- require the exact manifest and nested-record field inventories;
- require exactly one `logits` tensor per shard, no uncontrolled metadata,
  exact F32 dtype, exact shape, exact offsets, no gaps or overlap, exact physical
  extent, and no trailing bytes;
- recompute every whole-file and raw-tensor SHA-256;
- stream-scan all 115,230,720 FP32 values for finiteness and reject constant
  causal rows;
- zip manifest rows to the validated frozen prompt order and require exact
  prompt IDs, token counts, and token-ID hashes;
- reconcile row, split, global element, and raw-byte totals to every frozen
  constant in Section 2.2;
- independently recompute the stable cache-content and manifest-body identities;
- use no-follow opens and before/after file identity checks to detect mutation
  during audit; and
- after traversal, re-open, re-read, and re-verify the manifest identity so a
  concurrent replacement cannot validate.

The auditor shall return invalid evidence on the first trust-boundary breach or
an exhaustive deterministic error inventory, but it shall never downgrade a
breach to a warning or trust a producer-supplied `audit_pass` value.

## 8. Build and family-gate integration

1. Add a new external input kind `teacher_cache_artifact` in
   `src/mlx_vq/build/recipe.py`. Preserve legacy `teacher_cache`, whose current
   meaning is a metadata file.
2. Add a strong directory-content hash in `src/mlx_vq/build/hashing.py`. It shall
   stream the manifest and actual shard bytes or incorporate shard hashes that
   were independently recomputed from those bytes. Manifest bytes plus
   names/sizes is insufficient because it misses same-size shard mutation.
3. Register GLM-5.2 producer and auditor operations in
   `src/mlx_vq/build/ops.py`. The producer is artifact-producing and therefore
   uses the executor's heavy-job lock path. The auditor consumes the raw cache
   root and emits evidence without trusting a prepared summary.
4. Preserve the existing negative teacher-metadata authority unchanged; it
   truthfully records that no cache exists in schema v2.
5. Extend `benchmarks/check_glm52_family_gate.py` with raw cache-root and
   manifest inputs. The checker shall call the strict validator directly and
   shall never accept a prepared `audit_pass` boolean or self-rehashed summary
   as cache proof.
6. Add gate schema v3 in `src/mlx_vq/quality/glm52_family.py` while preserving
   historical v1/v2 composition and validation behavior. Schema v3 requires the
   complete current schema-v2 raw trio: non-VQ package evidence, composite
   artifact audit, and production generation evidence.
7. Extend `src/mlx_vq/build/gates.py` with an exact v3 blocked contract while
   preserving the exact v1/v2 blocked contracts. V3 changes the teacher-cache
   check to true only after direct raw audit and requires the cache's full
   authenticated identity and release-eligible memory state.
8. Cross-bind the audited cache identity into every later raw candidate-eval
   row. A candidate row for another cache identity is invalid, not merely
   incomparable.

The direct checker exit semantics are:

| Input state | Required result |
|---|---|
| No cache supplied | Existing schema-v2 valid blocked checkpoint; exit 2 |
| Corrupt, unauthenticated, identity-mismatched, or memory-dirty release cache supplied | Invalid evidence; exit 1 |
| Fully audited, release-eligible clean cache supplied | Schema-v3 valid blocked checkpoint; exit 2 |

The clean-cache v3 checkpoint removes only
`dequantized_source_teacher_cache_payload`. The remaining blockers are
full-vocabulary source-relative evaluation, route/math diagnostics, and the
same-machine pinned-FP4 benchmark. Cache completion alone shall not remove the
high-level compiler guard or mark the family gate passed.

## 9. Testing strategy: required 14-step proof ladder

Use TDD and progress in this exact order:

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

Cheap structural and synthetic gates must pass before any full source pass.
Failure at any rung blocks all more expensive rungs that depend on it. A
tolerance-only result cannot replace a byte-identity requirement.

## 10. Rejected alternatives

- **Prompt-major sequential source forward:** This offers simple prompt-level
  resume but may reread most routed source weights up to 66 times. It is
  unacceptable for the full cache.
- **Dense decoded routed layer or model:** Holding all 168 experts for one sparse
  layer is roughly 25.37 GB of decoded routed weights, creates dangerous copy
  peaks, and violates the no-dense-checkpoint guard.
- **External vLLM/TP8 native W4A4:** This requires unavailable and unapproved
  external hardware/spend and changes the reference semantics toward
  activation-quantized W4A4 instead of the selected deterministic
  dequantized-weight reference.

## 11. Security, trust boundaries, and failure semantics

### 11.1 Trust boundaries

- The only source authority is the pinned model ID and immutable revision with
  the authenticated profile, config, index, decoder, source audit, and payload
  audit identities. Network resolution, symbolic revisions, or ambient cache
  contents cannot replace them.
- Prompt text is not an input authority during production. Only the frozen,
  authenticated ordered token rows are authoritative.
- The non-VQ package, checkpoint chain, producer ledger, final manifest, and
  every shard remain untrusted until independently validated against their raw
  bytes and cross-bound identities.
- The final cache root is data, never executable configuration. Paths from it
  must not select modules, commands, or arbitrary output locations.
- The manifest is an index and completion marker, not a root of trust. Its
  hashes, booleans, counters, and file inventory are claims to recompute.
- Filesystem traversal is hostile: no symlinks, following links, path escape,
  special files, mutation during read, or unexpected entries are accepted.
- Producer and auditor roles are logically separate. Family-gate composition
  calls the auditor on the raw root and does not accept producer-authored audit
  summaries.
- Holdout rows cannot influence implementation choices, tuning, selection, or
  recovery. Only selection rows may guide recovery.
- No external hardware, paid service, peer host, Thunderbolt/RDMA/JACCL work,
  upload, or publication is part of this design.

### 11.2 Failure semantics

- **Identity or schema mismatch:** fail before source compute where possible;
  publish neither a shard nor a manifest; direct gate evidence is invalid.
- **Lock failure or concurrent producer:** do not proceed without exclusive
  ownership; never allow two producers to publish into the same run/cache.
- **Crash before rename:** the unique temporary is orphaned and
  non-authoritative; a later run may remove it only under its explicit temp-file
  ownership rules.
- **Crash after rename but before ledger append:** the final file is unledgered
  and therefore fails loudly; it is not silently adopted or overwritten.
- **Malformed or tampered checkpoint:** fail loudly at the highest observed bad
  final checkpoint; do not fall back to an earlier chain element.
- **Missing manifest:** the cache is incomplete even if all 66 shards appear to
  exist.
- **Invalid or extra final-tree entry:** the entire cache is invalid; the
  auditor and direct checker return invalid evidence.
- **Non-finite or constant causal row:** the entire cache is invalid. Partial
  valid rows do not create a release cache.
- **Unknown memory counter:** preserve unknown as unknown. It may coexist with
  structurally valid diagnostic evidence but forces
  `all_producer_memory_counters_known=false` and `release_eligible=false`.
- **Known nonzero pageout/swapout delta or non-default wired policy:** preserve
  payload-integrity evidence if it is otherwise valid, but the supplied release
  cache is memory-dirty and family-gate integration returns invalid evidence,
  exit 1.
- **Clean cache with other family blockers:** emit a valid schema-v3 blocked
  checkpoint, exit 2. Never relabel it as a passing release gate.

## 12. Verified live-code anchors

These anchors were verified against the checkout used to write this spec. The
two `.venv` anchors identify the locked `mlx-lm==0.31.3` implementation recorded
in `uv.lock:390-404`; they are dependency behavior that the new implementation
must cover with tests rather than assume indefinitely.

| Verified claim | Live anchor |
|---|---|
| Pinned model/revision, vocabulary 154,880, and current v1/v2 gate schema constants | `src/mlx_vq/quality/glm52_family.py:18-36` |
| Frozen teacher claim booleans and full-vocabulary policy | `src/mlx_vq/quality/glm52_family.py:1249-1277` |
| Frozen prompt pack authenticates content, requires exactly 66 ordered rows, validates token hashes/counts, and fixes split/domain inventories | `src/mlx_vq/quality/glm52_family.py:1455-1580` |
| Profile fixes 78 layers, sparse layers 3--77, hidden size 6,144, 168 routed experts, 8 experts/token, and vocabulary 154,880 | `models/glm52-reap-504b-v2.yaml:1-13` |
| Adapter constructs exactly `num_hidden_layers` layers, preserves final norm and LM head | `src/mlx_vq/models/glm52_vq_adapter.py:497-578` |
| Sparse-layer selection is profile/config-driven and MTP layer 78 is outside `num_hidden_layers` | `src/mlx_vq/models/glm52_vq_adapter.py:266-289` |
| Sparse MoE owns the existing gate/shared expert, requires bound `switch_mlp`, and applies score-weighted route-rank sum before the shared expert | `src/mlx_vq/models/glm52_vq_adapter.py:427-467` |
| Strict binding enumerates sparse layers only within `range(num_hidden_layers)` | `src/mlx_vq/models/glm52_vq_adapter.py:662-725` |
| `create_causal_mask` treats `right_padding` as a pad count through `(offset + N) - right_padding` | `.venv/lib/python3.14/site-packages/mlx_lm/models/base.py:24-42` |
| Indexer returns `None` when key length is at most `index_topk` | `.venv/lib/python3.14/site-packages/mlx_lm/models/deepseek_v32.py:55-113` |
| Router performs gate matmul, FP32 sigmoid, correction, top-k, normalization, and scaling | `.venv/lib/python3.14/site-packages/mlx_lm/models/deepseek_v32.py:282-340` |
| ModelOpt decoder allocates C-contiguous FP32 output, decodes low/high nibbles, applies block/global scales, and rejects non-finite output | `src/mlx_vq/convert/nvfp4.py:228-247` |
| ModelOpt bundle reader independently reads weight, block-scale, and global-scale shards | `src/mlx_vq/convert/nvfp4.py:257-277` |
| Air allowed kinds exclude dequantized ModelOpt source | `src/mlx_vq/quality/teacher_cache.py:17-17` and `src/mlx_vq/quality/teacher_cache.py:568-578` |
| Air validator accepts several float dtypes and does not require exact vocabulary width | `src/mlx_vq/quality/teacher_cache.py:104-118` and `src/mlx_vq/quality/teacher_cache.py:623-652` |
| Air validator requires top-k/logprob/top1 side tensors | `src/mlx_vq/quality/teacher_cache.py:654-744` |
| Air full-logit writer downcasts to FP16 | `src/mlx_vq/quality/teacher_cache.py:1061-1135` |
| Air default validation is header-oriented and defaults `check_values=false` | `src/mlx_vq/quality/teacher_cache.py:905-965` |
| Recipe kinds currently distinguish directory artifacts from file-based legacy `teacher_cache` | `src/mlx_vq/build/recipe.py:25-33` and `src/mlx_vq/build/recipe.py:163-184` |
| Current artifact directory hash covers manifest bytes plus names/types/sizes, not shard bytes | `src/mlx_vq/build/hashing.py:57-78` |
| Operation registry exposes artifact outputs and manifest completion | `src/mlx_vq/build/ops.py:45-112` and `src/mlx_vq/build/ops.py:207-212` |
| Artifact-producing operations automatically acquire the global heavy-job lock | `src/mlx_vq/build/executor.py:139-156` and `src/mlx_vq/build/executor.py:449-463` |
| Current GLM-5.2 ops expose negative teacher metadata and the v1 family-gate inputs, but no FP32 cache producer/auditor inputs yet | `src/mlx_vq/build/ops.py:1314-1397` |
| Current checker requires the complete schema-v2 raw trio and hard-codes teacher cache readiness false | `benchmarks/check_glm52_family_gate.py:346-376` and `benchmarks/check_glm52_family_gate.py:428-470` |
| Current checker emits v1/v2 and returns 1 for invalid input, 2 for valid blocked evidence | `benchmarks/check_glm52_family_gate.py:474-559` and `benchmarks/check_glm52_family_gate.py:680-743` |
| Build gate accepts only exact v1/v2 blocked contracts and requires teacher-cache readiness false | `src/mlx_vq/build/gates.py:352-378` and `src/mlx_vq/build/gates.py:460-506` |

No live-code discrepancy was found with the approved architecture. The current
code intentionally lacks the new GLM-5.2 cache module, strong cache-directory
hash, producer/auditor ops, raw cache checker inputs, and gate schema v3; those
are implementation work defined by this specification, not contradictions.
