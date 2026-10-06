Read-only recon complete. No files were modified. The existing worktree already had unrelated untracked files; they were left untouched.

## Executive porting decision

The numerical low-rank machinery is already family-neutral. The GLM52 port should reuse:

- `QuantizedVQSwitchLinear` low-rank fields and runtime math.
- `RouteLocalSwitchLinearSurrogate`.
- The KL, target-NLL, teacher-top1-margin, tail-KLD, initialization, clipping, and manual-SGD helpers.
- The continuous-sidecar safetensors tensor layout.

The parts requiring GLM52-specific work are:

1. Teacher-cache ingestion: GLM52 uses one authenticated manifest plus one safetensors shard per prompt, not Air JSONL rows.
2. Selected-layer forward: GLM52 decoder layers return `(hidden, topk_indices)` and use MLA/IndexShare state, unlike Air’s single-tensor GQA-style layers.
3. Artifact authority: current GLM52 recovery candidates authenticate replacement routed-group files. A continuous adapter must be a separately authenticated child overlay; it cannot simply be dropped into the existing recovery directory.
4. Router training/binding: current “router KD” is heuristic bias materialization, not gradient distillation. GLM52 has no Air-style router-correction binder.

---

# 1. Air harness contract

Primary source: [finetune_glm45_air_vq_continuous.py](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:38).

## Data objects and cache loading

The training row contract is:

```python
@dataclass(frozen=True)
class PreparedTeacherRow:
    row_index: int
    prompt_id: str
    input_token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    teacher_logits: mx.array
```

Defined at [lines 38–45](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:38).

Air cache loading is:

```python
def _prepare_teacher_rows(
    *,
    teacher_jsonl: Path,
    cache_root: Path,
    max_rows: int | None,
    max_positions: int | None,
    row_indices: tuple[int, ...] | None = None,
) -> list[PreparedTeacherRow]
```

At [lines 210–254](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:210). It:

- Reads JSONL metadata with `read_teacher_cache_rows`.
- Requires `full_logits_available is True`.
- Reads `input_token_ids`, `positions`, and `target_token_ids`.
- Resolves a cache-relative `logit_shard`, rejecting absolute and escaping paths at [lines 194–207](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:194).
- Loads `logit_tensor`, default `"logits"`, and casts selected rows to FP32.
- Applies `max_positions` as a prefix, not arbitrary position selection.

Both selection and validation caches are validated as complete full-logit caches at [lines 257–276](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:257). Training can use `selection`, `validation`, or an interleaving of both at [lines 1136–1159](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:1136).

For GLM52, preserve the default as selection-only. The GLM52 authority expressly forbids report/holdout tuning.

## Loss function

The complete signature is:

```python
def _kl_loss(
    model,
    row: PreparedTeacherRow,
    *,
    target_nll_weight: float = 0.0,
    teacher_top1_margin_weight: float = 0.0,
    teacher_top1_margin: float = 0.0,
    teacher_top1_competitor_token_ids:
        tuple[tuple[int, ...], ...] | None = None,
    teacher_top1_include_hardest_competitor: bool = False,
    tail_kld_weight: float = 0.0,
    aux_loss_position_indices: tuple[int, ...] | None = None,
    loss_scope:
        Literal["full_model", "final_layer_selected", "selected_layer"]
        = "full_model",
    layer: int = 45,
    surrogate_projections: frozenset[str] = frozenset(),
    surrogate_output_chunk_size: int = 256,
    final_layer_prefix_cache: FinalLayerPrefixCache | None = None,
    teacher_log_probs: mx.array | None = None,
    teacher_probs: mx.array | None = None,
) -> mx.array
```

At [lines 623–641](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:623).

The base loss is mean full-vocabulary teacher-to-student KL:

```python
mean(sum(teacher_probs * (teacher_log_probs - vq_log_probs), axis=-1))
```

At [lines 680–681](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:680).

Target NLL is additively balanced, not normalized against KL:

```python
loss += target_nll_weight * -mean(student_log_prob[target_token])
```

At [lines 682–685](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:682). Therefore changing the number of positions does not change the per-term mean convention, but changing `target_nll_weight` directly changes its scale relative to KL.

The teacher-top1-margin loss is:

```python
def _teacher_top1_margin_loss(
    vq_logits: mx.array,
    teacher_logits: mx.array,
    *,
    margin: float,
    competitor_token_ids:
        tuple[int, ...] | tuple[tuple[int, ...], ...] | None = None,
    include_hardest_competitor: bool = False,
) -> mx.array
```

At [lines 742–749](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:742).

For each position it identifies the teacher’s top-1 token, then applies:

```text
max(student_competitor_logit - student_teacher_top1_logit + margin, 0)
```

The dynamic-hardest-competitor term is averaged over positions at [lines 754–764](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:754). Explicit competitor groups are instead accumulated with a sum over all supplied competitors at [lines 765–781](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:765). That different reduction is material when porting hyperparameters.

`aux_loss_position_indices` filters only the margin and tail-KLD terms; base KL and target NLL still use all selected positions, as implemented at [lines 686–710](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:686).

Tail loss is the maximum per-position full-vocabulary KL, not a percentile:

```python
def _max_token_kld_loss(...) -> mx.array:
    token_klds = ...
    return mx.max(token_klds)
```

At [lines 714–720](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:714).

## Selected-layer execution

Air has three paths:

- Full model: `_selected_logits`, [lines 306–309](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:306).
- Arbitrary selected sparse layer with downstream continuation: `_selected_logits_selected_layer`, [lines 467–513](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:467).
- Final-layer selected positions: `_selected_logits_final_layer`, [lines 516–552](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:516).

Frozen prefix activations use `mx.stop_gradient`. Final-layer prefixes can be cached once per row via `_cache_final_layer_training_row` at [lines 562–588](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:562).

## Trainable low-rank tensors

Initialization is at [lines 792–833](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:792):

```text
left:  [experts, output_dims, rank], initially zero
right: [experts, rank, input_dims], seeded normal(scale=low_rank_init_scale)
```

Existing `continuous_low_rank_left/right` are resumed unless `--reinitialize-existing-sidecars` is set.

Zero-left/random-right is intentional: the first gradient updates left while preserving an exactly zero initial residual.

Assignment calls:

```python
projection.set_continuous_sidecar(
    scale_delta=...,
    output_bias=...,
    low_rank_left=left,
    low_rank_right=right,
)
```

At [lines 862–884](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:862).

## Optimizer and loop

This is manual clipped SGD, not Adam/AdamW:

```python
updated = param - learning_rate * clipped_grad
```

At [lines 887–921](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:887).

Each step:

- Chooses `train_rows[step % len(train_rows)]`.
- Calls `mx.value_and_grad`.
- Clips each parameter tensor independently by its own norm.
- Rebinds updated sidecars.
- Evaluates loss, parameters, and norms.
- Appends a JSONL step record.
- Clears the MLX cache if available.

See [lines 1262–1316](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:1262).

There is no optimizer state and no intermediate resumable checkpoint. Only the final artifact is persisted. A crash retains JSONL evidence but not the last trained tensors.

## CLI surface

CLI arguments span [lines 930–1072](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:930). The port should mirror these names except where the GLM52 authority makes them invalid:

- Replace the two JSONL/root pairs with authenticated GLM52 manifest and prompt-pack inputs.
- Default `--layer` must not remain Air’s `41`.
- Remove or reject `--train-cache validation|both`; GLM52 training must be selection-only.
- Replace `--model-id` default and `load_resident_air` inputs with validated GLM52 baseline/recovery authority inputs.
- Keep `--projection[s]`, `--trainable`, low-rank, loss, clipping, surrogate, row/position limits, and ledger options.
- Rename Air record types.

---

# 2. `RouteLocalSwitchLinearSurrogate`

Exact class starts at [mlx_surrogate.py:198](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/mlx_surrogate.py:198).

Constructor contract:

```python
RouteLocalSwitchLinearSurrogate(
    *,
    input_dims: int,
    output_dims: int,
    num_experts: int,
    codes: mx.array,
    scales: mx.array,
    codebook: mx.array,
    group_size: int,
    code_bits: int,
    bias: mx.array | None = None,
    rht_signs: mx.array | None = None,
    sidecar: SwitchLinearSidecar | None = None,
    output_chunk_size: int = 256,
)
```

The preferred entry point is:

```python
@classmethod
def from_layer(
    cls,
    layer: QuantizedVQSwitchLinear,
    *,
    sidecar: SwitchLinearSidecar | None = None,
    output_chunk_size: int = 256,
) -> RouteLocalSwitchLinearSurrogate
```

At [lines 278–301](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/mlx_surrogate.py:278). It rejects sparse residual rows because those are not represented in the surrogate.

It avoids the native gather/custom-kernel VJP problem by expressing the routed projection entirely in ordinary MLX operations:

- Select only routed experts with `mx.take`.
- Decode only output chunks.
- Expand group scales over codewords.
- Multiply decoded vectors, inputs, and scales.
- Reduce with `mx.sum`.
- Add sidecar bias and low-rank residual.

The route-local core is [lines 331–365](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/mlx_surrogate.py:331). This avoids decoding `[all_experts, all_outputs, input]` while retaining an autodiff graph.

It supports both:

```text
token input: x [..., input], indices [..., top_k]
route input: x [..., top_k, input], indices [..., top_k]
```

at [lines 303–329](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/mlx_surrogate.py:303).

`down_proj` is not special inside the surrogate. The Air harness makes it consume the post-SwiGLU routed activation:

```python
hidden = nn.silu(gate) * up
down_proj(hidden, indices)
```

at [finetune lines 320–344](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:320). Therefore:

- Gate/up input dimension is model hidden size.
- Down input dimension is MoE intermediate size.
- Down output dimension is model hidden size.
- The same per-expert low-rank equation applies.

The low-rank correction itself is:

```text
latent     = right[expert] @ x
correction = left[expert] @ latent
y          = y_base + correction
```

at [mlx_surrogate.py:357–361](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/mlx_surrogate.py:357).

No GLM52-specific surrogate class is necessary.

---

# 3. Save and bind contract

## Runtime fields

`QuantizedVQSwitchLinear.__init__` accepts:

```python
continuous_low_rank_left: mx.array | None = None
continuous_low_rank_right: mx.array | None = None
```

at [switch_linear.py:92–119](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:92).

Both-or-neither and dimensional validation occur at [lines 152–161](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:152). They are stored as FP32 at [lines 192–198](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:192).

The mutating binder is:

```python
def set_continuous_sidecar(
    self,
    *,
    scale_delta: mx.array | None = None,
    output_bias: mx.array | None = None,
    low_rank_left: mx.array | None = None,
    low_rank_right: mx.array | None = None,
) -> None
```

At [lines 229–272](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:229).

Runtime application is at [lines 427–454](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:427). It supports both token/top-k and per-route input layouts.

## Sidecar artifact format

The artifact API is in [continuous_sidecar.py](/Users/jack.mazac/Developer/keep/src/mlx_vq/io/continuous_sidecar.py:13).

Per projection:

```text
continuous_params/layer-{layer:05d}-{projection}.safetensors
```

The safetensors file contains any of:

- `scale_delta`
- `output_bias`
- `low_rank_left`
- `low_rank_right`

Low-rank tensors are saved as FP32. Writer signature and metadata are at [lines 161–223](/Users/jack.mazac/Developer/keep/src/mlx_vq/io/continuous_sidecar.py:161).

The derived `conversion-manifest.json` contains:

```json
{
  "continuous_parameters": {
    "schema_version": 1,
    "enabled": true,
    "format": "switch_linear_scale_delta_output_bias",
    "seed_artifact_dir": "...",
    "sidecars": [...]
  }
}
```

At [lines 226–250](/Users/jack.mazac/Developer/keep/src/mlx_vq/io/continuous_sidecar.py:226). The format string predates low-rank tensors and is semantically stale, although the entries explicitly list the low-rank tensor names and shapes.

Air materialization:

- Symlinks seed `layer-*.safetensors`.
- Copies non-target continuous sidecars.
- Copies declared final-logit bias.
- Writes replacement target sidecars.
- Writes a derived conversion manifest.

See [finetune lines 1318–1402](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:1318).

Air binds sidecars during `load_glm45_air_vq_switch_glu`: load projection, call `load_switch_linear_continuous_sidecar`, then `set_continuous_sidecar`, at [glm45_air_vq_adapter.py:514–568](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm45_air_vq_adapter.py:514).

GLM52 currently does not do this. `load_glm52_vq_switch_glu[_from_paths]` only loads the three routed groups at [glm52_vq_adapter.py:581–628](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm52_vq_adapter.py:581).

---

# 4. Router-distillation surfaces

## Current “router KD”

The build operation named `train-router-kd` invokes:

```text
benchmarks/materialize_glm45_air_router_correction.py
```

not a gradient trainer. See [build/ops.py:2717–2736](/Users/jack.mazac/Developer/keep/src/mlx_vq/build/ops.py:2717).

It consumes route-disagreement JSON and writes an expert-bias vector. `materialize_router_correction` defaults to 128 experts at [materializer lines 69–77](/Users/jack.mazac/Developer/keep/benchmarks/materialize_glm45_air_router_correction.py:69), with a hardcoded 128 in automatic planning at [plan_next.py:750–765](/Users/jack.mazac/Developer/keep/src/mlx_vq/build/plan_next.py:750).

The sidecar contract is:

```python
@dataclass(frozen=True)
class RouterCorrectionSidecar:
    expert_bias_delta: mx.array | None = None  # [experts]
    temperature: mx.array | None = None        # scalar or [1]
```

At [router_correction.py:16–23](/Users/jack.mazac/Developer/keep/src/mlx_vq/io/router_correction.py:16). Load/write signatures are at [lines 41–78](/Users/jack.mazac/Developer/keep/src/mlx_vq/io/router_correction.py:41) and [81–128](/Users/jack.mazac/Developer/keep/src/mlx_vq/io/router_correction.py:81).

Air applies temperature before sigmoid routing and adds `expert_bias_delta` to `gate.e_score_correction_bias`, at [glm45_air_vq_adapter.py:144–182](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm45_air_vq_adapter.py:144).

## GLM52 router representation

GLM52’s MoE owns `self.gate = mlx_lm.models.deepseek_v32.MoEGate(config)` at [glm52_vq_adapter.py:427–440](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm52_vq_adapter.py:427).

Its source/non-VQ parameters include:

```text
model.layers.{layer}.mlp.gate.weight
model.layers.{layer}.mlp.gate.e_score_correction_bias
```

`e_score_correction_bias` is deliberately excluded from dtype casting so it remains FP32 at [glm52_vq_adapter.py:560–565](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm52_vq_adapter.py:560). Validation constructs the correction tensor explicitly at [glm52_vq.py:237–263](/Users/jack.mazac/Developer/keep/src/mlx_vq/validate/glm52_vq.py:237).

Training router surfaces would touch:

- `gate.weight`: `[168, 6144]` per sparse layer.
- `gate.e_score_correction_bias`: `[168]` per sparse layer.
- An optional new scalar temperature, because GLM52 currently has no temperature field or correction wrapper.
- Routing indices/scores, hence downstream expert selection and all three expert projections.

A genuine router KD loss should operate on pre-selection routing logits or sigmoid scores. Distilling only selected indices is nondifferentiable and insufficient. It should separately report:

- Router distribution KL at a configurable temperature.
- Top-k set agreement.
- Route-score error on the union of teacher/student top-k.
- Load-balancing/expert-usage drift.

Do not call the current bias materializer gradient router KD; it is a route-trace-derived correction heuristic.

---

# 5. GLM52 teacher/model/authentication equivalents

## Teacher cache

Artifact root:

[glm52-teacher-cache-fp32-v5-20260710](/Users/jack.mazac/Developer/keep/artifacts/quality/glm52-teacher-cache-fp32-v5-20260710).

Its manifest declares 66 safetensors shards, each containing one tensor named `logits`, FP32, shape:

```text
[token_count - 1, 154880]
```

The exact shard contract is at [glm52_teacher_cache.py:950–994](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/glm52_teacher_cache.py:950) and validated at [lines 1164–1188](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/glm52_teacher_cache.py:1164).

It stores:

- Full-vocabulary logits only.
- No hidden states.
- No router logits/scores/indices.
- No token IDs inside each shard.
- Token count and token-ID SHA in the manifest.

Actual token IDs come from the authenticated prompt pack represented by:

```python
@dataclass(frozen=True)
class GLM52TeacherCachePrompt:
    prompt_id: str
    split: str
    domain: str
    tuning_eligible: bool
    encoded_token_ids: tuple[int, ...]
```

At [lines 368–408](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/glm52_teacher_cache.py:368).

For language-model next-token training:

```text
input_token_ids = encoded_token_ids
positions       = range(token_count - 1)
target_token_ids = encoded_token_ids[1:]
teacher_logits.shape = [token_count - 1, vocab]
```

The selection authority is exactly:

- 22 prompts.
- 255 prediction positions.
- 8 route prompts / 96 positions.
- 7 math prompts / 80 positions.
- 7 instruction prompts / 79 positions.
- `tuning_eligible=True`.
- Report and holdout forbidden for tuning.

The strict 22/255 validator is [glm52_recovery.py:24–69](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/glm52_recovery.py:24). The frozen global counts are enforced at [glm52_teacher_cache.py:830–871](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/glm52_teacher_cache.py:830).

The inspected v5 manifest is `release_eligible=false`; any training run using it must preserve that diagnostic/non-release status rather than promoting the output.

## Model structure

`GLM52VQModel` is at [glm52_vq_adapter.py:548–578](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm52_vq_adapter.py:548).

Relevant constants:

- 78 main decoder layers: `0..77`.
- MTP layer 78 is excluded.
- Sparse layers: `3..77`, 75 layers.
- 168 routed experts.
- Top-8 experts per token.
- Hidden size 6144.
- MoE intermediate size 2048.

Sparse-layer selection is centralized in:

```python
def is_glm52_sparse_layer(
    config: GLM52VQModelArgs,
    layer_idx: int,
) -> bool
```

At [lines 266–273](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm52_vq_adapter.py:266).

`Glm52VQMoE` binds the same `QuantizedVQSwitchGLU`, so projection-side numerical code is reusable unchanged. See [lines 427–467](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm52_vq_adapter.py:427).

## Authenticated loading

Baseline production loading authenticates inputs, binds non-VQ weights, binds exactly sparse layers `3..77`, and rechecks authority between phases at [glm52_composite_loader.py:1350–1417](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm52_composite_loader.py:1350).

Current recovery validation signature is:

```python
def validate_glm52_recovery_candidate_inputs(
    baseline: GLM52ValidatedProductionInputs,
    *,
    recovery_dir: str | Path,
    expected_seed_manifest_sha256: str,
    expected_stats_manifest_sha256: str,
    expected_full_source_blob_inventory_sha256: str,
    expected_routed_source_blob_inventory_sha256: str,
    expected_recovery_lever: str,
    expected_recovery_policy: Mapping[str, str],
    accepted_composite_audit_json: str | Path,
    expected_composite_audit_sha256: str,
    expected_recovery_candidate_identity_sha256: str,
    expected_recovery_manifest_body_sha256: str,
    ...
) -> GLM52ValidatedRecoveryCandidate
```

At [lines 1186–1206](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm52_composite_loader.py:1186).

Loading snapshots the exact 225 audited routed-group files and binds from authenticated paths at [lines 1420–1516](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm52_composite_loader.py:1420).

A low-rank adapter is additive and does not replace those 225 files. Therefore it needs a new authenticated overlay identity; extending the existing recovery manifest with undeclared sidecar files would either leave them unauthenticated or violate exact artifact-tree/accounting semantics.

---

# 6. Required GLM52 porting contract

## A. Training module

Create:

```text
benchmarks/finetune_glm52_vq_continuous.py
```

Prefer extracting family-neutral helpers into:

```text
src/mlx_vq/quality/continuous_distillation.py
```

Required GLM52-specific interfaces:

```python
@dataclass(frozen=True)
class PreparedGLM52TeacherRow:
    row_index: int
    prompt_id: str
    input_token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    teacher_logits: mx.array
    teacher_manifest_body_sha256: str
    teacher_shard_sha256: str
```

```python
def prepare_glm52_teacher_rows(
    *,
    teacher_cache_dir: Path,
    prompt_pack_path: Path,
    split: Literal["selection"] = "selection",
    max_rows: int | None,
    max_positions: int | None,
    row_indices: tuple[int, ...] | None = None,
    allow_non_release_teacher_cache: bool = False,
) -> list[PreparedGLM52TeacherRow]
```

This must:

- Authenticate/audit the cache manifest and prompt pack.
- Join prompts and shards by the manifest’s ordered prompt IDs.
- Require selection-only and exact tuning eligibility.
- Verify token-ID SHA and shard SHA before use.
- Derive positions and targets as above.
- Record the non-release waiver if used.

```python
def target_glm52_projection(
    model: GLM52VQModel,
    *,
    layer: int,
    projection: Literal["gate_proj", "up_proj", "down_proj"],
) -> QuantizedVQSwitchLinear
```

It must require `isinstance(layer.mlp, Glm52VQMoE)` and a bound `switch_mlp`.

```python
def call_glm52_vq_moe_with_route_local_surrogates(
    moe: Glm52VQMoE,
    x: mx.array,
    *,
    layer: int,
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
) -> mx.array
```

Use `moe.gate(x)` followed by the unchanged switch-surrogate helper.

```python
def selected_glm52_logits_selected_layer(
    model: GLM52VQModel,
    row: PreparedGLM52TeacherRow,
    *,
    layer: int,
    surrogate_projections: frozenset[str],
    output_chunk_size: int,
) -> mx.array
```

This is the principal rewrite. It must carry `prev_topk_indices` through every GLM52 decoder layer and call attention as:

```python
attention, next_topk = layer_module.self_attn(
    layer_module.input_layernorm(h),
    mask,
    cache[layer_idx],
    prev_topk_indices,
)
```

Do not copy Air’s `layer_module.self_attn(...)->tensor` assumption.

For the final-layer cached path, the cache object must include the incoming IndexShare state:

```python
@dataclass(frozen=True)
class GLM52FinalLayerPrefixCache:
    row: PreparedGLM52TeacherRow
    hidden: mx.array
    attention_mask: mx.array | None
    prev_topk_indices: mx.array | None
    teacher_log_probs: mx.array
    teacher_probs: mx.array
```

## B. Authenticated adapter artifact

Create a separate overlay contract, for example:

```text
src/mlx_vq/quality/glm52_continuous_adapter.py
src/mlx_vq/validate/glm52_continuous_adapter_artifact.py
```

Required identity object:

```python
@dataclass(frozen=True, slots=True)
class GLM52ValidatedContinuousAdapter:
    baseline: GLM52ValidatedProductionInputs
    routed_candidate: GLM52ValidatedRecoveryCandidate | None
    adapter_dir: Path
    adapter_manifest_body_sha256: str
    adapter_candidate_identity_sha256: str
    sidecar_paths: Mapping[tuple[int, str], Path]
```

Validation signature:

```python
def validate_glm52_continuous_adapter_inputs(
    baseline: GLM52ValidatedProductionInputs,
    *,
    adapter_dir: str | Path,
    expected_parent_candidate_identity_sha256: str,
    expected_teacher_manifest_body_sha256: str,
    expected_adapter_manifest_body_sha256: str,
    expected_adapter_candidate_identity_sha256: str,
    routed_candidate: GLM52ValidatedRecoveryCandidate | None = None,
) -> GLM52ValidatedContinuousAdapter
```

The identity must bind:

- Accepted baseline composite identity.
- Optional recovery-candidate identity.
- Exact teacher cache manifest body and prompt authority.
- Training configuration and selection row/position set.
- Every sidecar file SHA, tensor name, dtype, and shape.
- Layer/projection uniqueness.
- Parent routed-group hashes.
- Non-release teacher status/waiver.
- Router sidecars, if any, as a separate typed collection.

## C. GLM52 binding

Add a helper around the existing routed-group loader:

```python
def bind_glm52_switch_glu_continuous_sidecars(
    switch_mlp: QuantizedVQSwitchGLU,
    *,
    layer: int,
    validated_adapter: GLM52ValidatedContinuousAdapter,
) -> None
```

Then add an authenticated composite loader rather than weakening the existing one:

```python
def load_authenticated_glm52_continuous_candidate(
    validated: GLM52ValidatedContinuousAdapter,
    *,
    snapshot_scratch_dir: str | Path,
    allow_snapshot_copy_fallback: bool = False,
    ...
) -> tuple[GLM52VQModel, GLM52ContinuousCandidateLoadReport]
```

Required order:

```text
verify baseline and parent recovery identity
snapshot authenticated routed groups and adapter sidecars
construct model
bind baseline non-VQ
verify all identities
bind exact routed groups
verify all identities
bind exact continuous sidecars
verify all identities
reject missing/unexpected sidecars
retain snapshots for model lifetime
```

Do not make the GLM52 binder read sidecars opportunistically from an unauthenticated `conversion-manifest.json`.

## D. Router KD

For a faithful port, create a separate trainer, not an option buried in projection training:

```python
def glm52_router_kd_loss(
    student_router_logits: mx.array,
    teacher_router_logits: mx.array,
    *,
    temperature: float,
    distribution_kld_weight: float,
    topk_score_weight: float,
) -> mx.array
```

But the current teacher cache lacks router logits. Genuine router KD therefore requires either:

- A new authenticated selection-only router-teacher cache, or
- A simultaneous authenticated source-teacher forward.

If retaining only the existing heuristic correction, port:

```python
def materialize_glm52_router_correction(
    *,
    disagreement_json: str | Path,
    parent_candidate: GLM52ValidatedContinuousAdapter | GLM52ValidatedRecoveryCandidate,
    output_dir: str | Path,
    num_experts: int = 168,
    scale: float = 1.0,
    max_abs_delta: float | None = None,
) -> dict[str, Any]
```

and add GLM52 runtime fields equivalent to Air’s `router_expert_bias_delta` and `router_temperature`.

---

# 7. Reuse versus rewrite

| Surface | Decision |
|---|---|
| `QuantizedVQSwitchLinear` low-rank tensors/math | Reuse unchanged |
| `SwitchLinearSidecar` tensor layout | Reuse |
| `RouteLocalSwitchLinearSurrogate` | Reuse unchanged |
| Projection surrogate composition: gate, up, SwiGLU, down | Reuse |
| `_teacher_distribution` | Reuse |
| `_teacher_top1_margin_loss` | Reuse, preserving explicit-group sum semantics |
| `_max_token_kld_loss` | Reuse |
| `_filter_logits_by_position_indices` | Reuse |
| Low-rank initialization/assignment | Reuse |
| Per-tensor gradient clipping/manual SGD | Reuse initially |
| Air JSONL cache loader | Rewrite |
| `load_resident_air` | Replace with authenticated GLM52 baseline/recovery load |
| `GLM45AirVQMoE` type checks | Rewrite for `Glm52VQMoE` |
| Selected-layer/final-layer forward | Rewrite for `(hidden, topk_indices)` and IndexShare |
| Air artifact symlinking/promotion | Replace with authenticated overlay candidate |
| Air sidecar opportunistic bind | Do not copy; authenticate before bind |
| Router correction runtime | Implement for GLM52 |
| Genuine router KD | New cache/capture and loss required |
| Build op hardcoded `num_experts=128` | Change to profile-derived 168 |
| Air JSONL record names/default layer/model ID | Rewrite |

---

# 8. Minimal synthetic end-to-end smoke test

Create one test such as:

```text
tests/test_glm52_continuous_distillation_smoke.py
```

It must use no source model and no real artifact.

Test contract:

1. Construct three tiny `QuantizedVQSwitchLinear` projections with:

```text
experts=2
top_k=1
hidden=8
moe_intermediate=8
group_size=8
code_bits=8
use_gather_vqmm=False
```

2. Bind them into `QuantizedVQSwitchGLU` and a fake `Glm52VQMoE`-compatible layer.
3. Use a fake GLM52 decoder whose attention returns:

```python
(attention_output, prev_topk_indices)
```

This specifically proves the GLM52 tuple/IndexShare continuation contract.
4. Create one `PreparedGLM52TeacherRow` with:

```text
sequence length=3
positions=(0, 1)
targets=(token1, token2)
teacher_logits=[2, tiny_vocab]
```

5. Initialize rank-2 residuals on gate/up/down.
6. Run exactly two `mx.value_and_grad` SGD steps through `selected_glm52_logits_selected_layer`.
7. Enable:

```text
target_nll_weight > 0
teacher_top1_margin_weight > 0
tail_kld_weight > 0
surrogate_projections = all three
```

8. Assert:

- Loss and all gradients are finite.
- At least one left tensor changes from zero.
- Parameter shapes remain exact.
- Output is `[positions, vocab]`.
- Route-local surrogate output matches the ordinary decoded surrogate within established tolerance.
- Saved safetensors contain FP32 `low_rank_left/right`.
- The authenticated synthetic adapter validator detects one-byte replacement.
- Fresh synthetic load binds the sidecar and reproduces the trained output.
- Removing either left or right is rejected.
- A sidecar declaring 128 experts against a 168-expert profile is rejected.
- Report/holdout rows are rejected for tuning.

This proves the training loop, GLM52 forward protocol, artifact round-trip, and binding without allocating a real 504B model. It does not prove Metal custom-kernel parity or production memory viability.

---

# 9. Open risks

1. **Teacher cache is diagnostic-only.** The inspected v5 cache has `release_eligible=false`. Outputs trained from it inherit a non-release evidence boundary unless a narrowly governed waiver is explicitly accepted.

2. **No router supervision exists in the cache.** Full-vocabulary logits can train expert sidecars end-to-end, but cannot directly supervise router logits. Genuine router KD needs new authenticated data.

3. **IndexShare propagation is the largest correctness trap.** Air forward helpers silently assume attention returns one tensor. GLM52 attention returns output plus top-k indices, and shared-indexer layers require prior indices at [glm52_vq_adapter.py:321–424](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm52_vq_adapter.py:321).

4. **Prefix caching may be expensive.** GLM52 hidden states are 6144-wide, and cached IndexShare state may add substantial memory. Cache only selected rows and measure wired memory before making it default.

5. **Adapter payload can be large.** Per projection, FP32 rank-\(r\) payload is:

```text
168 × r × (input_dim + output_dim) × 4 bytes
```

At rank 8 this is approximately:

- Gate/up: `168 × 8 × (6144 + 2048) × 4` ≈ 44 MiB each.
- Down: same total dimensions, ≈ 44 MiB.
- Roughly 132 MiB per fully adapted layer.
- Roughly 9.7 GiB across all 75 sparse layers.

The first port should target a bounded layer/projection set.

6. **Manual SGD has no resumable state.** A long GLM52 run needs authenticated periodic parameter checkpoints plus step/row cursor identity, or a failure loses all updates since the final artifact.

7. **Per-tensor clipping changes joint optimization.** Gate/up/down and left/right are clipped independently. Preserve this for parity first, but do not assume it is equivalent to global-norm clipping.

8. **Explicit margin competitors use a sum.** The magnitude scales with competitor count, unlike the averaged hardest-competitor term. Hyperparameters are not portable across different group sizes without adjustment.

9. **Existing GLM52 recovery loading is replacement-oriented.** Sidecars must not be smuggled through its 225-group identity. Use an additive authenticated child-candidate layer.

10. **Router bias is already non-VQ authority.** Updating `e_score_correction_bias` changes a source/non-VQ parameter. It must be represented as an explicit authenticated overlay rather than silently mutating the accepted non-VQ package.

11. **Air defaults are unsafe for GLM52.** Hardcoded assumptions include Air imports/type checks, default layer 41, 128 experts in router planning, Air record types, `load_resident_air`, and single-output attention. The complete hardcoded set is visible in the harness at imports [15–25](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:15), type checks [279–287](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:279), and CLI defaults [930–971](/Users/jack.mazac/Developer/keep/benchmarks/finetune_glm45_air_vq_continuous.py:930).

12. **Synthetic proof is not host proof.** The smoke test demonstrates differentiability and artifact semantics only. Production acceptance still requires authenticated cache/candidate evaluation and a Metal-capable memory-safe run.


Codex session ID: 019f547e-7393-7f80-905d-e137637104d0
Resume in Codex: codex resume 019f547e-7393-7f80-905d-e137637104d0
