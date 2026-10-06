# New Model Family Template For KEEP

Use this template before publishing a KEEP release candidate for a model family
other than GLM-4.5-Air. The goal is to make the new family reproducible through
the same public surfaces as the GLM-4.5-Air balanced RC: adapter, artifact,
audit, eval, benchmark, training/calibration, summary, and tests.

Copy the checklist into the new family doc or issue, replace every
`<family-name>` placeholder, and keep the command run status honest. If a command
is model-loading or hardware-heavy, mark it as not run in that slice and record
the evidence path that will satisfy it later.

## Family Identity

- Family name: `<family-name>`
- Source model path or repo: `<source-model-path>`
- VQ artifact root: `artifacts/<family-name>-vq`
- Protected seed artifact root, if continuing from a seed:
  `artifacts/<family-name>-seed`
- Public RC output root: `artifacts/rc/<family-name>-<preset>-<date>`
- Supported presets: `balanced`, `quality`, or `<custom-preset>`
- Dense or routed: `routed`, `dense`, or `hybrid`
- Primary comparison baseline: `<q-control-or-source-baseline>`

## Adapter Contract

Add or update a RAMP model adapter under `src/ramp/models/` while the legacy
implementation still lives under `src/mlx_vq/models/`.

- The adapter loads non-expert tensors from source shards without materializing
  routed dense expert tensors.
- Routed expert tensors are recognized by deterministic source-name patterns.
- MTP, speculative, or auxiliary layers are skipped or loaded intentionally.
- Sparse/routed layers are identified from the source config, not hard-coded
  only by artifact paths.
- The adapter validates hidden size, routed hidden size, expert count, and top-k
  before binding switch projections.
- The adapter supports the same projection call contract used by
  `QuantizedVQSwitchGLU`: token input shaped `[..., input_dim]` and selected
  expert indices shaped `[..., top_k]`.
- The adapter exposes enough metadata for benchmark and audit rows to report no
  dense routed experts and no unbound VQ experts.
- The family registers a `FamilyBinding` in `ramp.models.registry` and its
  converter kind via `mlx_vq.models.profiles.register_converter`, so nothing
  about the family has to be edited into a shared module.

Reference surfaces in the current GLM path:

```text
src/mlx_vq/models/glm45_air_vq_adapter.py
src/mlx_vq/models/glm4_moe_adapter.py
src/mlx_vq/nn/switch_linear.py
src/mlx_vq/io/load.py
```

Canonical imports should use `ramp.models`, `ramp.nn`, and `keep.io` for new
code. Existing `mlx_vq.*` imports remain supported as a compatibility shim.
`ramp.models.registry` is the target seam for new families, introduced
2026-08-11 with the DeepSeek-V4-Flash pivot: new family code lives in
`ramp.models` / `keep.convert` and registers itself there rather than being
edited into shared modules. It is a target, not a completed migration — the
`mlx_vq.models` export map is still the live wiring for the families that
predate it (GLM-4.5-Air, GLM-4-MoE, GLM-5.2, Qwen-MoE), and those keep
resolving through it until they are moved deliberately.

## Source Tensor Mapping

Document the source-to-artifact tensor mapping before writing artifacts.

- Source expert weight names for gate/up/down or the family equivalents:
  `<source-patterns>`
- Artifact projection keys:
  `<artifact-prefix>.gate_proj`, `<artifact-prefix>.up_proj`,
  `<artifact-prefix>.down_proj`, or `<dense-projection-name>`
- Layer index range and any dense-to-routed transition:
  `<layer-range-and-transition>`
- Expert count and top-k:
  `<expert-count>`, `<top-k>`
- Input and output dimensions for every projection class:
  `<projection-shapes>`
- Group size policy:
  `<group-size-policy>`
- Codebook family and code width:
  `E8`, `E8P`, or `<codebook>`
- Non-routed tensors that remain in source precision:
  embeddings, attention, norms, router, shared experts, LM head, or
  `<family-specific-list>`

## Manifest Requirements

Every public artifact must have manifest metadata that lets an audit reproduce
the artifact lineage.

- Artifact schema version and model family.
- Source model identity and source revision if available.
- Seed artifact path when the artifact is derived from another artifact.
- `seed_artifact_mutated=false` or equivalent proof.
- Projection count, layer count, expert count, code bits, group sizes, and
  codebook hashes.
- Symlinked versus rewritten projection counts.
- Precision-tier counts for mixed artifacts.
- Continuous sidecar manifest, if sidecars are present.
- Sparse residual or router-correction sidecar manifest, if present.
- Trainable surface and recipe metadata for sidecar-derived artifacts.
- Evidence paths for eval, benchmark, and audit rows used to select the RC.

## Artifact Audit

Add or extend an audit path that can run without model generation.

The audit must report:

- total routed projection count
- missing projection files
- unbound VQ experts
- dense routed expert fallbacks
- high-precision routed projection count
- code-bit and group-size distributions
- continuous sidecar count
- NAX/Metal compatibility or the family-specific fast-path equivalent
- fallback layers and reasons
- effective routed bits per weight

For routed MoE families, the audit should fail the RC gate if it cannot prove
`dense_routed_experts=false` and `unbound_vq_experts=false`.

## Eval Contract

Add a row-level eval wrapper before accepting a release candidate.

- Eval rows are JSONL objects.
- Every row records prompt ID, split, clean-memory counters, NLL/PPL metrics,
  KLD metrics, top-1 agreement, and token-level tails when a teacher is used.
- Report, selection, and holdout split names are stable and non-overlapping.
- Dirty rows are excluded from acceptance or explicitly marked as rejected
  evidence.
- Teacher-cache metadata records source model, prompt set, logit mode, and cache
  root.
- The summary path recomputes metrics from JSONL rows instead of trusting
  terminal output.

Public command shape:

```bash
uv run python benchmarks/eval_<family-name>_teacher_cache.py \
  --artifact-dir artifacts/<family-name>-vq \
  --teacher-jsonl artifacts/quality/<family-name>-teacher-cache-report/metadata.jsonl \
  --teacher-cache-root artifacts/quality/<family-name>-teacher-cache-report \
  --append-jsonl artifacts/quality/<family-name>-report.jsonl
```

Run status: placeholder command, not run until the family-specific wrapper
exists.

## Benchmark Contract

Add a benchmark wrapper that compares the VQ artifact to a relevant control.

- Fresh-process rows are preferred for resident benchmarks.
- Candidate and control rows use the same scenario and quiet-window discipline.
- Pageouts and swapouts are acceptance gates, not footnotes.
- Benchmark rows record effective bpw, dtype parity, dense routed expert status,
  unbound VQ expert status, and non-expert dtype status when applicable.
- The summary reports median, valid repetition count, invalid reasons, and the
  candidate/control ratio.
- A public family gate must name the comparison baseline. Placeholder baselines
  such as `source_or_control_runtime` are not enough for publication; the gate
  should state a same-machine reference and a maximum candidate/reference ratio.

Public command shape:

```bash
uv run python benchmarks/bench_<family-name>_quant_compare.py \
  --engine <family-name>_vq \
  --scenario prefill_1k \
  --repetitions 2 \
  --append-jsonl artifacts/benchmarks/<family-name>-prefill1k.jsonl
```

Run status: placeholder command, not run until the family-specific wrapper
exists.

### Qwen3.6 Hardened Gate Example

The Qwen3.6-35B-A3B recipe hardens this generic contract with:

- `22` prompt rows per `report`, `selection`, and `holdout` split (`66` rows
  total) and `minimum_total_clean_rows=64`.
- Source-teacher logits from the source model, not reused Air thresholds.
- Benchmark scenarios `prefill_1k` and `decode_128`, with `2` candidate/control
  repetitions per scenario.
- Same-machine baseline
  `same_machine_qwen_source_switch_projection_control`.
- Maximum candidate/reference latency ratio `3.0`.

Evidence path:
`artifacts/quality/qwen36-35b-a3b-family-gate-hardened-20260704.json`.

## Train And Calibrate

Expose the smallest command path for users who want to train or adapt their own
artifact.

- Calibration or imatrix collection command.
- Materialization command for the VQ or mixed-precision artifact.
- Optional sidecar trainer command.
- Teacher-cache comparison command.
- Tail/domain inspection command.
- Safety rule for protected seeds and output overwrite.

For GLM-4.5-Air, these surfaces are:

```text
benchmarks/collect_glm45_air_imatrix.py
benchmarks/materialize_glm45_air_dynamic_imatrix_sweep.py
benchmarks/finetune_glm45_air_vq_continuous.py
benchmarks/run_glm45_air_rc_pipeline.py --write-focus-json
```

For a new family, either generalize those wrappers or create clearly named
family-specific equivalents.

## RC Summary

Create a summary wrapper or extend an existing one so a user can run one cheap
command before heavy reruns.

The summary should:

- preflight required local paths and entrypoints
- read recorded eval and benchmark evidence
- recompute split metrics from JSONL rows
- run or read the artifact audit
- check hard gates and aspirational quality targets separately
- write `rc-summary.json`
- write `RC_SUMMARY.md`
- write a compact focus JSON when tail/domain follow-up is useful
- include exact rerun commands with run status

Public command shape:

```bash
uv run python benchmarks/run_<family-name>_rc_pipeline.py --preflight
uv run python benchmarks/run_<family-name>_rc_pipeline.py --overwrite
```

Run status: placeholder command, not run until the family-specific wrapper
exists.

## Required Tests

Add tests before calling the family public-ready.

- Adapter config parsing.
- Non-expert binding skips routed expert tensors.
- Routed projection binding validates shapes and expert counts.
- Artifact loading does not materialize dense routed expert weights.
- Manifest lineage records source and seed paths.
- Audit catches missing projections, high-precision routed fallbacks, dense
  routed experts, and unbound VQ experts.
- Eval summarization rejects dirty rows and recomputes metrics from JSONL.
- Benchmark summarization rejects memory-pressure rows.
- RC summary hard-gate and quality-target thresholding.
- Sidecar loading, if the family supports continuous, sparse, or router
  sidecars.

Reference GLM tests:

```text
tests/test_glm45_air_vq_adapter.py
tests/test_quantized_vq_switch_linear.py
tests/test_glm45_air_quant_compare.py
tests/test_glm45_air_teacher_cache.py
tests/test_glm45_air_rc_pipeline.py
```

## Publication Gate

Do not publish the family RC until the following are recorded:

- artifact path
- source or seed path
- artifact audit path
- report eval path
- selection eval path
- holdout eval path, or a documented reason no holdout exists yet
- benchmark path and control path
- summary output path
- hard pass/fail status
- current quality misses and caveats
- commands verified in the current slice
- commands intentionally not run, with the reason
