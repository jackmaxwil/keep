# GLM-4.5-Air Heavy Quantization Recovery Design Note

This note is a gate, not an implementation plan. RHT, QuIP-style incoherence,
GPTQ-like calibration, adaptive rounding, or error-propagation recovery must not
be implemented until a separate reviewed plan accepts the estimates and stop
conditions below.

## Current Baseline

- Active target: `zai-org/GLM-4.5-Air`.
- Stable resident artifact: `artifacts/glm-4.5-air-vq`.
- Baseline artifact size from the conversion ledger: `12,905,574,720` bytes.
- Planned VQ source shards present: `45` shards, `213,219,635,784` bytes.
- Resident proof remains the invariant: no dense routed expert tensors in
  `model.parameters()`, no raw `model.layers.*.mlp.experts.*` tensors loaded in
  resident inference, and 4K remains the active context baseline.
- Q3 `percentile_99` one-projection candidate was rejected: prompt-derived
  layer-1 source-weighted cosine regressed and 3 of 5 Q1 prompt NLL probes
  worsened.
- Q4 activation stats cover layer-1 fixed-prompt routing: the five Q1 prompts
  cover `126/128` experts, and the 1K prompt covers `118/128` experts.
- Q5 mixed-code-bit policy is metadata/planning only; no 16-bit runtime artifact
  or kernel is implemented.

## Candidate Methods

1. Randomized Hadamard transform or QuIP-style incoherence before VQ rounding.
2. Hessian-aware or GPTQ-like selected projection calibration.
3. Layer-output error propagation or asymmetric correction.
4. Adaptive/codebook rounding for selected layer/projection groups.
5. Selective VQ-2/E8P 16-bit groups only after runtime kernel and memory gates
   exist.

## Disk Estimate

- Full source read input remains the current Air planned-VQ source set:
  `213.22 GB` across 45 shards.
- A full baseline VQ-1 artifact is about `12.91 GB`.
- The Q5 example mixed policy (`1:gate_proj=16`, `*:down_proj=16`) dry-run
  estimates `16,701,718,528` code bytes plus `448,266,240` scale bytes before
  safetensors/container overhead.
- Any heavy candidate must write to a new directory such as
  `artifacts/glm-4.5-air-vq-rht-<date>` and must not overwrite
  `artifacts/glm-4.5-air-vq`.
- Minimum free-space gate before a full heavy run: source already present plus
  at least `2x` expected output size for temp/resume safety. For current Air,
  require at least `40 GB` free for a VQ-1-like variant and at least `60 GB`
  free for a selective mixed-code-bit variant.

## Runtime Estimate

- Q3 one-group layer-1 `gate_proj` conversion with `percentile_99` took about
  `20.83s` for one 128-expert projection group.
- Naive linear extrapolation for all `135` Air VQ groups is about `47 minutes`
  before heavier math.
- RHT or Hessian-aware calibration can multiply runtime by calibration samples,
  transform passes, or Hessian/block solves. A first heavy slice must benchmark
  exactly one selected group and estimate the full run from measured timings
  before converting more than one group.
- A heavy full-run gate must declare:
  - max groups to convert in the first run,
  - expected seconds per group,
  - expected total runtime,
  - source tensors read,
  - peak source tensor bytes,
  - output bytes written,
  - resume point after interruption.

## Resume Manifest

Any heavy method needs a manifest before implementation:

- `model_id`, source revision, source index path, source shard inventory.
- Method name and parameters, including seeds for randomized transforms.
- Layer, projection, expert set, group size, code bits, scale estimator, and
  calibration corpus hash.
- Converted, existing, failed, and skipped groups.
- Per-group runtime seconds, source tensors read, peak source tensor bytes, and
  output path.
- Verification status for Q2 source-oracle rows and Q1 prompt metrics.
- A clear flag for whether the artifact is partial, validation-only, or
  resident-loadable.

## Rollback Path

- Baseline artifact remains `artifacts/glm-4.5-air-vq`.
- Heavy artifacts must live in a new artifact directory.
- Resident benchmarks and quality scripts default to the baseline unless an
  explicit `--artifact-dir` points to the candidate.
- If a candidate fails Q1/Q2/Q4 gates, record it under `artifacts/quality/` and
  keep the baseline as the resident default.
- Do not add BF16 dense routed expert fallback to resident inference as a
  rollback mechanism.

## Required Gates Before Implementation

- One-group heavy-method microbenchmark with disk/runtime/source-read metrics.
- Q2 source-oracle before/after for the selected group.
- Q1 prompt metrics before/after for a resident-loadable candidate or a clear
  statement that the candidate is layer-local only.
- Memory/no-dense invariant check for any resident candidate.
- `git diff --check` and focused pytest for the touched quantization/converter
  surface.

## Stop Conditions

- Full dense Air residency is required.
- Raw routed expert tensors would enter `model.parameters()` in resident
  inference.
- The method requires overwriting `artifacts/glm-4.5-air-vq`.
- One-group runtime or disk usage exceeds the declared estimate by more than
  `2x` without an updated plan.
- Q1 prompt metrics regress without a documented reason and reviewer approval.

## Decision

No heavy recovery implementation is part of the current plan. The next safe
step, if quality remains insufficient, is a separate reviewed implementation
plan that starts with a one-group heavy-method benchmark and the manifest schema
above.
