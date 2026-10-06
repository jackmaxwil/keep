# GLM52 Raw Family-Gate Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the already-proven full 225-group artifact, accepted 98,433,923,808-byte / 1.593443277857996-bpw payload, and authenticated production bind/generation into the GLM52 family gate without weakening its remaining teacher, quality, route/math, or benchmark blockers.

**Architecture:** Preserve the historical six-group schema-v1 gate as a backward-compatible blocked snapshot. Add a schema-v2 blocked checkpoint that consumes the raw non-VQ package evidence, strict composite audit, and detailed production-generation probe together. Derive the path-free common artifact identity from the audit and non-VQ evidence, require the production probe to repeat that identity exactly, cross-check overlapping input file SHA-256 values in the CLI, and teach the declarative gate evaluator the one exact schema-v2 blocked state. Do not accept family-eval or family-benchmark summaries until their raw-row validators exist.

**Tech Stack:** Python 3.11+, pytest, KEEP recipe YAML, `mlx_vq.quality`, `mlx_vq.build`, canonical JSON SHA-256 evidence contracts.

## Global Constraints

- Keep `/Users/jack.mazac/Developer/keep/.keep-heavy-job.lock` and `/Users/jack.mazac/Developer/keep/runs/` untouched and untracked.
- Do not run model work, teacher-cache generation, candidate evaluation, benchmark work, Air work, peer work, or any custom MLX wired-memory configuration.
- Do not claim cold production residency: the accepted probe proves clean warm/steady-state bounded-prefill generation while `production_residency_proven` remains false.
- Preserve the user-accepted exact artifact target: 98,433,923,808 tensor-payload bytes and 1.593443277857996 exact bpw; the frozen policy carries the rounded 1.5934433 value.
- Preserve layers 3 through 77 only, the exact 225 ordered routed groups, no layer 78 MTP, and no dense routed-expert fallback.
- Keep `release_pass_enabled=false`, `raw_release_evidence_validators_ready=false`, and `regate_can_complete=false` until raw teacher/eval/route/benchmark validators exist.
- Do not stage or commit generated artifact payloads or unrelated worktree state.
- Create local reviewed checkpoint commits task by task; do not push or publish them.

---

## Task 1: Specify and implement schema-v2 raw evidence validation with TDD

**Files:**

- Modify: `tests/test_glm52_family_gate.py`
- Reference: `artifacts/quality/glm52-wave6-full-artifact-audit-20260709.json`
- Reference: `artifacts/build/glm52_reap_504b_materialization_probe_20260709/steps/non_vq_pack-a678bc3c/evidence/evidence_json.json`
- Reference: `artifacts/quality/glm52-wave6-production-generation-bounded-prefill-warm-resident-20260710.json`

- [x] Add faithful compact fixtures for the raw non-VQ evidence, full composite audit, common artifact identity, and detailed production generation probe. The fixtures must carry the real schema fields and exact constants, not the old optimistic summary fixture.

- [x] Add a happy blocked-checkpoint test that supplies all three raw inputs plus a full 225-group bind preflight and asserts exactly:

```python
assert report["gate_schema_version"] == 2
assert report["checks"]["full_225_group_artifact_ready"] is True
assert report["checks"]["accepted_artifact_bytes_and_bpw_ready"] is True
assert report["checks"]["production_binding_and_generation_ready"] is True
assert report["checks"]["teacher_cache_full_vocabulary_ready"] is False
assert report["checks"]["full_vocabulary_source_relative_family_eval_ready"] is False
assert report["checks"]["route_math_diagnostics_ready"] is False
assert report["checks"]["same_machine_pinned_fp4_benchmark_ready"] is False
assert report["missing_requirements"] == [
    "dequantized_source_teacher_cache_payload",
    "full_vocabulary_source_relative_family_eval",
    "route_math_diagnostics",
    "same_machine_pinned_fp4_benchmark",
]
assert report["release_pass_enabled"] is False
assert report["raw_release_evidence_validators_ready"] is False
assert report["family_gate_pass"] is False
```

- [x] Assert the gate emits the common identity body and its canonical digest, plus granular validator readiness showing artifact and production validators ready while eval and benchmark raw validators remain false.

- [x] Add parameterized fail-closed tests for these independent mutations:

  - audit record/status/pass or lineage drift;
  - any audit check false or any blocker present;
  - non-canonical/missing/reordered routed groups or any layer 78 entry;
  - E8/group-512/max-abs drift;
  - routed codes/scales/codebook/non-VQ/whole-model byte-accounting drift;
  - exact bpw or accepted policy target drift;
  - non-VQ package hash or inventory mismatch;
  - common identity body mismatch or rehashed identity alias mismatch;
  - production scope/binding/generation/whole-runtime false;
  - dense or unbound VQ experts;
  - non-finite or non-full-vocabulary logits;
  - generated-token count/range/greedy/direct-prefill/no-lookahead drift;
  - any production input-fingerprint revalidation flag false;
  - a custom MLX wired limit or non-default wired policy;
  - false-to-true changes to quality, speed, long-context, or benchmark claims.

- [x] Add a CLI test where a valid production JSON records the wrong composite-audit file SHA-256 and assert exit 1 with `glm52_family_gate_invalid`; then fix the hash and assert exit 2 with the four genuine blockers.

- [x] Run the focused tests and confirm they fail for the missing schema-v2 behavior:

```bash
UV_CACHE_DIR=/tmp/keep-uv-cache uv run python -m pytest -q \
  tests/test_glm52_family_gate.py
```

Expected red phase: failures identify absent raw inputs/validators and schema-v2 evaluator behavior, not fixture syntax errors.

### Implement strict composite and production validators

**Files:**

- Modify: `src/mlx_vq/quality/glm52_family.py`
- Modify: `benchmarks/check_glm52_family_gate.py`
- Test: `tests/test_glm52_family_gate.py`

- [x] Add immutable schema constants while retaining v1 compatibility:

```python
GLM52_FAMILY_GATE_V1_SCHEMA_VERSION = 1
GLM52_FAMILY_GATE_V2_SCHEMA_VERSION = 2
GLM52_FAMILY_GATE_SCHEMA_VERSION = GLM52_FAMILY_GATE_V1_SCHEMA_VERSION
GLM52_FAMILY_GATE_V2_MISSING_REQUIREMENTS = (
    "dequantized_source_teacher_cache_payload",
    "full_vocabulary_source_relative_family_eval",
    "route_math_diagnostics",
    "same_machine_pinned_fp4_benchmark",
)
```

  Keep the shared latest-schema alias at v1 during Task 1 so the unchanged
  declarative evaluator continues to authenticate historical checkpoints. The
  family-gate checker must select `GLM52_FAMILY_GATE_V2_SCHEMA_VERSION`
  explicitly for complete raw-trio output. Task 3 advances the shared alias to
  v2 only after its evaluator supports both schema versions.

- [x] Add a type-strict non-VQ validator that requires the pinned v1 `glm52_non_vq_package_manifest`, ready/pass/resume/production fields, exact 1,194 tensors / 9 shards / 18,560,731,704 parameters / 37,121,488,608 payload bytes, exact source/config/index/profile lineage, and a green nested package audit whose manifest/package-set identity matches the top level.

- [x] Add a strict full-artifact validator that requires the v1 ready/pass audit, exact ordered 225-group inventory for layers 3–77, empty partial layers/blockers, all checks true, byte identity and resume verification, no dense routed experts, E8/group-512/max-abs, exact tensor accounting, and exact full payload bpw. Cross-bind its non-VQ audit hashes to the validated non-VQ evidence.

- [x] Derive the path-free identity as a new mapping from validated fields. Its canonical body must contain exactly the 24 production identity fields already emitted by the production probe, including:

```python
{
    "schema_version": 1,
    "identity_kind": "glm52_production_composite_v1",
    "model_id": PINNED_GLM52_MODEL_ID,
    "source_revision": PINNED_GLM52_REVISION,
    "profile": "glm52-reap-504b-v2",
    "profile_sha256": GLM52_PROFILE_SHA256,
    "profile_contract_sha256": GLM52_PROFILE_CONTRACT_SHA256,
    "config_sha256": GLM52_REAP_CONFIG_SHA256,
    "source_index_sha256": GLM52_REAP_INDEX_SHA256,
    "source_blob_inventory_sha256": non_vq["source_blob_inventory_sha256"],
    "source_inventory_sha256": non_vq["source_inventory_sha256"],
    "non_vq_manifest_sha256": non_vq["manifest_sha256"],
    "non_vq_package_index_sha256": non_vq["package_index_sha256"],
    "non_vq_package_set_sha256": non_vq["package_set_sha256"],
    "routed_manifest_sha256": audit["manifest_sha256"],
    "routed_group_set_sha256": audit["group_set_sha256"],
    "routed_group_inventory_sha256": canonical_sha256(
        audit["expected_group_keys"]
    ),
    "routed_group_count": 225,
    "code_bits": 8,
    "group_size": 512,
    "scale_estimator": "max_abs",
    "whole_model_parameter_count": 494_194_805_304,
    "whole_model_tensor_payload_bytes": 98_433_923_808,
    "whole_model_tensor_payload_bpw": 1.593443277857996,
}
```

- [x] Add a strict production validator that requires the detailed v1 ready/pass probe, the exact derived identity under both aliases and both canonical digests, production-composite scope, all 225 groups and layers 3–77, full bind/generation/runtime proof, no dense/unbound experts, exact non-VQ bind counts, all six fingerprint revalidation flags, and the one-token direct-prefill/full-vocabulary/default-wired contract.

- [x] Preserve negative-claim honesty by requiring `quality_claim`, `speed_claim`, `same_machine_benchmark_proven`, `full_vocabulary_eval_proven`, `long_context_indexshare_proven`, `cold_residency_memory_clean`, and `generation_warmup_memory_clean` to remain false, while requiring warm and steady-state proof true. Do not require `production_residency_proven=true` for the bind/generation check.

- [x] Update `check_glm52_family_gate` to support exactly two modes:

  1. no raw trio supplied: emit the existing schema-v1 six-blocker checkpoint unchanged;
  2. non-VQ + composite audit + production generation all supplied: emit schema v2 with the artifact/size/production checks true and exactly four blockers.

  Any partial trio is invalid. Family eval and family benchmark inputs remain rejected until their raw-row validators exist.

- [x] Include all three new mappings in `input_evidence_contract_sha256`, emit the derived common identity and digest, and retain a canonical `gate_contract_sha256` over the entire output.

- [x] Run the Task 1 focused tests until green.

## Task 2: Cross-bind files, CLI, op registration, and a checked evidence recipe

**Files:**

- Modify: `benchmarks/check_glm52_family_gate.py`
- Modify: `src/mlx_vq/build/ops.py`
- Add: `recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml`
- Modify: `tests/test_glm52_family_build_integration.py`

- [x] Add CLI flags and optional OpDef inputs:

```text
--non-vq-evidence-json
--composite-artifact-audit-json
--production-generation-json
```

The OpDef must keep the legacy six required inputs, make the three raw files optional as a complete trio, retain accepted return codes `{0, 2}`, remain non-cacheable, require the `glm52_family` declarative gate, and keep `regate_can_complete=false`.

- [x] Before composing schema v2, compare the production probe's recorded input file hashes against the actual CLI input files for these four overlaps:

```python
{
    "family_policy_json": sha256_file(args.family_policy_json),
    "full_bind_preflight_json": sha256_file(args.full_bind_preflight_json),
    "non_vq_evidence_json": sha256_file(args.non_vq_evidence_json),
    "composite_audit_json": sha256_file(args.composite_artifact_audit_json),
}
```

Reject missing, malformed, or mismatched hashes before writing a ready gate checkpoint.

- [x] Make output alias protection cover the raw trio and every existing transitive referenced path exactly as the legacy CLI does.

- [x] Add a one-step checked recipe whose external inputs are the exact existing evidence files with their current file SHA-256 values:

  - policy `0975f7dc...72ce2`;
  - prompt pack `697677a4...edf31`;
  - teacher metadata `621a013e...d7a`;
  - full bind `9078a36c...46a3`;
  - IndexShare `94785cf0...ad25`;
  - synthetic generation `baa9c821...62a9`;
  - non-VQ package `ccbedd87...e80d2`;
  - composite audit `8026322a...91a`;
  - production generation `758b5bbe...c82`.

- [x] In build-integration tests, preserve the legacy recipe assertions and add v2 assertions for exact raw input wiring, command flags, external file hashes, accepted return codes, declarative gate profile, and no boolean bypass parameters.

- [x] Validate and dry-run the new recipe with normal hashing of its nine small
  JSON authorities:

```bash
UV_CACHE_DIR=/tmp/keep-uv-cache uv run keep validate \
  recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml
UV_CACHE_DIR=/tmp/keep-uv-cache uv run keep build \
  recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml \
  --dry-run
```

Expected: structural validation and all nine pinned file hashes succeed; dry-run
renders the schema-v2 gate command and no model or payload work.

## Task 3: Teach the declarative gate evaluator the exact v2 blocked state

**Files:**

- Modify: `src/mlx_vq/build/gates.py`
- Test: `tests/test_build_gates.py`

- [x] Retain exact schema-v1 blocked-contract verification for historical evidence.

- [x] After the evaluator recognizes both the historical v1 checkpoint and the
  exact v2 blocked checkpoint, advance `GLM52_FAMILY_GATE_SCHEMA_VERSION` from
  the temporary v1 alias to `GLM52_FAMILY_GATE_V2_SCHEMA_VERSION`.

- [x] Add declarative evaluator tests for an authenticated schema-v2 blocked checkpoint. Assert `blocked_contract_consistent` is true while the overall gate remains false because the four open checks are false.

- [x] Add a rehashed all-green schema-v2 forgery test and assert it is rejected by `blocked_contract_consistent`, preserving the existing anti-boolean-bypass invariant.

- [x] Add exact schema-v2 blocked-contract verification requiring:

  - schema version 2;
  - release pass disabled and global raw validators not yet ready;
  - the exact check inventory;
  - full routed coverage, full artifact, accepted bytes/bpw, production bind/generation, and dense-free checks true;
  - teacher cache, family eval, route/math, and benchmark checks false;
  - exactly the four schema-v2 missing requirements;
  - canonical common artifact identity authentication;
  - granular validator readiness with only artifact and production true;
  - `family_gate_pass=false` and blocked status.

- [x] Ensure a rehashed mutation cannot promote a v2 checkpoint to green. The declarative evaluator must not treat a self-computed `gate_contract_sha256` as authorization to change the fixed accepted state.

- [x] Run both focused suites:

```bash
UV_CACHE_DIR=/tmp/keep-uv-cache uv run python -m pytest -q \
  tests/test_glm52_family_gate.py \
  tests/test_glm52_family_build_integration.py \
  tests/test_build_gates.py -k 'glm52 or family_gate'
```

## Task 4: Produce the real checkpoint, document it, review it, and commit

**Files:**

- Modify: `benchmarks/check_glm52_family_gate.py`
- Modify: `tests/test_glm52_family_gate.py`
- Generate: `artifacts/quality/glm52-family-gate-raw-evidence-20260710.json`
- Modify: `docs/GLM52_PIPELINE_READINESS.md`
- Modify: `WORK_LOG.md`
- Modify: this plan's checkboxes as tasks complete

- [x] Harden the single-read JSON authority loader to reject duplicate object keys and non-standard `NaN`/`Infinity` constants, with focused CLI regressions, before authenticating or emitting the final checkpoint.

- [x] Run the CLI against the exact current raw inputs and capture exit 2:

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

- [x] Independently inspect the generated JSON and verify: exact identity digest `ef9d2e49...ee5067`; 225/225 groups; accepted bytes/bpw; artifact/size/production checks true; four open blockers; no quality/speed/release claim.

- [x] Update readiness documentation and work log with the exact evidence path, file SHA-256, command, promoted checks, remaining blockers, and the explicit warm-vs-cold residency distinction.

- [x] Run broad proportional verification:

```bash
UV_CACHE_DIR=/tmp/keep-uv-cache uv run python -m pytest -q \
  tests/test_glm52_family_gate.py \
  tests/test_glm52_family_build_integration.py \
  tests/test_build_gates.py \
  tests/test_glm52_production_generation.py
UV_CACHE_DIR=/tmp/keep-uv-cache uv run python -m compileall -q \
  benchmarks/check_glm52_family_gate.py \
  src/mlx_vq/quality/glm52_family.py \
  src/mlx_vq/build/gates.py \
  src/mlx_vq/build/ops.py
git diff --check
```

- [x] Run a separate specification review and code-quality review. Fix every finding inside this wave, then rerun focused verification.

- [x] Confirm the worktree contains only the intended source/docs/recipe changes plus the pre-existing untracked lock and runs directory:

```bash
git status --short
```

- [x] Commit only the remaining intended tracked files with a local checkpoint message such as:

```bash
git add \
  benchmarks/check_glm52_family_gate.py \
  src/mlx_vq/quality/glm52_family.py \
  src/mlx_vq/build/gates.py \
  src/mlx_vq/build/ops.py \
  tests/test_glm52_family_gate.py \
  tests/test_glm52_family_build_integration.py \
  tests/test_build_gates.py \
  recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml \
  docs/GLM52_PIPELINE_READINESS.md \
  docs/superpowers/plans/2026-07-09-glm52-raw-family-gate-evidence.md \
  WORK_LOG.md
git commit -m "feat: authenticate GLM52 family gate evidence"
```

Do not stage `.keep-heavy-job.lock`, `runs/`, or generated artifact payloads.

## Task 5: Close final evidence-chain trust gaps

**Files:**

- Modify: `src/mlx_vq/quality/glm52_family.py`
- Modify: `src/mlx_vq/build/gates.py`
- Modify: `recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml`
- Modify: `tests/test_glm52_family_gate.py`
- Modify: `tests/test_glm52_family_build_integration.py`
- Modify: `tests/test_build_gates.py`
- Modify: this plan

- [x] Make the evidence recipe fail closed on authority drift by setting `strict_inputs: true`. Assert strictness and a mismatched-hash planning failure in integration tests. Update the checked dry-run command to hash the nine small JSON authorities normally instead of using `--no-hash`.

- [x] Replace outer-container-only equality with recursive exact-type equality for nested mappings and lists. Add regressions proving `1` cannot stand in for `true`, `false` cannot stand in for `0`, and `3.0` cannot stand in for integer layer `3`.

- [x] Validate the full non-VQ raw inventories: require exactly 1,194 unique, ordered, typed tensor records and nine canonical shard records; reconcile dtype, parameter, payload, offset, shard-membership, per-shard inventory digest, file digest shape, and package-set digest. Add missing/empty/reordered/duplicate/hash-inconsistent regressions.

- [x] Validate production memory rows instead of trusting readiness booleans. Require and type-check the relevant `phase_memory` records; cross-bind `cold_residency_memory`, `generation_warmup_phase_memory`, post-warmup clear/quiet, `pre_generation_memory`, and `steady_state_generation_memory` to their phase labels; derive pageout/swapout cleanliness; and reject contradictory warm/cold/steady-state claims.

- [x] Strengthen schema-v2 declarative evaluation to require the exact nine-entry captured-file and canonical-contract hash maps, then cross-bind accepted bytes/bpw, routed counts, production flags, and negative proof flags to the pinned common identity and promoted check state. Add fully rehashed omission and contradictory-summary regressions.

- [x] Run the focused gate, build-integration, and declarative-evaluator suites. Validate and dry-run the strict one-step recipe with normal hashing of its nine small files.

```bash
UV_CACHE_DIR=/tmp/keep-uv-cache uv run python -m pytest -q \
  tests/test_glm52_family_gate.py \
  tests/test_glm52_family_build_integration.py \
  tests/test_build_gates.py -k 'glm52 or family_gate'
UV_CACHE_DIR=/tmp/keep-uv-cache uv run keep validate \
  recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml
UV_CACHE_DIR=/tmp/keep-uv-cache uv run keep build \
  recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml --dry-run
```

- [x] Recompose `artifacts/quality/glm52-family-gate-raw-evidence-20260710.json` from the same authorities and verify its semantic contract remains unchanged: schema v2, identity `ef9d2e49...ee5067`, `225/225/0`, exact accepted bytes/bpw, three promoted evidence surfaces, and exactly four blockers. Record any file-SHA change before updating documentation.

- [x] Run broad proportional tests, compileall, and `git diff --check`; obtain a clean independent review before closing the task. Do not run model or payload work and do not commit generated evidence, `.keep-heavy-job.lock`, or `runs/`.
