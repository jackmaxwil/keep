# Teich → KEEP coding-agent distillation corpus — design

**Goal:** Build a real coding-agent training corpus from the user's own Claude Code / Codex / Cursor transcripts (via teich), and distill it into GLM-5.2-REAP-KEEP low-rank adapters to make the model a **coding-agent specialist**. This run trains a **personal** model (not released), so no confidentiality filtering; public release is a separate future run with its own data scoping.

**North star for this run:** raise top-1 agreement with the original model on **held-out coding-agent tokens** (the new specialist gate), using the trained-adapter class proven to generalize (+3–4 pts/layer from one layer).

## Decisions (locked)

- **Specialization:** coding-agent specialist (option A). Transcripts are *the* corpus; eval moves to coding-agent tasks.
- **Data scope:** full corpus, personal model, unreleased (option C). Skip anonymization. Confidentiality/leakage handling is deferred to the future public-release run.
- **teich usage:** upstream (Apache-2.0), installed as an isolated `uv tool`; KEEP adds a thin bridge. No fork.
- **Teacher logits:** top-K (~256) + tail-mass, fp16, generated only at supervised positions — NOT full-vocab FP32 (which is ~600 GB at scale).
- **Eval:** hold out ~10–15% of transcripts (disjoint) as the coding-agent gate; keep the frozen general 66-prompt pack as a secondary regression sanity check.

## Architecture (5 components)

1. **Extraction (teich, upstream).** `teich extract {claude,codex,cursor}` stages sessions → `teich convert` + prepare/mask rendered through **GLM-5.2's tokenizer chat template** → JSONL rows: `input_ids`, `labels` (response-only, `-100` masked), `tools`, `metadata`. Teich's audit reports dropped/oversized/malformed rows.

2. **Bridge (`src/mlx_vq/quality/teich_corpus.py`, new).** teich rows → KEEP prompt-pack rows:
   - `encoded_token_ids = input_ids`
   - **supervised `positions` = indices where `labels != -100`** (distill only on tokens the model generates: assistant + tool calls)
   - `target_token_ids = labels` at those positions
   - length bucketing (256–1024 tok target; drop/truncate over a cap), dedup by token-hash, split assignment (train/holdout), provenance (provider, session id, teich audit refs). Disjoint from any eval set and from the frozen 66.

3. **Top-K teacher cache (extend `glm52_adapter_training` teacher-cache generation + loss).** Generate top-K teacher logits (index+value) + tail-mass at supervised positions; store fp16. Add a top-K KL loss that matches the current full-vocab KL within tolerance on a shared sample (parity gate). Enables scaling to millions of tokens.

4. **Splits + coding-agent eval.** Hold out disjoint transcript slice → coding-agent eval; metric = top-1 agreement on held-out response tokens. Frozen 66 kept as secondary gauge.

5. **Adapter training (existing, now fed real data).** Batched-boundary fast path + validated recipe (lr 0.2, init 2e-2), worst-8 layers, overfitting-aware (early-stop on held-out), trained on the corpus.

## Data flow

transcripts → `teich extract`/`convert`/mask (GLM-5.2 template) → bridge → KEEP prompt-pack (train + holdout) → top-K teacher-logit generation → adapter training → held-out coding-agent top-1 verdict (+ frozen-66 regression check).

## Error handling

- teich audit surfaces dropped/oversized/malformed rows; the bridge records counts and reasons, and fails closed on schema drift.
- Teacher-gen failures per row are logged and skipped, not silently zero-filled.
- Bridge asserts `len(input_ids)==len(labels)`, at least one supervised position/row, and no eval/train leakage (token-hash disjointness).

## Testing

- Bridge: synthetic teich rows → assert positions/targets == supervised (`labels!=-100`) mapping; dedup + disjoint-split invariants; malformed rows rejected.
- Top-K KL: parity vs full-vocab KL within tolerance on a fixed sample; deterministic.
- End-to-end smoke on a tiny transcript slice (host, real tokenizer) before the full run.

## Out of scope (future public-release run)

- Confidentiality curation, anonymization, proprietary-content filtering, memorization/leakage audits.
- General-purpose / multilingual expert coverage (this run is coding-specialist).

## Present-to-user checkpoint

Before any training, present the assembled dataset to the user: row counts by provider/split, token-length distribution, supervised-token totals, a few sample rendered rows, and teich audit summary. Proceed to training only after that review.
