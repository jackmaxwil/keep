# Teich corpus v2 — SOTA curation, compress-to-fit, CKA + router-balance loss (approved 2026-07-12)

Supersedes the window-size and volume choices in the v1 spec. Personal coding-agent-specialist run on real Claude/Codex/Cursor transcripts. Grounded in July-2026 SOTA: STITCH/Less-Is-More (curated <1K traces beat scale), targeted-distillation-on-real-distribution, Long-Insight compression, SERA agentic format, CKA-QAD (2606.05682), B-Distill MoE router balance.

## Approved decisions

- **(a) Compress-to-fit ~64K**, not raw 1M. Compress full sessions (noise-cut) so whole coherent sessions fit ~64K; train with frozen-prefix disk boundaries + gradient checkpointing + segment-based grad + CPU/SSD activation offload + native MLA/DSA. Measure the exact ceiling; push beyond 64K only if a behavior needs it.
- **(b) Parallel Codex curation waves**, up to **20 subagents concurrently** (disjoint session shards → no file conflict; the 3-writer cap is for shared-source code, not data shards). Use as many as is efficient.
- **(c) CKA + router-balance loss IN SCOPE now** (not deferred): top-K KL base + intermediate-layer CKA regularizer (dynamic loss balancing) + B-Distill entropy-aware router distillation + Monte-Carlo expert exploration.

## Memory plan (compress-to-64K training)

- Frozen prefix (0→trained layer): inference-only, expert-streamed from disk (LightLX) + MLA + DSA; boundary cached to disk (built). One-time per row.
- Suffix (trained layer→head) backward: segment-based gradient computation (LeMo: chunk sequence, per-segment grad, discard activations, aggregate → peak÷N) + gradient checkpointing + activation offload to CPU/SSD (MemAscend/ZeRO-Infinity). Native DSA keeps attention ~linear past 2K.
- Teacher signal on disk (top-K logits + router + optional CKA probes).

## Dataset — canonical format

Canonical agentic (SERA-style): `system(tool schemas+protocol) → user(task) → assistant(reasoning+tool_calls) → tool_output(truncated) → … → assistant(answer)`. OpenAI `messages`/`tools` + per-message `loss_weights` (supervise assistant reasoning/tool-calls/answers; zero user/system/tool-output) + metadata (provider, quality 0-10, difficulty, edge-case tags, lineage id) + leakage gate vs frozen eval.

## Curation pipeline

1. **Deterministic canonicalize + compress** (one script, prepenv): teich convert → canonical messages; middle-truncate tool outputs >T tokens (head+tail+marker); strip base64/binary/data-URIs; dedup repeated reads; drop empty reasoning; trim trailing duplicate submits. Emit compressed sessions + manifest (provider, session, compressed token len). Shard into ≤20 balanced batches. (Long-Insight: 60-80% token cut preserving causal structure.)
2. **Parallel Codex curation waves (≤20 concurrent):**
   - Macro filter: score each session; drop low-signal/repetitive/incoherent/anti-pattern runs (test-avoidance, circular loops, late-stage waste).
   - Micro extract + tag: decision-critical segments; difficulty × edge-case tags; compress verbose reasoning.
   - Each subagent owns a disjoint shard → writes a scored/tagged/filtered shard.
3. **Deterministic merge:** stratify by (difficulty × edge-case × provider), cap per bucket; **quality-tier weighting: Claude/Codex up-weighted, Cursor a tier below**; dedup (LSH/token-hash); session-disjoint train/holdout; 0 collisions with frozen 66; compress-to-≤64K; emit KEEP prompt-pack + loss_weights + report.

## Loss (in scope now)

- **Top-K KL** (k≈2048–8192) at supervised positions — base. Requires top-K teacher cache.
- **CKA regularizer**: intermediate-layer representation matching (teacher vs adapter-corrected student) at selected layers; dynamic loss balancing `L = L_KL + sg(L_KL/(L_CKA+eps))·L_CKA`. Requires teacher intermediate activations at probe layers (extend teacher cache OR teacher forward during training on a probe-layer subset).
- **Router balance (B-Distill)**: entropy-aware router distillation (KL(teacher_router‖student_router) − β·H(student_router)) + Monte-Carlo expert exploration (perturb router so all experts get gradient — fixes expert-coverage deficiency). Requires teacher router distributions at supervised positions.

**Teacher-signal coupling (design note):** CKA + router distillation need richer teacher signal than final logits (intermediate activations + router logits). Decide during loss-recon whether to (i) extend the disk teacher cache with probe-layer activations + router top-k, or (ii) run the teacher forward during training for those terms (2× model resident — feasible if packed). Prefer (i) for probe-layer subset to keep precompute discipline.

## Out of scope (this run)

Public release, confidentiality/anonymization, general/multilingual coverage, on-policy distillation (future lever).
