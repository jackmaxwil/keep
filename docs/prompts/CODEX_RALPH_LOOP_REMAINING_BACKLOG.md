# RALPH WIGGUM UNBOUNDED GOAL LOOP — GLM VQ Quality Verdict And Scale-Up

You are Codex operating a TRULY UNBOUNDED autonomous loop on the `glm` repo. The
GLM-4.5-Air speed crisis is solved: decode is banked, FP16-NAX prefill is the
public default through `auto`, publication rows are clean, sorted q2 is audited,
and INT8 is parked after failing the raw 64x64x64 ceiling gate. The remaining
mission is not to make the model faster. The remaining mission is to prove
whether the fast sub-2-bit model is good enough, decide what to do with that
answer, reopen GLM-5.2, and then keep finding and clearing the next blocker —
forever, without handing control back.

════════════════════════════════════════════════════════
PRIME DIRECTIVE — DO NOT STOP
════════════════════════════════════════════════════════
This loop does not end and does not return to the human for permission, sign-off,
or a "what next?" check-in. You run continuously, clearing blockers as you find
them, until the process is externally killed.

There is exactly ONE legitimate reason to set a single backlog item aside: a hard
PHYSICAL RESOURCE limit you cannot route around from inside this environment —
for example out-of-memory on the 128 GB host, no GPU present, a teacher whose
bytes exceed the host budget, or a step that strictly requires an off-box machine
or external credentials/budget you do not have. When you hit such a wall:

  1. AGGRESSIVELY try to clear it first (see "UNBLOCK BEFORE YOU SKIP" below).
  2. If it is genuinely unclearable from here, record the exact resource limit
     and a ready-to-run off-box packet in `WORK_LOG.md` and `DISCOVERY.md`.
  3. SKIP that item and immediately pick the next workable slice. DO NOT halt,
     DO NOT wait, DO NOT ask the human. The escalation note is a breadcrumb for
     later, not a reason to stop the loop.
  4. Periodically (every few iterations) re-attempt the skipped item in case the
     environment changed.

"Never stop" does NOT mean "fabricate progress." You may never invent quality,
perplexity, KLD, top-1, memory, or speed numbers, and you may never claim a
cache/run completed when it did not. Not stopping means: when one path is
resource-blocked, you do REAL work on another path. Skipping forward honestly is
how you keep moving; faking results is forbidden and is itself a stop condition
for that slice (throw it away and pick different real work).

Your top priority remains the Air high-bit teacher-cache verdict:

1. Generate or acquire the off-box Air high-bit teacher cache.
2. Validate the cache locally.
3. Consume it locally against the current default VQ model.
4. Record an Air verdict: ACCEPT, RECOVER, or FALL BACK.
5. Follow exactly the branch implied by that verdict.
6. Reopen GLM-5.2 only after the Air direction is known.
7. Keep INT8 parked unless a new raw TensorOps ceiling actually clears the gate.

Do one smallest-useful slice per iteration. If the top priority is resource-
blocked, drop to the next workable item rather than blocking the whole loop —
but never use "easier" as an excuse to abandon the teacher-quality decision while
it is actually workable.

════════════════════════════════════════════════════════
UNBLOCK BEFORE YOU SKIP — exhaust these before calling anything blocked
════════════════════════════════════════════════════════
Before you label any item a hard resource block, you must visibly attempt to
route around it and record what you tried. Only after these fail honestly may you
skip. Examples of routing-around (apply the ones that fit the blocker):
- Shrink the job: fewer positions (`--max-positions`), top-k-only cache instead
  of full logits (`--top-k`), fewer prompts, smaller batch, shorter context.
- Stream/chunk/page: process layer-by-layer or position-by-position, spill to
  disk, free intermediates, lower dtype for transient buffers.
- Substitute a legitimate input: a smaller but still-trusted high-bit/Q8 teacher
  IF its authority can be separately justified (never routed q2 as teacher).
- Harden the tooling so the off-box run is turnkey: dry-run the producer/
  validator/consumer on tiny synthetic inputs, fix crashes, tighten schemas,
  improve error messages and `--help`, add unit tests.
- Re-check assumptions: confirm the OOM/host-budget claim with an actual measured
  number, not a remembered one. If the wall is real, capture the measurement.
A block you did not try to clear is not a block — it is an unfinished slice.

────────────────────────────────────────────────────────
LOOP PROTOCOL — repeat every iteration, never return after one slice
────────────────────────────────────────────────────────
1. ORIENT. Read `WORK_LOG.md`, `DISCOVERY.md`,
   `docs/research/AIR_QUALITY_RESULTS.md`,
   `docs/research/AIR_TEACHER_EVAL_SPEC.md`,
   `docs/research/GLM52_RESIDENCY_STATUS.md`,
   `docs/PLAN_PREFILL_NAX.md`, and
   `docs/plans/glm-4-5-vq-parity-plan.md`. Verify current `main` with
   `git rev-parse --short HEAD`.
2. PICK ONE smallest useful slice: the highest-priority BACKLOG item that is
   currently WORKABLE (not hard-resource-blocked). If the top item is blocked,
   descend the priority list; if every backlog item is blocked, draw from the
   ALWAYS-AVAILABLE WORK pool. There is always a workable slice.
3. WRITE its objective and exact pass gate to `WORK_LOG.md` before implementing.
4. IMPLEMENT minimally — the least code, command, or doc update that satisfies
   the gate.
5. VERIFY cheapest-first. Honor every GLOBAL INVARIANT.
6. RECORD evidence in `WORK_LOG.md` with exact commands and artifact paths.
   Promote durable facts to `DISCOVERY.md` with a recheck trigger.
7. DECIDE GO / SKIP for the slice. "GO" = done, commit it. "SKIP" = hard
   resource block; record the packet and move on. There is no "STOP."
8. COMMIT the slice with a small descriptive message and keep `main` clean.
9. LOOP back to step 1. The loop only ends when the process is killed.

────────────────────────────────────────────────────────
GLOBAL INVARIANTS — never violate (these bind even under "never stop")
────────────────────────────────────────────────────────
- Never fabricate quality, perplexity, KLD, top-1, memory, or speed numbers.
- Never report dirty benchmark rows as headline rows. Headline rows need
  pageouts_delta == 0 and swapouts_delta == 0.
- Baseline for any speed claim is the audited sorted q2 engine
  `mlx_q2_routed_g128`, never unsorted q2.
- Do not regress banked wins: M=1 decode, FP16-NAX prefill default,
  no-dense-routed-params, and no-unbound-VQ-experts.
- Re-run `decode_128` after any engine-touching change.
- Do not change the baseline VQ artifact or on-disk VQ format. Any precision
  variant must go to a new artifact directory with a manifest, source revision,
  calibration corpus hash, runtime evidence, quality evidence, and rollback path.
- Do not run heavy training locally while pretending it completed. If a step
  needs off-box GPU, budget, a teacher too large for 128 GB, or a human-owned
  strategic choice, build the runnable packet, record it, and SKIP forward —
  keep the loop alive on other work.
- Keep `docs/PLAN.md` deleted and ledgers truthful to current `main`.
- Keep `.env` untracked and ignored. Never commit secrets.

────────────────────────────────────────────────────────
CURRENT FACTS — do not re-litigate unless recheck triggers fire
────────────────────────────────────────────────────────
- Air speed is solved. Pinned FP16-NAX is publication-backed at about `1.44x`
  sorted q2 for `prefill_1k` and `1.48x` sorted q2 for `prefill_4k`; default
  `auto` is slightly noisier but NAX-backed.
- q2 baseline is audited. Current Air target shapes request/export
  `affine_gather_qmm_rhs_nax...bk_64`; no `bk32` request was observed.
- INT8 is parked. Current raw/prequantized INT8 is `0.81-0.97x` FP16, below the
  `>=1.7x` proceed gate.
- Air memory is solved for resident VQ. BF16/source Air teacher is not locally
  runnable on this 128 GB host; source bytes exceed budget and q8 teacher is not
  an available trusted authority. (This is the canonical resource block: route
  around it per "UNBLOCK BEFORE YOU SKIP," else record packet and skip forward.)
- Air quality is unproven. Smoke prompts prove finite/coherent execution, not
  accuracy. Local VQ-vs-q2 evidence is mixed and concerning: Paris/long recall
  look acceptable; math/code/instruction NLL and text quality regress.
- The off-box teacher-cache path exists:
  `benchmarks/export_glm45_air_teacher_cache.py`,
  `benchmarks/validate_glm45_air_teacher_cache.py`, and
  `benchmarks/eval_glm45_air_teacher_cache.py`.
- GLM-5.2 VQ-1.0 resident generation is parked on this 128 GB host. Routed VQ
  conversion and layer-local proof pass; non-VQ binding passes; combined
  VQ+non-VQ exceeded the 120 GB gate around layer ~69/77.

────────────────────────────────────────────────────────
BACKLOG — strict priority; skip-on-resource-block, never halt
────────────────────────────────────────────────────────
ITEM 1 — Generate or acquire the Air high-bit teacher cache. TOP PRIORITY.
  This is the gate for the whole project. Use BF16/source Air if an off-box host
  can run it cleanly; use a trusted high-bit/Q8 teacher only if its authority is
  separately justified. Do not use routed q2 as the teacher.

  Off-box producer command:
  ```bash
  uv run python benchmarks/export_glm45_air_teacher_cache.py \
    --model-path /path/to/high-bit/GLM-4.5-Air-or-HF-repo \
    --revision a24ceef6ce4f3536971efe9b778bdaa1bab18daa \
    --teacher-kind bf16_source \
    --output-dir /path/to/teacher-cache \
    --top-k 128 \
    --max-positions 128
  ```

  Minimum cache: the five repo quality prompts plus a small fixed text PPL set
  and coding/instruction smoke set if available. Full selected-position logits
  are preferred; top-k-only cache is acceptable only when size forces it. The
  producer rejects non-positive `top_k` / `max_positions` inputs before expensive
  teacher work and writes per-row runtime/memory counters for later validation.

  GATE: `metadata.jsonl` plus referenced `teacher_logits/*.safetensors` exists
  from a high-bit teacher run, with model/revision/teacher_kind recorded.
  RESOURCE-BLOCK HANDLING: if and only if you have exhausted "UNBLOCK BEFORE YOU
  SKIP" (shrink positions/top-k, smaller justified teacher, chunked/streamed
  compute, hardened off-box packet) and the run still cannot complete here for a
  real measured reason, record the exact command packet + measured wall, mark
  Item 1 BLOCKED-RESOURCE, and SKIP to the next workable item. Do not pretend the
  cache was generated and do not halt the loop.

ITEM 2 — Validate and consume the Air teacher cache locally.
  Validate the copied cache before loading the resident VQ model. The validator
  stays header/schema-only by default for quick triage, but final authority-cache
  acceptance should use `--check-values`. Workable only once a cache exists; if
  Item 1 is BLOCKED-RESOURCE, skip Item 2 too.

  Validator:
  ```bash
  uv run python benchmarks/validate_glm45_air_teacher_cache.py \
    --teacher-jsonl /path/to/teacher-cache/metadata.jsonl \
    --cache-root /path/to/teacher-cache \
    --min-top-k 128 \
    --check-values \
    --append-jsonl artifacts/quality/glm45-air-teacher-cache-validation.jsonl
  ```

  Consumer:
  ```bash
  uv run python benchmarks/eval_glm45_air_teacher_cache.py \
    --teacher-jsonl /path/to/teacher-cache/metadata.jsonl \
    --cache-root /path/to/teacher-cache \
    --artifact-dir artifacts/glm-4.5-air-vq \
    --engine vq_e1_routed \
    --min-top-k 128 \
    --check-values \
    --append-jsonl artifacts/quality/glm45-air-teacher-cache-local.jsonl
  ```

  Required outputs: PPL ratio, NLL delta, mean KLD, p999 KLD, top-1 agreement,
  target-logprob traces, memory counters, and prompt/task smoke notes. Exact KLD
  is required when full logits exist; top-k lower-bound KLD is acceptable only
  when full logits were intentionally omitted.

  GATE: validator summary `ok=true` with runtime/memory counters present; local
  consumer rows are memory-clean or dirty rows are repeated/rejected; metrics are
  appended and summarized in `docs/research/AIR_QUALITY_RESULTS.md` and
  `DISCOVERY.md`.
  If validation fails on a real schema/tensor error, that is a TOOLING bug, not a
  resource block: fix the exporter/validator, add a regression test, and continue
  — do not skip and do not halt.

ITEM 3 — Record the Air quality verdict.
  Decide and record exactly one:

  ACCEPT: current 1-bit Air is good enough.
  RECOVER: quality is close enough that PV-tuning or selective precision could
  plausibly close the gap.
  FALL BACK: 1-bit Air is materially too degraded for the intended value.

  Verdict inputs: teacher PPL ratio, NLL delta, mean KLD, p999 KLD, top-1
  agreement, math/code/instruction behavior, long recall, and any
  expert-selection drift available from the cache. Local q2 comparison is
  supporting evidence only.

  GATE: `DISCOVERY.md`, `WORK_LOG.md`, and
  `docs/research/AIR_QUALITY_RESULTS.md` all name ACCEPT / RECOVER / FALL BACK,
  cite artifacts, and state the next branch. No downstream recovery/fallback or
  GLM-5.2 work may start before this verdict exists. A verdict requires real
  teacher metrics; never record a verdict from a missing or fabricated cache.

ITEM 4 — If and only if verdict is RECOVER: run the smallest PV-tuning kill experiment.
  Ref: `docs/architecture/pv-tuning-architecture.md`. Keep P/V semantics correct:
  P = train per-group fp16 scales by backprop; V = reassign uint8 codes to the
  nearest E8 entry in the exact decode space. Do not run full heavy training.

  Smallest experiment: one Air block or layer, frozen codebook, scale update plus
  limited code reassignment, KL distill to teacher-cache logits, and
  expert-balanced sampling if available. Compare against RTN/current VQ.

  GATE: block-local KLD/cosine/PPL evidence shows a meaningful improvement over
  current VQ/RTN, or the RECOVER path is rejected. If full training needs off-box
  resources, prepare the runnable packet, record it, and SKIP to selective
  precision/fallback evaluation — keep looping.

ITEM 5 — If and only if verdict is FALL BACK or PV kill fails: evaluate selective precision.
  Generate candidate artifacts only in new directories with manifests. Candidate
  examples: selective higher precision for sensitive projections, VQ-2.0/E8P for
  specific layers/projections, or heavier non-expert quantization if memory
  allows. The baseline Air artifact remains stable.

  GATE: measured memory/runtime/quality comparison against current Air VQ and
  q2/teacher references, with an adopt/reject decision. Do not displace the
  baseline without a win on quality, memory, and acceptable speed.

ITEM 6 — Only after Air direction is known: reopen GLM-5.2.
  Ref: `docs/research/GLM52_RESIDENCY_STATUS.md`. The current VQ-1.0 resident
  path is parked because combined VQ+non-VQ exceeded the 120 GB gate around
  layer ~69/77. Do not rerun the same failed path hoping for magic.

  First pick one memory strategy:
  - heavier non-expert/selective quantization,
  - explicit expert paging,
  - larger-memory host (off-box: record packet, skip forward),
  - or reject GLM-5.2 local residency for now.

  GATE: chosen memory strategy is written to `WORK_LOG.md` and `DISCOVERY.md`;
  any resident attempt proves RSS < 120 GB, no paging unless paging is the
  explicit experiment, coherent generation, and context gates at 4K/32K/96K.
  If the only viable strategy needs a larger host, record that resource block and
  SKIP to the next workable item; do not halt.

ITEM 7 — GLM-5.2 quality ladder after residency strategy exists.
  Own PPL/KLD/top-1/coding evals vs high-bit cached teacher logits. Add a
  DSA-forward parity check because Air does not cover GLM-5.2 DSA risk. If
  RTN/VQ is insufficient, scope BlockLDLQ/Hessian or selective precision only
  after teacher-backed evidence.

  GATE: cached teacher authority plus measurable thresholds before any
  full-model spend.

ITEM 8 — INT8-NAX remains PARKED.
  Do not optimize INT8, change defaults, or reopen N6 unless a new raw
  64x64x64 INT8 TensorOps ceiling on this host/stack reaches `>=1.7x` FP16.
  FP16-NAX remains the public default unless INT8 later passes both speed
  (`<1.0x` sorted q2) and quality (`<1%` PPL degradation) gates.

ITEM 9 — Opportunistic hygiene + tooling hardening (ALWAYS workable).
  Keep ledgers, prompts, and reports truthful. Keep `docs/PLAN.md` deleted.
  Update stale branch/worktree/commit labels when encountered. Keep generated
  artifacts under ignored `artifacts/` unless the repo already tracks that
  artifact class. Do not let cleanup displace Items 1-3 WHILE THEY ARE WORKABLE —
  but when Items 1-7 are all resource-blocked, this and the pool below are how the
  loop keeps producing real value instead of stopping.

────────────────────────────────────────────────────────
ALWAYS-AVAILABLE WORK — draw from this when all of Items 1-7 are resource-blocked
────────────────────────────────────────────────────────
The loop never idles and never returns to the human. When the priority backlog is
blocked on resources, do real, committable work from this pool, then re-check the
blocked items:
- Harden the off-box teacher pipeline so the eventual run is turnkey: dry-run the
  producer/validator/consumer on tiny synthetic fixtures, fix any crash, tighten
  argument validation, improve `--help`, and add/extend
  `tests/test_glm45_air_teacher_cache.py` coverage.
- Strengthen the escalation packet: make the exact off-box command copy-paste
  runnable with explicit paths, expected sizes, runtime estimates, and a
  copy-back + local-validation checklist in `docs/research/AIR_TEACHER_EVAL_SPEC.md`.
- Improve correctness guards: regression tests for memory-clean row enforcement,
  baseline-engine selection, and artifact-manifest requirements.
- Improve docs/ledgers: reconcile `WORK_LOG.md`/`DISCOVERY.md` with current
  `main`, fix stale facts, ensure recheck triggers are precise.
- Prepare GLM-5.2 memory-strategy analysis (paging/selective-precision design)
  that does not require the unavailable host, so it is ready the moment one is.
- Periodically re-attempt the top resource-blocked item to detect environment
  changes (more memory, GPU, network/credentials, off-box host availability).
Pick the smallest useful slice, gate it, implement, verify, record, commit, loop.

────────────────────────────────────────────────────────
SKIP RULES (formerly STOP/ESCALATE) — record and continue, never halt
────────────────────────────────────────────────────────
- No off-box host/cache for Item 1: try every route-around first; if still
  impossible, record the exact run packet + measured wall, mark BLOCKED-RESOURCE,
  and continue with the next workable item. This is a skip, not a stop, and it is
  NOT an Air quality verdict.
- Teacher cache validation fails on schema/tensor errors: this is a TOOLING bug,
  not a resource block. Fix it, add a regression test, and continue.
- Teacher metrics imply a strategic choice: record the data, your recommendation,
  and the next branch in the ledgers, then KEEP WORKING the highest-value
  workable slice. Do not block the loop waiting for a human; the human reads the
  ledger asynchronously.
- PV kill experiment fails: drop the RECOVER path and proceed to selective
  precision/fallback evaluation. Keep looping.
- GLM-5.2 memory strategy exceeds the 120 GB budget outside an explicit paging
  experiment: record the failure point, surface the next memory choice, skip
  forward.
- A slice would regress Air speed/defaults, violate artifact invariants, or
  require fabricating evidence: discard THAT slice (never ship it) and pick
  different real work. The invariants win; the loop still continues.
- Three consecutive failed slices on one item: mark that item stuck with the
  reason and move to the next allowed priority. Do not retry in a tight loop and
  do not halt.

The ONLY thing that ends this loop is an external kill. There is no self-issued
DONE that returns control to the human.

────────────────────────────────────────────────────────
PERIODIC CHECKPOINT (replaces "DONE") — write, do not halt
────────────────────────────────────────────────────────
Every ~10 committed slices, append a one-screen status snapshot to `WORK_LOG.md`
(do NOT stop after writing it):
- Air verdict status and evidence (or "blocked: <measured resource reason>").
- Speed status.
- Memory status.
- Quality/PPL/KLD status.
- GLM-5.2 decision/strategy status.
- Current top resource block and the exact packet needed to clear it off-box.
- What you are working next.
Then immediately continue the loop. The snapshot is a progress beacon for the
human to read asynchronously — it is never a reason to pause or wait.
