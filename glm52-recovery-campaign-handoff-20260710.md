# GLM-5.2-REAP-KEEP-504B recovery-campaign handoff — next-session execution prompt

## Standing authority (do not re-ask)

Jack gave **FULL APPROVAL** on 2026-07-10 covering: the FP32 source-teacher-cache
architecture, unbounded Codex orchestration toward the community-wow RC goal,
and the two recovery lanes (producer pipelining + quality recovery). Orchestrate
Codex workers (GPT 5.6 Sol) at `--effort high` or `--effort medium` by task
complexity through the codex-companion runtime
(`~/.claude/plugins/cache/openai-codex/codex/*/scripts/codex-companion.mjs`),
per the `codex-orchestrator:orchestrating-codex` skill: Claude plans/verifies,
Codex implements; adversarial-review every substantive wave; exclusive file
ownership per concurrent write worker; always pass `--effort`, never `--model`.

Jack's verdict on quality: he is **happy** with 48% top-1 at 1.03 bpw as a
first trial on the already-REAP-pruned (1T→504B) model. Do not frame it as
failure. The campaign now RECOVERS quality on the current candidate.

## Hard guards (unchanged from project authority)

- No push, merge, publish, Hugging Face upload, rented hardware, or paid
  service without explicit approval (AWS FP8 comparison is parked until Jack
  approves spend).
- Leave `GLM_MLX_WIRED_LIMIT_GB` / `GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB` unset.
- Only the **selection** split (22 prompts / 255 positions) may guide recovery;
  holdout tuning forbidden; report split for reporting.
- Heavy model operations hold `.keep-heavy-job.lock` (advisory flock at repo
  root). `runs/` and the two handoff .md files at repo root are protected.
- `artifacts/` is **gitignored** — evidence lives on disk, referenced by hash.
- Tests: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest`.
- Codex worker sandboxes have **no Metal** — all real MLX compute runs on the
  host via Bash. Launch long host jobs with `nohup ... & disown` (plain
  run_in_background children die if the Claude session restarts).

## Repo state

Branch `keep-glm52-pipeline-and-p1-lock` at `9d74c5dc` (24 commits this
campaign from `6863815c`, nothing pushed). Session task tracker holds tasks
#1–#9 matching the plan below. Working tree clean except protected untracked
paths.

Pinned model: `0xSero/glm-5.2-reap-504B-v2` @
`6c9241aa05fb243a0edb7c804c213ec1cf5c920d`; snapshot at
`$HOME/.cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2/snapshots/6c9241aa05fb243a0edb7c804c213ec1cf5c920d`.

## Key evidence and numbers (2026-07-10)

| Fact | Value / path |
|---|---|
| Accepted artifact | 98,433,923,808 bytes / 1.5934 bpw whole-main; routed 1.03125 bpw semantic; raw `quip_e8` + `max_abs`, zero levers |
| Teacher cache | 4 complete runs, **all 66/66 shards byte-identical**; canonical: `artifacts/quality/glm52-teacher-cache-fp32-v5-20260710` (+ ledger/checkpoints siblings); all runs memory-dirty → `release_eligible=false` |
| Source route traces | `artifacts/quality/glm52-source-route-traces-v5-20260710` (release-class authority) |
| Candidate cache (diagnostic) | `artifacts/quality/glm52-candidate-cache-fp32-20260710` (+ sibling `-diagnostic-manifest.json`, `-ledger.jsonl`) |
| Baseline comparison | `artifacts/quality/glm52-candidate-eval-comparison-20260710.json`: mean KLD **1.6340**, top1 **0.4786**, p999 KLD 13.07, PPL ratio 4.093 — every frozen check fails (gate: 0.30 / 0.85 / 3.0 / 1.05) |
| Capture faithfulness | `benchmarks/diag_glm52_candidate_capture_consistency.py` → byte-exact match batched vs plain forward; teacher true-token logprob −5.807 vs candidate −7.907 |
| Recovery stats | `artifacts/quality/glm52-recovery-wave1-stats-20260710` (75 sparse layers, selection split, 17.5 min with pipelined pool ≈ 2× speedup) |
| Attribution | `artifacts/quality/glm52-recovery-wave1-attribution-20260710.json`: worst = layer 77 (err 2007, 2× runner-up), then 75, 74, 70, 76, 72, 73, 71 — damage concentrated in layers 70–77 |
| 2-layer probe re-eval | `artifacts/quality/glm52-recovery-wave1-reeval-20260710.json`: KLD 1.634→**1.566**, top1 0.4786→**0.4888**, PPL 4.09→**3.77** — lever proven |
| Worst-16 rematerialization | **detached job launched ~19:26 PDT**, output `artifacts/quality/glm52-recovery-wave1-mixed16-artifact-20260710`, log `producer scratchpad recovery-remat16.log` (scratchpad dir may differ next session — check `pgrep -f run_glm52_recovery` and the output dir contents); resumable with `--resume` |
| Frozen invariants | recipe `recipes/glm52_reap_504b_family_evidence_v2_20260710.yaml` validate+dry-run exit 0, step key `48146c10b53be9db`; no-cache checker exit 2 byte-identical to `artifacts/quality/glm52-family-gate-raw-evidence-20260710.json` |
| Producer identities JSON | `artifacts/quality/glm52-teacher-cache-artifact-identities-20260710.json` (required `--artifact-identities-json`) |

## Gotchas learned this session (do not relearn)

- Never route reads through `/dev/fd` (macOS dup shares file offset → corrupt
  safetensors headers). The resolver reads HF `blobs/` with fresh `O_NOFOLLOW`
  descriptors + identity checks.
- The 0/0 pageout release rule fails on a busy machine because wiring tens of
  GB displaces other apps' pages; idle rate is 0/0. Needs Jack's quiet-machine
  window (task #7).
- Producer test suites fake `mlx` in `sys.modules`; hygiene fixtures restore
  everything — keep it that way or later Metal tests abort the interpreter.
- BF16 mx arrays cannot view-convert to NumPy; go through `astype(float32)`.
- The composite loader's parent-dir attestation compares only stable identity
  (dev/inode) — do not reintroduce mtime comparisons of shared dirs.
- `--allow-non-release-teacher-cache` marks the whole evidence chain
  `diagnostic_only`; the family gate correctly refuses it. Release evidence
  needs clean caches.
- Attribution/rematerialize/reevaluate are decode-heavy: run detached, poll.

## Execute in this order

### 1 (P0). Re-evaluate the worst-16 artifact — likely ready at session start
Check the detached job finished (`conversion-manifest.json` +
`recovered-groups/` + 225-file `artifact/` under
`glm52-recovery-wave1-mixed16-artifact-20260710`; resume with the same
`rematerialize` command + `--resume` if not). Then run `reevaluate` exactly as
the 2-layer probe (same args; swap `--recovery-conversion-dir` to the mixed16
dir, output `artifacts/quality/glm52-recovery-wave1-reeval16-20260710.json`)
detached, and record the coverage curve: 1.634 → 1.566 (2 layers) → ? (16
layers). The slope decides investment: strong slope → prioritize full-coverage
rounding passes; weak slope → jump to mixed precision (task 3) immediately.

### 2 (P0). Adversarial review of the unreviewed waves — run in parallel with 1
`codex-companion adversarial-review --wait --base 0fe60378` over commits
`ed56f412`/`9d74c5dc` focus: decode-pool byte identity under failure paths
(cancellation, worker exceptions), thread shutdown on all paths,
selection-split-only enforcement in `glm52_recovery.py`, mixed-artifact
provenance integrity in `glm52_recovery_materialize.py` (can a recovered group
masquerade as a seed group?), and stats-capture non-interference with logits.
Route findings to fix workers with exclusive ownership; re-verify; commit.

### 3 (P1). Per-layer mixed precision — the decisive experiment
Design + implement per-layer codebook/rate mixing: worst 8–16 layers (from the
attribution ranking, re-ranked after task 1's artifact) at ~2× rate (higher-rate
E8P codebook — see Air's e8p usage), remaining layers unchanged. Touches: the
routed artifact format/conversion manifest (per-layer codebook provenance),
`bind_glm52_decoder_layer_vq_experts`/adapter kernels for mixed codebooks, the
recovery materializer, and audits that assume a single global codebook. Budget
≈ +0.8 GB per doubled layer; keep total ≤ ~112 GB. Then rematerialize worst-8,
reevaluate, and compare against the task-1 curve. Adversarial-review this wave
(it changes artifact trust surfaces). Expectation: the largest single quality
jump available; also record the new artifact accounting honestly (the accepted
98.434 GB identity changes — that is sanctioned by Jack's recovery approval,
but record it as a new candidate identity, never overwrite the accepted one).

### 4 (P1). Full 75-layer importance+Hessian pass (overnight)
Zero new code: `rematerialize --worst-layer-count 75` (verify diagonal-Hessian
rounding is actually engaged in the materializer — if it is importance-only
today, wire the Hessian term first). ~13 h detached, resumable. Then
`reevaluate`, then re-run `attribute` against the recovered artifact to
re-rank (the worst list changes after fixes). Best launched at end of a
working session.

### 5 (P2). Port EBSS + learned rotations
Two sequential Codex lanes adapting `src/mlx_vq/quality/ebss.py` and
`learned_rotation_{training,materialization}.py` to GLM52 E8 groups,
selection-guided, stacking on top of whatever 3+4 produced. Scope each lane
with a recon task first; TDD; adversarial review.

### 6 (P2). Candidate route-trace capture + route/math compare
Add `--route-trace-root` capture to the candidate eval forward (mirror the
teacher producer's same-pass capture; release-class authority binding the
candidate identity), then `benchmarks/check_glm52_route_math.py compare`
against `glm52-source-route-traces-v5-20260710`. Completes blocker 3's
machinery and diagnoses route-divergence vs weight-error for recovery.

### 7 (P3). Quiet-machine window — REQUIRES JACK
Ask Jack to schedule (or grant permission to close apps). Then, in one window:
teacher-cache producer rerun (now ~2× faster; bytes will match the four
byte-identical runs — only the memory evidence is new), release-class candidate
cache (no diagnostic flag), and the same-machine benchmark `pair` runs (3 clean
reps of prefill_1k + decode_128). All must show 0/0 pageouts/swapouts.

### 8 (P3). Walk the family gate v3→v6
With release-class evidence: strict cache audit → v3; recomputed cache-pair
eval passing frozen thresholds → v4; route evidence → v5; benchmark → v6
(`family_gate_pass=true`, checker exit 0). Author the schema-v3+ family
evidence recipe binding it into the build graph. CAVEAT: the recovered artifact
has a NEW identity — composite audit, teacher-metadata gate inputs, and
candidate identity envelopes must be reissued for it (deliberate reissue, per
the identity-contract rules in `docs/GLM52_PIPELINE_READINESS.md`).

### 9 (P4). Product acceptance + publication pack
Raw-HF end-to-end `keep build` through the real 23-step DAG with product-level
interruption/resume at several phases; real-artifact smoke of `keep chat` and
`keep report`; render the release model card (unlocks at gate pass);
prepare — but do not upload — the Hugging Face publication pack.

## Parked (do not start without Jack)

- Pipeline floor-baseline improvements (make importance+Hessian the default
  materialization) — after the current candidate is recovered.
- AWS GPU rental for base GLM-5.2 FP8 comparison — external spend approval
  required.
- Any Hugging Face upload — explicit approval required.

## Reporting style

Jack wants zoomed-out plain-language updates — invoke the `status-update`
skill (`~/.claude/skills/status-update/`): re-ground what the project is,
per-workstream bold verdicts, honest failing/unknown items with trend, ordered
costed path, end on the genuine open question.
