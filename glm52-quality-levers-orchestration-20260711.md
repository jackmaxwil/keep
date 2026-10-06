# GLM-5.2 quality-lever orchestration state — 2026-07-11 (evening)

Orchestrator: Claude Opus 4.8; workers: Codex via companion. Window: to 10am
Sun 2026-07-12. Jack's priorities: **5 → 4 → 3 → 2 → 1 → 6**, parallelized
where files don't overlap.

| P | Lever | Plain meaning | State |
|---|---|---|---|
| 5 | Adapters + router-KD | port the Air-proven trained low-rank correction + router distillation to GLM-5.2 | recon done → build wave |
| 4 | Rotations | QuIP#-style incoherence rotations on routed projections | recon done → build wave |
| 3 | EBSS | selection-error-driven scale search replacing the scale formula | recon done → build wave |
| 2 | LDLQ lever | wire `ldlq_feedback.py` (landed `bd7f12e4`) into the materializer as a declared lever | integration wave |
| 1 | Combined seed | re-run worst8-E8P seeded on Full75 (needs seed-contract extension) | integration wave |
| 6 | YAQA sketches | feasibility spike: Hessian sketch on this hardware | deferred last |

## Scope matrix (conflict control)

- `src/mlx_vq/convert/glm52_recovery_materialize.py` = CONFLICT ZONE (P1+P2+P3+P4
  all touch it) → ONE serialized integration worker, single brief, after the
  parallel new-file wave.
- Parallel new-file builds (disjoint): P5 harness (new files under
  `src/mlx_vq/quality/` + `benchmarks/`), P4 rotation module (new file), P3 EBSS
  search module (new file).
- Campaign recipe (`recipes/glm52_recovery_campaign_v1_20260711.yaml`) edits:
  serialize into the integration worker too.

## Codex Metal access (Jack authorized)

Companion patched (v1.0.5, `codex-companion.mjs`): new `--full-access` flag
(requires `--write`) maps to codex `danger-full-access` sandbox → Metal
available. **Default unchanged** (workspace-write). Grant ONLY to workers whose
brief forbids: git mutations, network, `artifacts/` writes, model loads, big
allocations. Patch is in plugin cache — re-apply after plugin updates.

## Job ledger (update as waves progress)

- Recon R1 (P5 port contract): task-mrh9q9ns-v13yg5 — done, collect
- Recon R2 (P4 rotations map): task-mrh9q9ui-abajij — done, collect
- Recon R3 (integration map): task-mrh9qa10-agz8jg — done, collect
- Recon outputs saved: docs/superpowers/specs/2026-07-11-glm52-{adapter-port,rotations,materializer-integrations}-recon.md
- W2a P5 harness (FULL-ACCESS/Metal): task-mrha1u1n-b29acn — running
- W2b P4 rotation_search: task-mrha1u8d-q1el1j — running
- W2c P3 ebss: task-mrha1ues-3fj14k — running
- W2 wave: DONE, committed 56ed6779 (adversarial review: 5 findings -> fixed, 26 tests green)
- W3 materializer integration: DONE, committed 23746a64 (recipe YAML additions REVERTED — strict campaign config validation rejected them, correctly; new experiments need proper config-change process)
- Provider glue: DONE, committed d3c61c97; baseline config at artifacts/quality/glm52-training-baseline-accepted-20260711.json (+ config_path/profile_path/source_index_path fields)
- Adapter real-model smoke: FAILED with '[scatter] Cannot calculate VJP with respect to indices' at glm52_adapter_training.py:736 — fix job task-mrhbz851-xig5nf running (full-access)
- P1 combined-seed run: LAUNCHED 22:06:56 PID in scratchpad/combined-seed/PID, output artifacts/quality/glm52-recovery-worst8-e8p-full75seed-artifact-20260711, ~2.5h, resumable; teacher-manifest-body-sha 5e25d36d13a72a72e2112094f2ddbf0e34cbf269c1f54f342375d715d1d5aef3
- Adapter vjp fix: committed d372485c; REAL-MODEL SMOKE PASSED 22:17 (loss 0.659, sidecar manifest at artifacts/quality/glm52-adapters-smoke-20260711/, release_eligible=false)
- P1 combined-seed: running (1/24 at 22:19), watcher b5swe1mnr, output ...worst8-e8p-full75seed-artifact-20260711
- P5 FULL training run QUEUED behind it (scratchpad/adapter-full/launch.sh, PID file there): layers 75,74 all projections rank 4 steps 300, output artifacts/quality/glm52-adapters-worst2-r4-20260711/

## Combined-seed candidate: NEW BEST (2026-07-12 01:21)

- Materialization COMPLETE 24/24 (00:35), audit PASS (candidate f2e7a9a8..., 104.8 GB)
  after two fail-closed validator extensions (recovery-seeded manifest shape
  `01840181`; two-hop alias roots).
- **Silent-wrong-bytes eval bug found + fixed (`16f570b3`)**: the reeval binder
  assumed inherited groups == accepted artifact; first eval scored worst8-alone
  bytes (0.5481, identical fingerprint exposed it). Binder now resolves through
  authenticated candidate links.
- **TRUE combined verdict (reeval2): OVERALL top-1 0.5895** — selection 0.6290 /
  holdout 0.5750 / report 0.5624; mean KLD ~0.80 (-24% vs Full75); PPL 1.34-1.69;
  p999 4.85-6.19. Trajectory: 0.4786 -> 0.5633 (Full75) -> 0.5895 (combined).
  Gates still fail (0.85); best candidate yet.
  Evidence: artifacts/quality/glm52-recovery-worst8-e8p-full75seed-reeval2-20260712.json
- Adapter training relaunched after the eval window (~01:23), on the ACCEPTED
  baseline (provider extension to recovery candidates = next session).


## 2026-07-12 — adapters trained; memory wall solved (unbounded RC campaign)

**Adapter training now works on the real 504B model, memory-safe.** Fixed a
step-0 crash (_StopGradientIndexer None passthrough, 7e7f15f1) then a cascade of
OOMs. Root causes found by MEASUREMENT (per-stage mx.get_active/peak):
- model resident ~35GB; a full forward wires it to ~89GB (all MoE experts).
- per-step suffix (trained-layer..head) forward+backward is the big transient;
  it scales with SUFFIX DEPTH (layers between the trained layer and the head),
  NOT sequence length or positions.
Memory fixes committed (all flag-gated, byte-identical when off; 20 tests):
- bound MLX cache + per-step mx.clear_cache (10825bc4)
- gradient checkpointing on non-trainable suffix layers, GLM52_TRAIN_GRADIENT_CHECKPOINT (6b7fd391)
- **disk-backed frozen-prefix boundaries, GLM52_TRAIN_BOUNDARY_DISK (fdbae9c1)** — compute once, load one/step
- truncate forwarded sequence to supervised positions (58b9a4e9)
- per-step progress + peakGB logging (GLM52_TRAIN_PROGRESS)
**Winning config: --layers <topmost worst layer> (suffix depth 1) + disk boundaries
+ max-positions 64 → peak 104GB, ~1s/step, 300 steps in ~10min.**
First adapter: artifacts/quality/glm52-adapters-layer77-r4-20260712 (trained on ACCEPTED baseline).

**Eval-path binding shipped** (317e46cf): reevaluate --adapter-sidecar-dir/--expected-adapter-manifest-sha256,
fail-closed auth (parent-candidate identity, routed-inventory, shapes).

**recovery_candidate_baseline_provider** (2c75af80): train adapters against a recovery
candidate (loads accepted + binds+audits candidate; identity matches reeval).
Combined-seed training config: artifacts/quality/glm52-training-baseline-combined-seed-20260712.json.
NOTE: Codex's reeval-side refactor was REVERTED — it imported a runtime module before
the fail-closed audit gate (broke 8 reeval tests; its sandbox never ran them). Host-verify always.

**IN FLIGHT:** retraining layer-77 adapter against combined-seed candidate (0.5895), then
reeval combined-seed+adapter (script: scratchpad/combined-adapter-reeval/run.sh) for the
GO/NO-GO: does the trained-parameter class beat 0.5895 toward the 0.85 gate?
Baselines: original 0.4786 -> Full75 0.5633 -> combined-seed 0.5895 (gate 0.85).
Companion --full-access confirmed working (Codex self-runs Metal tests).
NU176-526B newer base: deferred (337GB disk / 3x memory; resets evidence chain).

## Morning checklist (next session)
1. Verify combined-seed completed 24/24 + manifest complete; run audit (precedent: session-3 audit argv, substitute dirs + manifest sha) + frozen reeval (ledger seq-3 argv precedent) -> compare vs 0.5633/0.548.
2. Check adapter-full run.log: final_loss trajectory + sidecar artifact.
3. NOT YET BUILT: eval-path binding of adapter sidecars into the composite (needed to reeval the adapter candidate); EBSS/rotation/LDLQ real runs (levers integrated, off by default); campaign evidence registration CLI; new-experiment declarations via proper config-change process.
4. Companion --full-access patch is in plugin cache v1.0.5 (reverts on plugin update).

## Overnight execution schedule (heavy jobs serialize on .keep-heavy-job.lock)

1. After W3 lands+verified+committed: REAL-MODEL smoke of adapter CLI
   (benchmarks/finetune_glm52_low_rank.py, --layers 75 --steps 2) to prove
   memory viability — first ever real-model run of the harness.
2. If viable: full adapter training run on worst8 layers [75,74,76,72,73,77,71,70]
   (selection split only, rank 4-8, hours; lock held; nohup+disown; PYTHONUNBUFFERED).
3. After adapters: P1 combined-seed rematerialize via new --seed-recovery-artifact-dir
   (seed = Full75 artifact artifacts/quality/glm52-recovery-wave1-mixed75-artifact-20260710,
   manifest sha 544163a747..., audit sha c65f98ed23...; GLM52_E8P_BACKEND=metal; ~2.5h).
4. Morning: frozen 66-row reevals of both candidates (precedent argv in ledger seq 3;
   substitute --recovery-conversion-dir + --output-json); compare vs Full75 0.5633.
5. P6 YAQA sketch feasibility spike: only if window remains.

Key facts: worst8-E8P reeval = 0.548 overall top-1 (vs Full75 0.5633; gates need 0.85).
Adapter payload rank8 ~132MiB/layer -> bound layer set. Teacher cache lacks router
supervision -> router-KD stubbed. Companion --full-access patch active (v1.0.5).

## Ground rules carried forward

Heavy runs hold `.keep-heavy-job.lock`; selection-only tuning; no push/publish;
`GLM_MLX_WIRED_LIMIT_GB` unset; artifacts/ gitignored; handoffs at repo root
protected. Byte-identity required only for swaps of accepted paths; NEW levers
use determinism + objective-dominance + honest diagnostic labeling.

## Context for next session (if compacted)

Today's earlier campaign landed: fused Metal E8P search (87×), worst8-E8P
complete + audited (candidate `0bbd540c…`) + reevaluated (top-1 ≈0.548 vs
Full75 0.5633 → re-rounding plateau), ledger reconciled via new authorize-retry.
Commits `e681bca0..bd7f12e4` on `keep-glm52-pipeline-and-p1-lock`. Main handoff:
`glm52-e8p-speedup-campaign-handoff-20260711.md`.
