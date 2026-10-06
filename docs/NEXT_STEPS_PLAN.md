# KEEP/RAMP — Next-Steps Execution Plan (2026-07-08)

**Goal:** an optimized, validated, one-command `base → community-wow RC` pipeline that
runs GLM-5.2-REAP on rented GPU with no research-on-the-clock. GLM-4.5-Air is the cheap
proving ground. Strategy doc: `docs/GLM52_PIPELINE_READINESS.md`. Evidence log: `DISCOVERY.md`.

**SERVING TARGET (confirmed 2026-07-08):** GLM-5.2-REAP compressed by KEEP, served via
**MLX / E8P VQ + NAX on Metal, on ONE 128GB Mac M5 (this machine).** NVIDIA/CUDA is for
**production/research compute only** (huge REAP teacher forwards + recovery training that
don't fit locally) — NOT for serving.

**Guiding principles (settled this session):**
1. **Serving = Apple/Metal/E8P/NAX.** These Apple-only kernels are the correct, on-target runtime.
   **No CUDA E8P port, ever.** Affine quant / MLX-CUDA is relevant only as a research-velocity
   accelerator, never as the serving representation.
2. **CUDA role = production only.** The REAP bf16 teacher is ~700GB (350B) to ~1TB (504B) — it
   won't fit on this Mac (or 2 Macs), so rent big NVIDIA to generate imatrix + KD targets and
   train the surrogate recovery, then bring the compressed artifact home to serve on Metal.
3. **Final gates are measured HERE on Metal** (the serving backend) — clean, no cross-hardware
   transfer for the shipped RC. The only drift to watch: recovery trained on CUDA (surrogate +
   CUDA teacher logits) must be RE-VALIDATED on this Mac's E8P kernel before acceptance.
4. **BINDING CONSTRAINT — single-Mac serving memory.** ~90-95GB resident-weight budget on a
   128GB Mac (all experts resident; total params drive memory). => ~176B REAP fits at ~2bpw
   (~44GB); ~350B is tight at ~2bpw (~88GB); **504-526B needs ~1.4 bpw avg (~90GB) — ~30% more
   aggressive than Air's 2.036, a materially harder quant problem.** The REAP variant's size sets
   the bpw target and difficulty. OPEN: confirm whether a <=~350B REAP variant exists or ~504B is
   the floor.
5. Prove everything cheap on Air/Metal (this Mac) before renting NVIDIA.

Current completion: P1 ~85% · P2 ~60% · P3 ~15% · P4 ~45%.

---

## Phase A — Close P1 + validate the recovery step  (Air/Metal, ~1 heavy run)

**Why:** plan1 is the promoted RC but misses 3 wow gates; the weakest is instruction (top1
0.77) because plan1's recovery sidecar trained on route+math rows only. Adding instruction
rows is the direct fix and doubles as end-to-end validation of the recovery step.

**Tasks:**
- A0. **Source instruction distillation prompts from HuggingFace, do NOT hand-author or reuse
  the eval rows.** The existing teacher cache has only ~20 instruction rows/split and they
  overlap the eval set (train/test contamination). Instead: pick a standard instruction-following
  HF dataset matching the eval's `instruction` style (confirm the exact HF id first), sample a
  bounded set of prompts DISJOINT from the eval prompts, and run the teacher (bf16) forward once
  to cache their logits (cheap; forward-only, no training). These become fresh KD rows.
- A1. Assemble the recovery training-row set = plan1's route+math rows (existing cache) + the
  fresh HF-derived instruction rows (A0). This is row SELECTION over cached (prompt, teacher-logit)
  pairs — not a new dataset. Same recipe/hyperparams as plan1 (rank-4, layer-45 gate/up/down, KD loss).
- A2. Train the sidecar on the accepted-RC-or-plan1 seed (one heavy run, serial).
- A3. Eval on report/select/holdout 128-row caches; run `rc-summary` gates. Holdout instruction
  rows stay untouched by training → honest generalization measure (no contamination).
- A4. Re-measure Lane S against the golden baseline (must stay ≤1.15×).

**Acceptance:** instruction top1 ≥ 0.80 on ≥2 splits without regressing code/math/global;
ideally also pull mean-KLD ≤0.30 / p999 ≤3.0. If it clears all wow gates → promote it over
plan1; P1 → ~100%. If it lifts instruction but not the tail-KLD gates, still promote (strict
upgrade) and record the residual gap.

**Cost/risk:** low. One training run + evals. Risk: instruction rows trade off against
route/math — mitigate by keeping the domain mix balanced, not instruction-heavy.

---

## Phase B — P2: representation-pluggable one-command base→RC pipeline  (core deliverable)

**Why:** this is the artifact that makes GLM-5.2 cheap. Must run end-to-end from a single
command AND be able to target either E8P-VQ (Apple) or MLX-native-affine (CUDA-portable)
behind the same recipe/gates.

**Tasks (robustness prereqs first — they prevent wasted paid-GPU runs):**
- B1. **Engine⇄tier compatibility guard.** `vq_e1_routed_nax_e8` requires code_bits=8 on every
  group; mixed E8/E8P candidates crash at eval. Validate engine vs the candidate's tier map
  before eval (or pick the engine from the tier map). Locus: eval-step setup in the runner /
  `run_glm45_air_rc_pipeline.py`.
- B2. **Serial heavy-job lock.** Concurrent full-model loads OOM (observed). Add a global
  heavy-job lock in the runner so model-loading steps serialize. Locus: `src/mlx_vq/build/executor.py`.
  (Already shipped: Lane S golden-baseline gate fix; fail-loud output guard.)
- B3. **Fail-loud unit test + sweep reround>0 refinement** (finish the fail-loud guard: also
  assert `reround_action_count>0` for reround recipes; add a runner test).

**Tasks (the pipeline itself):**
- B4. **Full base→RC recipe** `recipes/glm45air__base_to_rc__*.yaml`:
  `collect-imatrix (affinity) → materialize-sweep → P-step closed-form scale refit → low-rank
  KD recovery sidecar (instruction-balanced from Phase A) → RC gates`. Root at the base model,
  not the r26 seed.
- B5. **Pluggable quantized representation.** Introduce a `representation` recipe field
  (`e8p_vq` | `mlx_affine`) that selects codebook + eval engine + gate profile, with the
  driver/ledger/gates unchanged. E8P path = current custom kernels (Apple fast-path);
  mlx_affine path = `mx.quantize` + `mx.gather_qmm` (Metal today, CUDA-portable).
- B6. **Fold in the keeper levers** as default recipe steps: AGQ `affinity_weighted_importance`
  as the sweep default; the P-step closed-form scale refit as a pre-recovery step.

**Acceptance:** `keep build recipes/glm45air__base_to_rc__…yaml` runs end-to-end on Air with
one command and reproduces/exceeds plan1 (E8P path); the same recipe with
`representation: mlx_affine` runs to a gated candidate. → P2 → ~95%.

**Cost/risk:** B1–B3 are CPU (low risk, test-guarded). B4–B6 need heavy validation runs
(serial). Risk: pluggable representation touches the eval-engine/gate wiring — keep the driver
untouched; representation only swaps codebook+engine+profile.

---

## Phase C — (DEMOTED) affine quant as a research-velocity accelerator only  (optional)

**Status change (2026-07-08):** serving is E8P/Metal on this Mac, NOT NVIDIA — so affine quant is
**no longer a serving decision gate.** Its only remaining value is *research velocity*: MLX-CUDA
can eval affine quant (GatherQMM) fast on a rented GPU during iteration, as a PROXY, while the
real acceptance gate is always E8P on Metal (this Mac). Do this ONLY if CUDA-side eval iteration
speed becomes a bottleneck during REAP production. Otherwise skip. The former "does affine clear
the wow bar for NVIDIA serving" question is moot.

**Why (original, now moot):** KEEP's 2-bit quality edge historically came from E8P VQ beating affine.

**Tasks:**
- C1. Run the Phase-B pipeline with `representation: mlx_affine` on Air (Metal) to a gated RC.
- C2. Compare per-domain top1 / mean-KLD / p999 vs the E8P RC (plan1/Phase-A). Quantify the
  E8P-vs-affine quality gap under identical methodology.

**Acceptance (decision gate):**
- If affine clears the wow bar (or within a small, recoverable margin) → **NVIDIA serving is
  viable as-is**; no CUDA kernel work needed. Proceed to Phase D on either backend.
- If affine falls materially short → NVIDIA serving needs a **CUDA E8P kernel port** (large,
  deferred) OR you commit to Apple-silicon serving. Record the gap; escalate the decision.

**Cost/risk:** low (Air/Metal, reuses Phase B). This is the highest-information cheap experiment
in the whole plan.

---

## Phase D — P3: GLM-5.2-REAP RC  (only after A–C pass; heavy)

**Preconditions:** Phase B one-command pipeline validated; Phase C decision made; serving
target chosen.

**Tasks:**
- D0. **Confirm the REAP variant SIZE against the single-Mac serving budget (gating).** ~90-95GB
  resident-weight budget on 128GB. <=~350B => ~2bpw target (Air methodology transfers). 504-526B
  => ~1.4 bpw avg target (aggressive mixed-precision; may not reach Air's wow bar; escalate if so).
  If no variant fits even at aggressive bpw, single-Mac serving is infeasible — reconsider target
  (smaller variant / 2-Mac serve / accept lower bpw). This decision sets the whole D-phase bpw goal.
- D1. Pick the REAP variant that fits, download the **safetensors** version (not GGUF).
  Delete stock GLM-5.2 once chosen.
- D2. **Model-profile parameterization** — strip Air hardcoding: model id, layer/dim/expert
  counts, hard-layer set (derive via an attribution probe, not a fixed list), prompt sets,
  model-scoped golden-q2 path, tier floors. Land a `model_profile` config the runner reads.
- D3. Regenerate teacher caches (report/select/holdout) + imatrix calib set for REAP.
- D4. Measure the **golden-q2 baseline on the target serving hardware** (Apple or rented NVIDIA).
- D5. Run the validated pipeline → REAP RC. On rented GPU: follow the cost-minimal order
  (dry-run → golden → quant+refit → bare-quant gates → recovery training → full gates), with
  ledger checkpointing so a failed paid run resumes without re-paying completed steps.

**Acceptance:** a gated REAP RC on the target backend. → P3 toward completion.

**Cost/risk:** high (large model, possibly multi-device + paid GPU). De-risked by A–C.

---

## Deferred (do NOT start unless triggered)
- **NAX Option A** (fused decode-in-matmul C++ kernel) — real long-context speed, off the RC
  critical path.
- **CUDA E8P kernel port** — only if Phase C's affine check fails AND NVIDIA serving is chosen.
- Further P1-only polish beyond Phase A.

## Sequencing & parallelism
- A and B1–B3 can run together (A is one heavy run; B1–B3 are CPU) — respect the serial
  heavy-job rule for any model loads.
- B4–B6 depend on A (recovery row mix) and B1–B3 (robustness).
- C depends on B5 (pluggable representation).
- D depends on B (validated pipeline) + C (representation decision).

## Recommended immediate action
Start **Phase A** (instruction-row recovery run) and stand up **B1–B3** (CPU robustness fixes)
in the same pass. That advances P1→~100%, P2 robustness, and validates the recovery step —
all on cheap Air/Metal.
