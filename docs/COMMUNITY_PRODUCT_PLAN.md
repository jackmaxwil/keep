# KEEP as a Community Product — Intent + Brainstorm (2026-07-09)

## The contract (user intent, restated)

1. **GLM-4.5-Air is DONE.** plan1 (top1 0.852 / KLD 0.314 / Lane S 1.148x) is accepted as
   passing the user's personal bars. Air was the proving ground; no further Air quality work.
2. **New #1 priority:** `keep build <recipe>` from a **fresh HuggingFace download of
   GLM-5.2-REAP** to a **community-wow RC** — heavily quantized, fast on Metal, good quality,
   repeatable — in **one command**. After exact pinned-source accounting, the user accepted
   the revised size target of **98,433,923,808 tensor-payload bytes / 1.5934433
   whole-main-model bpw** (about 98.434 GB) for uniform routed E8 plus a lean
   source-precision non-routed repack; this remains
   a whole-model projection until all 225 routed groups exist and pass audit. The
   37,121,488,608-byte non-routed component is now physically packaged and audited.
   The registered six-step low-level recipe also strictly binds that package plus
   a bounded E8 trio into layer 3 and produces a finite BF16 decoder-layer forward
   with no dense routed parameters. A separate checked four-step runtime-preflight
   recipe now proves the pinned 21-full/57-shared IndexShare static contract plus a
   tiny full-to-shared cache/top-k handoff, exact local tokenizer/chat readiness,
   and two-token `generate_step` compatibility on an in-memory tiny fixture. Its
   final header-only full-bind gate sees six present routed groups and blocks on the
   exact 219 missing groups. The checked run completed the first three probes and
   then failed loud as designed (`4.76 s`, 606,306,304-byte maximum RSS, zero swaps;
   full-bind exit 2, overall build exit 1). This is not production whole-model bind,
   generation, long-context IndexShare, quality, speed, or RC readiness: the tiny
   generation record explicitly uses no production artifact or tokenizer, and the
   preflight reads no tensor payloads or constructs no model.
   Before recovery tuning, a separate checked three-step contract recipe now
   freezes 66 report/selection/holdout prompts, route/math/instruction domain
   quotas, exact full-vocabulary source-relative metric requirements, community
   thresholds, holdout prohibition, and the same-machine pinned-FP4 benchmark
   baseline. This is an immutable evaluation authority, not teacher-cache,
   candidate-quality, speed, or family-gate evidence.
3. **The repo itself is a deliverable**: a batteries-included tool a stranger can adopt —
   docs/guides, copy-paste examples, ergonomic CLI, **human-composable YAML** (current recipe
   YAML is explicitly "unapproachable/complicated"), run tracking + visualizations, legible
   per-domain quality/benchmark reports, testing infra, extensibility.
4. **UX bar:** a stranger with a Mac runs one command → gets a wow model → can *see* what
   happened, how fast, and how good, without archaeology. Quality is a long-term goal;
   ordering remains run → run fast → run well.

## What the adoptable tools do (research notes)

**LLaMA-Factory** (the adoption benchmark):
- ONE CLI (`llamafactory-cli`) with verbs: `train / chat / api / export / webui / env` —
  every verb takes **one YAML file** as its sole argument.
- The YAML is FLAT and human-scale (~15 keys): `model_name_or_path`, `template`, `dataset`,
  `finetuning_type`, `output_dir`, lr/epochs. No graphs, no plumbing. Power lives in
  documented argument groups (Model/Data/Training/Finetuning/Generating), all defaulted.
- **Dataset registry** (`dataset_info.json`): datasets referenced by short name.
- WebUI (LlamaBoard) is just a YAML-constructor that shells to the CLI — the CLI is the API.
- Monitors: TensorBoard/W&B/MLflow/SwanLab pluggable.
- Docs shape: Installation → Data Prep → Quickstart → Method guides → Eval → FAQ, plus a
  "three-command quick start" (train → chat → export).

**mlx-lm**: `mlx_lm.convert --hf-path X -q` = download + convert + quantize in one line;
`mlx_lm.generate` / `.lora` / `.fuse`; and the killer community loop: **one-command upload
back to the Hub with an auto-generated model card** (the mlx-community flywheel).

**axolotl**: an `examples/` tree full of ready-to-run YAMLs per model family — the docs ARE
runnable configs.

## Gap analysis — what KEEP has vs needs

| Capability | KEEP today | Community bar |
|---|---|---|
| One-command pipeline | ✅ `keep build` end-to-end (proven on Air, ~11 min) | point it at fresh HF (GLM-5.2-REAP) |
| Repeatability | ✅ strong (content-addressed ledger, resume, build_steps provenance) | surface it in UX/docs — it's a selling point |
| Recipe YAML | ❌ low-level step graph (step ids, external_inputs paths, gate profiles) — expert-only | flat high-level recipe that COMPILES to the graph |
| CLI | ~ `keep validate/build/promote/plan-next` — builder-oriented | add user verbs: get/chat/serve/report/publish/doctor |
| Fresh-HF start | ❌ assumes pre-converted local artifacts | `keep get <hf-id>` download+convert as a recipe op |
| Run visibility | ❌ ledger JSONL + scattered JSONL evidence | `keep report`: HTML/terminal run report (curves, domains, gates, timeline) |
| Metrics legibility | ~ rc-summary JSON (expert-only) | model-card-style report vs teacher + baselines |
| Docs | ❌ internal (WORK_LOG/DISCOVERY culture) | Quickstart-in-60s README, concepts, how-tos, FAQ |
| Examples | ~ recipes/ exist but are machine-shaped | examples/ of runnable, commented, high-level recipes |
| Extensibility | ~ op registry + model adapter exist, undocumented; Air hardcoded | model profiles, dataset/calib registry, documented plugin points |
| Testing | ✅ 170+ unit tests, dry-run | tiny-model smoke test: full pipeline in minutes on CI |
| Publish loop | ❌ none | `keep publish`: push RC + auto model card to HF Hub |

## Brainstorm — the batteries

### A. CLI surface (single `keep` entrypoint, LLaMA-Factory verb style)
- `keep get <hf-id>` — download + convert a fresh HF model into a KEEP-ready artifact (mlx-lm convert pattern; becomes a recipe op so `build` can start from nothing).
- `keep build <recipe>` — the one command (exists; keep polishing).
- `keep eval <artifact>` / `keep bench <artifact>` — standalone quality/speed checks.
- `keep chat <artifact>` — talk to the RC immediately (the wow moment; nothing sells like chatting with a 2-bit model that's good).
- `keep serve <artifact>` — OpenAI-compatible local API.
- `keep report <run>` — render the run report (see C).
- `keep runs` — list runs, status, resume points.
- `keep publish <artifact>` — push to HF Hub with auto-generated model card (metrics table, bpw, speed, recipe provenance). The community flywheel.
- `keep doctor` — environment/memory/Metal/disk preflight ("will a 504B REAP fit? here's the math").

### B. Human-composable YAML — two layers
- **High-level recipe** (what users write; flat, ~10 keys, all defaulted):
  ```yaml
  model: 0xSero/GLM-5.2-REAP-NU176-526B   # HF id or local path
  target: mac-128gb                        # serving budget preset -> bpw math
  quality: wow                             # wow | balanced | fast (gate profiles + recovery depth)
  calibration: default                     # or a registered/custom prompt set
  recovery: auto                           # auto | off | {layer/rank overrides}
  output: ./runs/glm52-reap-wow
  ```
- **Compiler** expands this into today's low-level step graph (which stays — it is the
  reproducibility record). `keep build` accepts either layer; `--emit-plan` shows the
  compiled graph. Progressive disclosure: novices never see step ids; experts keep full power.
- **Registries** (LLaMA-Factory dataset_info pattern): `models/` profiles (arch dims, hard
  layers, tier floors — also required for GLM-5.2 anyway) and named calibration/prompt sets.
- **Presets** encode our hard-won findings as defaults: AGQ affinity importance, P-step scale
  refit, uniform-representation guard, golden-baseline Lane S, speed-over-cleanliness mode.

### C. Run tracking + visualization
- Per-run directory (exists) + `keep report` renders **one HTML file**: step timeline w/
  durations, training loss curve, per-domain top1/KLD table with gate badges, speed vs q2,
  memory footprint, bpw, provenance (recipe hash, build_steps) — plus a terminal summary.
- Live progress: rich step/ETA output during `build` (today: raw subprocess logs).
- Optional W&B/TensorBoard emitters later (monitor plug-point, not core).

### D. Legible quality story
- **Model-card report**: per-domain metrics vs teacher AND vs naive q2 baseline ("KEEP vs
  llama.cpp Q2 at same bpw" is the community-wow framing).
- Optional deeper eval: lm-eval-harness integration as a `quality: wow+bench` step (later).

### E. Docs (the batteries manual)
- README: what/why + 60-second quickstart + a results table + one screenshot of the report.
- docs/: Installation · Quickstart (small model, minutes) · Concepts (imatrix → E8P VQ →
  KD recovery → gates, with a diagram) · How-to (new model / custom calibration / tune bpw /
  resume a run / publish) · Recipe reference (both layers) · Hardware guide (memory budgets:
  what fits on 64/128/192GB) · Troubleshooting/FAQ.
- `examples/`: runnable, commented high-level recipes (axolotl pattern) — Air-small,
  Air-full, GLM-5.2-REAP.

### F. Extensibility points (document + harden what exists)
- Model profiles (new arch = one YAML + adapter mapping), op registry (documented), codebook/
  representation plug point (E8P today), calibration registry, gate-profile customization.

### G. Testing infra
- **Tiny-model smoke test**: a minutes-scale fixture MoE through the FULL pipeline (CI-able)
  — "the one command works" is itself a tested invariant.
- Golden-run regression: recipe → expected metrics within tolerance (catches quality drift).
- Keep the 170+ unit suite green; document `uv run python -m pytest`.

## Verification vehicle (user, 2026-07-09): Qwen3.6-35B-A3B before GLM-5.2

Verify the new setup/CLI on a much smaller model first: **Qwen3.6-35B-A3B** (already in the
HF cache). It is a real MoE (exercises the routed-expert pipeline), its bf16 teacher (~70GB)
fits on ONE 128GB Mac (so the FULL fresh-HF → RC pipeline including teacher-cache generation
is locally verifiable), substantial tooling exists (convert/qwen_moe.py, qwen_moe_adapter.py,
validate/qwen_vq.py, qwen-family build ops), and no artifacts exist yet — making
`keep get Qwen/Qwen3.6-35B-A3B` a genuine fresh-from-HF test. Acceptance for the P0 spine:
one high-level recipe + one command takes Qwen3.6 from the HF cache to a gated, chat-able RC.

## Phasing (serves GLM-5.2 first — the two priorities share a spine)

- **P0 (shared spine):** `keep get` (fresh-HF op) · model profiles (de-hardcode Air) ·
  high-level YAML + compiler · examples/ — *all four are prerequisites or near-prerequisites
  for the GLM-5.2 one-command run anyway.*
- **P1 (visibility):** `keep report` + terminal progress · README/quickstart/concepts docs ·
  `keep chat` · `keep doctor` (with the REAP-fits-in-RAM math).
- **P2 (flywheel):** `keep publish` w/ model card · tiny-model CI smoke · lm-eval hook ·
  monitor emitters · (WebUI: explicitly deferred — CLI+YAML is the API, LlamaBoard proves
  the UI can come later as a YAML-constructor.)

## Non-goals (explicit)
- No WebUI in v1. No multi-backend serving story (Metal/E8P/NAX is the target). No further
  Air quality work. Cleanliness-rigorous gating stays available but non-default (speed-first).
