# GLM-4.5-Air VQ Release Candidate Pipeline

This is the GLM-4.5-Air release-candidate workflow for **KEEP** (the
KL-distilled Expert Encoding and Precision method) running on **RAMP** (the
Routed Accelerated MoE Pipeline). The legacy `mlx_vq` package remains available
as an import shim during the Air ladder, but new public code should use `keep.*`
for method/quality/artifact surfaces and `ramp.*` for runtime/kernels/benchmark
surfaces.

This repo now treats the accepted GLM-4.5-Air rank-4 low-rank residual artifact
as the balanced RC baseline:

```text
artifacts/glm-4.5-air-dynamic3p0-r26-lora-l45-gud-math8-r4-init0p05-s12-lr0p5-w2-m1-nll0p5-20260701
```

The artifact is linked from the protected r26 seed recorded in its manifest:

```text
artifacts/glm-4.5-air-dynamic3p0-target23-joint-gate-up-down-nextcycle-r25-r26-s48-lr1-w2-m1-nll0p5-20260701
```

## Declarative Build Recipes (`keep build`)

The candidate lineage is also encoded as a declarative recipe that one
command rebuilds end to end, content-addressed and resumable:

```bash
uv run keep validate recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml
uv run keep build recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml --dry-run
uv run keep build recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml
```

`scripts/glm45_air_rc.sh build <recipe> [args]` delegates to the same command
(and adds `--dry-run` under `GLM45_AIR_RC_DRY_RUN=1`). Key behaviors:

- Each step is keyed by `sha256(op, params, input identities)`; a completed
  step with a matching key is reused, never re-run — repeated builds do not
  hit the trainers' existing-output-dir refusal, because every changed step
  gets a fresh key-named directory under
  `artifacts/build/<recipe>/steps/<step>-<key8>/`. The runner never passes
  `--allow-existing` or `--overwrite`.
- Step classes enforce promotion integrity: `promotable` steps form artifact
  lineage, `verify` steps carry gate evidence, and `diagnostic` outputs (for
  example a position-scoped logit-bias bundle) can never feed promotable or
  verify steps — the recipe validator rejects such edges.
- Gates reuse the RC thresholds from `keep.quality.rc_gates` by named profile
  (`balanced_rc_split`, `community_wow`, `lane_s`, ...). Gate thresholds are
  not part of a step's execution key: `keep build <recipe> --regate`
  re-evaluates recorded evidence after a threshold change without re-running
  model work.
- Every promotable step appends a `build_steps[]` entry (op, params, input
  hashes, resolved argv, evidence paths, gate result) to its output
  `conversion-manifest.json`, so any artifact carries its full promotable
  ancestry plus a derived `recovery` tag list per `NAMING.md`.
- `--from <step>` force-invalidates a step and its descendants; partial
  outputs are quarantined by rename, never deleted; `gate_failed` evidence is
  kept as an immutable rejection record.
- `--continue-on-gate-fail` keeps executing later steps after a failed gate.
  Use it only for supervised evidence-collection runs where dirty memory is
  acceptable as a temporary diagnostic. The build still records the failed gate,
  writes promotion status from all available evidence, and exits nonzero if any
  gate failed.
- `uv run keep promote <recipe> --step <id> --as <base>__<repr><bpw>__<recovery>__<rev>__<date>`
  assigns the immutable published name (symlink under `artifacts/`) and
  stamps the mutable manifest `status: candidate` tag. Rev allocation stays a
  human decision.

`scripts/glm45_air_rc.sh reproduce` keeps working unchanged; the recipe above
is its declarative superset (same trainer flags, same verify chain).

### Next Recipe Step Driver

After a build has at least one completed promotable artifact plus report or
selection evidence, `keep plan-next` proposes the next recipe fragment without
editing the source recipe:

```bash
uv run keep plan-next recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml \
  --out recipes/glm45air__dmx2p0__sc-l45__r26__20260701.next.yaml
```

The proposal is rendered through the same op registry that `keep build` uses.
Its current priority order is:

1. A completed `top1-gap` tail-cleanup diagnostic with record type
   `glm45_air_tail_cleanup_target_selection` and decision
   `tail_cleanup_report_train_packet_ready`, which emits a selected-layer
   layer-41 rank-4 `train-low-rank` continuation with explicit report-cache
   row indices and auxiliary position indices. The target recipe's
   `top1_gap_tail_cleanup` step consumes repaired report evidence only, so it
   can be run as a cheap `--until top1_gap_tail_cleanup` slice before
   selection, holdout, or Lane S work.
2. A completed `top1-gap` diagnostic that recommends
   `bump-non-expert-precision`, for off-routed-path surfaces such as
   `embed_tokens`, `lm_head`, and `router_gates`. `plan-next` rejects broader
   surfaces such as `attention` and `shared_experts` on this automatic path,
   and the materializer itself only accepts those three cheap surfaces: broader
   resident-byte targets need a separate surface-preserving implementation, not
   the cheap precision-policy materializer.
3. A declared `sparse_residual_plan` or `sparse_plan` external input whose JSON
   contains `summary.route_source_sparse_residual_plan_rankings`, which emits a
   plan-backed `sparse-residual` proposal from the top ranked target/expert row.
   Do not point this at rejected replay plans; generate a fresh layer-probe plan
   for the current seed when using this path.
4. A route-domain top1 gap when the recipe declares a `route_disagreement`
   external input, which emits `train-router-kd`.
5. The previous rank-4 `train-low-rank` continuation selected from
   report/selection quality-focus rows. Holdout remains validation-only.

Additional bounded Track A levers are available as explicit recipe ops:

- `layer-probe-attribution` wraps
  `benchmarks/probe_glm45_air_layer_attribution.py` for source-oracle
  layer/projection attribution and sparse-residual plan generation. It writes
  `plan_json` evidence containing `summary.route_source_sparse_residual_plan_rankings`;
  use `--state-npz` inputs when replaying captured prompt states to avoid
  paying resident model capture again.
- `isolate-sparse-fp16` wraps
  `benchmarks/materialize_glm45_air_high_precision_projection.py` for sparse
  BF16/source routed-projection escapes such as `41:gate_proj`. It requires a
  pinned `model_id` and `revision`, links the seed artifact, rewrites only the
  requested routed projection shards with `dynamic_precision_tier=high`, and
  keeps the candidate promotable or diagnostic according to the recipe class.
  Use this for p999-tail isolation probes, not as a broad Lane S workaround:
  whole-projection BF16 escapes have already been rejected when over-applied.
- `sparse-residual` wraps
  `benchmarks/materialize_glm45_air_vq_sparse_residual.py` for bounded
  source-minus-VQ row residual sidecars. It can consume a layer-probe residual
  plan JSON (`--plan-json`, `--plan-target`, `--plan-expert`) or direct
  layer/projection/expert parameters, links the seed artifact, preserves
  continuous sidecars, and writes sparse residual rows as a normal candidate
  artifact. Use this for non-bias top1 mechanism probes when low-rank
  continuations and the current router checks are exhausted; do not treat
  non-expert precision as exhausted until the bug-signaled bump path has been
  rerun through the fixed loader and measured.

### Measured Track A Lever Verdicts

The target recipe now carries measured diagnostic branches for the free P1
levers that were previously unmeasured. These branches are evidence-producing
history, not promotion paths to rerun blindly:

- `bump_non_expert_precision_1` uses `bump-non-expert-precision` from
  `train_tail_cleanup_2` over only `embed_tokens`, `lm_head`, and
  `router_gates` in BF16. Its old resident-byte audit pass was tied to the same
  selected-surface loader/audit bug and is stale as runtime-residency evidence.
  The full report, selection, and holdout split evidence collapses mean top1
  agreement to `0.0` with PPL ratios far beyond normal quality movement. This is
  a BUG-SIGNAL result, not a lever verdict. The first root cause was
  loader-side: `non_expert_precision.surfaces` was used as a runtime binding
  filter, leaving unrequested non-expert tensors at fresh initialization. The
  loader now binds all source non-expert tensors while retaining requested
  surfaces as experiment/audit metadata, the resident-byte audit counts all
  runtime-loaded source non-expert tensors, and focused regression coverage
  proves materializer -> manifest -> loader -> one-row logit parity. The old
  full-split bump metrics and selected-surface byte reduction remain invalid
  until rerun through the fixed loader and audit. Fixed audit evidence at
  `artifacts/quality/glm45-air-bump-non-expert-fixed-loader-resident-byte-audit-20260704.json`
  shows no source non-expert byte reduction (`source_non_expert_delta_bytes=0`)
  and a small resident-byte increase from artifact bytes
  (`39793997312 -> 39827813888`), so the current bump is not a lower-residency
  mechanism as implemented. A corrected accepted-r4 `embed_tokens,lm_head`
  measurement at
  `artifacts/glm45-air-r4-bump-lm-head-embed-fixed-loader-20260704` is
  metadata-only in practice: resident bytes are exactly flat against accepted-r4
  (`39793997312 -> 39793997312`), repaired report and selection packets are
  both `125/128` clean and match accepted-r4 exactly (report top1/KLD/PPL/p999
  `0.7951634532160341`/`0.3685784389566362`/`1.0420245022610146`/
  `4.612546822034927`; selection `0.806739444608868`/
  `0.3754002484299577`/`1.048528893021234`/`4.612546822034927`), and the
  Lane S attempt had no valid timing rows because all measured repetitions
  paged out. Do not compose it, but also do not describe non-expert precision
  as rejected or exhausted from the old BUG-SIGNAL evidence.
- `train_router_kd_1` uses `train-router-kd` from `train_tail_cleanup_2` with
  the route-disagreement input, `scale=1`, `max_abs_delta=0.125`, and
  `num_experts=128`. Its repaired full-split evidence is memory-clean, but the
  best global top1 delta is slightly negative and route-domain top1 delta is
  `0.0`; it does not meet the `+0.01` global or `+0.05` route-top1 compose
  trigger. Treat this branch as rejected for promotion and composition.
- `sparse_residual_1` consumes the fresh layer41 route000 sparse residual plan
  generated from `train_tail_cleanup_2`. Focused report evidence is clean but
  top1-neutral, with only tiny likelihood/KLD movement and no row top1 flips.
  Do not broaden it to full splits unless a new sparse-aware mechanism or a
  fresh plan changes the hypothesis materially.

The durable full-split summary lives at:

```text
artifacts/quality/glm45-air-workstream1-free-levers-fullsplit-verdict-20260703.json
```

Peer 2 was not used for these local verdicts. Keep Peer 2 and transport/cache
work out of local quality documentation slices unless the operator explicitly
resumes Peer 2 use.

For bounded autonomous loops:

```bash
uv run keep build recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml --auto-extend 1
```

Accepted proposals are written into the generated recipe overlay and rejected
proposals are preserved under the overlay's `rejected` list. For zero-model-work
drift checks, use:

```bash
uv run keep build recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml --dry-run --no-hash
```

Lane S q2-control reuse policy: recipe builds should reuse a declared external
q2-control JSONL when the scenario, engine, repetition policy, and memory-clean
status match the accepted RC control. The current fixture points
`lane_s_candidate.inputs.control_evidence` at:

```text
artifacts/benchmarks/glm45-air-target23-nextcycle-r26-q2-control-prefill1k-quiet-20260701.jsonl
```

This avoids re-paying the q2 benchmark cost on every candidate recipe while
keeping the Lane S gate math unchanged. When a fresh same-window speed packet is
the actual goal, use the wrapper's paired `benchmark-lane-s` command; that path
still regenerates both `lane-s-q2-control.jsonl` and `lane-s-candidate.jsonl`
under the selected `--output-dir`.

Eval lower-residency policy: the target recipe's report/selection/holdout
`eval` steps pass `--mlx-cache-limit-gb 0 --mlx-clear-cache-before-load`.
This applies the best cheap evaluator-side memory lever seen so far without
changing the teacher-cache rows or enabling prefix truncation. Keep
`--truncate-input-to-selected-positions` as opt-in diagnostic tooling: it can
reduce long-row forward time, but prior full-report diagnostics saw worse
swapout behavior even when metrics matched.

Split evidence repair: the target recipe now separates each full split eval
from its gate. `eval_report`, `eval_selection`, and `eval_holdout` produce raw
128-row evidence, then `eval_report_repair`, `eval_selection_repair`, and
`eval_holdout_repair` rerun only rows that are dirty or missing and write the
merged evidence consumed by promotion gates. The repair tool prefers clean
prior rows, then clean retry rows, and leaves any still-dirty row in place, so
normal `balanced_rc_split` / `community_wow` gates remain the only authority.
This makes memory-noisy retries cheaper without turning dirty evidence into a
pass.

After a raw split exists, repeat attempts can preserve that expensive JSONL and
rerun only the repair step:

```bash
uv run keep build recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml \
  --repair-from eval_report \
  --continue-on-gate-fail \
  --accepted-rc-summary artifacts/rc/glm45-air-balanced-r4-20260701/rc-summary.json
```

`--repair-from eval_report` resolves to `eval_report_repair`, so the raw
`eval_report` step is not force-invalidated. The same pattern works for
`eval_selection` and `eval_holdout`.

Smoke recipe policy: the bounded
`recipes/glm45air__dmx2p0__sc-l45__r26__20260701-smoke16.yaml` recipe is for
runner/resume proof only, not RC promotion. Its split eval and Lane S gates set
explicit recipe-level smoke overrides with `max_rows: 16`, including
`allow_dirty_rows: true` and loose split-quality thresholds. Dirty memory and
16-row quality variance are tolerated only inside that smoke fixture. The
default gate profiles remain strict: full report/selection/holdout and RC
promotion still require clean evidence unless a recipe opts into an override by
name.

Base-build producer ops are available for extending the recipe back before the
protected seed:

- `stream-convert-vq` wraps `scripts/plan_stream_convert.py --convert-all` and
  emits a promotable VQ artifact directory. It can consume an explicit
  `source_dir`, or omit it and use the HF index snapshot directory as the
  source shard root. It also accepts optional `config_path` and `index_path`
  inputs plus conversion params such as `model_id`, `revision`, `group_size`,
  `code_bits`, `code_bits_policy`, and `expert_workers`.
- `export-teacher-cache-cleanroom` wraps the clean-room distributed cache
  runner through `benchmarks/export_glm45_air_cleanroom_cache.py`, so recipes
  render normal flags while the wrapper still delegates to
  `scripts/run_glm45_air_distributed_cleanroom_cache.sh`. Its promotable
  `teacher_cache` output resolves to `out/metadata.jsonl`, so seed-training
  steps can consume generated report/selection caches directly. It can consume
  a generated `rank_view_dir` input from `pipeline-source-views`; the wrapper
  derives deterministic rank roots from `rank-<n>` subdirectories and exports
  them as `GLM_RANK_VIEW_ROOTS_JSON` for the shell runner. Validation and
  local-consumption JSONLs are routed into the step evidence directory.
- `jaccl-hostfile` writes the two-rank MLX/JACCL hostfile used by the current
  direct Thunderbolt topology. It records the local direct IP, local/peer RDMA
  devices, and peer control SSH as a content-addressed recipe file output
  (`hostfile=out/hostfile.json`) instead of relying on an operator to hand-write
  `/tmp/glm-jaccl-hostfile-en1.json`.
- Seed-training cycles continue to use `train-low-rank`; generated caches from
  `export-teacher-cache-cleanroom` derive their cache roots from the
  `metadata.jsonl` parent directory just like external `teacher_cache` inputs.
- `scale-block-local-sidecars` is the cheap bridge for legacy alpha-scaled
  block-local artifacts: it links a seed artifact's routed shards, scales the
  declared continuous sidecars from a source sidecar artifact, and writes the
  normal `continuous_parameters` manifest. This avoids rerunning model
  training just to replay the accepted l18/l41 scalar sidecar bridge.
- `merge-block-local-sidecars` is the cheap bridge for older hand-assembled
  source sidecar artifacts: it links a seed artifact's routed shards, copies
  the declared continuous sidecars from ordered source artifacts, and writes a
  normal `merged_block_local_sidecars` manifest for downstream scaling.
- `fit-block-local-sidecar` wraps the source-projection residual fitter for
  legacy block-local source artifacts. It can consume an explicit `source_dir`,
  or omit it and resolve the full pinned HF snapshot from `model_id` and
  `revision` before reading source tensors.
- Legacy seed-training chains can be backfilled from existing
  `conversion-manifest.json` files with `keep recipe-lineage`. The generated
  recipe reconstructs every supported scaled-sidecar and continuous-training
  step and stops at the first non-continuous root artifact instead of guessing
  missing ancestry. The
  current accepted-r4 lineage recipe is:

  ```bash
  uv run keep recipe-lineage \
    artifacts/glm-4.5-air-dynamic3p0-r26-lora-l45-gud-math8-r4-init0p05-s12-lr0p5-w2-m1-nll0p5-20260701 \
    --name glm45air__dmx2p0__sc-l45__r26__legacy_lineage_20260701 \
    --base-revision a24ceef6ce4f3536971efe9b778bdaa1bab18daa \
    --include-cache-preflight \
    --out recipes/glm45air__dmx2p0__sc-l45__r26__legacy_lineage_20260701.yaml
  ```

  It currently emits 36 promotable steps: `stream-convert-vq` for the pinned
  8-bit base VQ packet, `collect-imatrix` from that generated base artifact,
  `stream-convert-vq` for the pinned 16-bit high-bit artifact, then
  `materialize-sweep` for the 3.0 bpw dynamic-imatrix candidate, four
  `fit-block-local-sidecar` steps plus nested/final l18/l41 merges to reproduce
  the older source sidecars, `scale-block-local-sidecars`, and 25
  `train-low-rank` steps. The base artifact's historical manifest has
  `existing` VQ groups but no conversion-run JSON, so the `--base-revision`
  argument is required to make that old complete Air root replayable. The
  lineage builder then reads safetensor `quantization_config` metadata from the
  135 base shards, infers code bits/group-size policy (`512` preferred,
  `352` effective for down projections), and emits the base convert command
  without carrying `artifacts/glm-4.5-air-vq` as an external input. The remaining
  explicit roots are teacher-cache inputs.

After a supervised build writes `promotion-result.json`, compare its gate
results with the accepted RC packet:

```bash
uv run keep rc-diff \
  artifacts/build/glm45air__dmx2p0__sc-l45__r26__20260701/promotion-result.json \
  --accepted-rc-summary artifacts/rc/glm45-air-balanced-r4-20260701/rc-summary.json
```

The diff normalizes recipe gate names back to RC summary check names such as
`checks.split_hard.report.clean_128` and `checks.lane_s.ratio_le_1p15`.
It exits nonzero on any mismatch.

For the supervised P0 loop, the build can write that diff automatically:

```bash
uv run keep build recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml \
  --from eval_report \
  --continue-on-gate-fail \
  --accepted-rc-summary artifacts/rc/glm45-air-balanced-r4-20260701/rc-summary.json
```

The default diff path is
`artifacts/build/glm45air__dmx2p0__sc-l45__r26__20260701/rc-diff.json`.
If that first run produces raw split evidence but misses cleanliness, use the
`--repair-from <split-step>` form above for row-scoped repeat attempts instead
of force-rerunning the whole 128-row split.

## One-Command RC Summary

Run the environment preflight before summary or heavy reruns:

```bash
scripts/glm45_air_rc.sh env-preflight
```

This checks MLX, mlx-lm, Metal kernel availability, required Hadamard transform
sizes, and safetensors integer roundtrips without loading GLM-4.5-Air.

Run the broader non-mutating RC preflight next:

```bash
scripts/glm45_air_rc.sh preflight
```

This checks that the accepted artifact, protected seed artifact, report,
selection, holdout, Lane S, q2-control, teacher-cache metadata, teacher-cache
roots, source model availability, and public workflow entrypoints are present.
The source model check uses the local Hugging Face cache with `local_files_only`
by default and verifies config/index metadata plus runtime safetensors shard
presence without loading tensor payloads. It ignores source shards outside the
runtime layer range, such as auxiliary/MTP-only shards.

To point preflight at an explicit local snapshot, pass:

```bash
scripts/glm45_air_rc.sh preflight --source-dir /path/to/GLM-4.5-Air-snapshot
```

It does not load the model, write summary files, or mutate artifacts. Use
Markdown output for a human check:

```bash
scripts/glm45_air_rc.sh preflight-md
```

Before heavy Lane S benchmark reruns, run the host memory preflight:

```bash
scripts/glm45_air_rc.sh memory-preflight
```

This checks that macOS pageout/swapout counters can stay flat for the required
quiet window without loading the model. The public Lane S benchmark commands use
the same required quiet-window gate and exit before worker launch if the host is
already paging.

Run the metadata/evidence summary first:

```bash
scripts/glm45_air_rc.sh summary --overwrite
```

For the full cheap public packet in one command, including focus, frontier, and
next-plan JSON exports, run:

```bash
scripts/glm45_air_rc.sh packet --overwrite
```

This reads the accepted report, selection, holdout, Lane S, and q2-control
evidence; recomputes split summaries from JSONL row records; audits artifact
prefill compatibility; and writes:

```text
artifacts/rc/glm45-air-balanced-r4-20260701/rc-summary.json
artifacts/rc/glm45-air-balanced-r4-20260701/RC_SUMMARY.md
artifacts/rc/glm45-air-balanced-r4-20260701/quality-focus.json
artifacts/rc/glm45-air-balanced-r4-20260701/quality-frontier.json
artifacts/rc/glm45-air-balanced-r4-20260701/quality-plan.json
```

The default command is intentionally cheap: it does not load the model or mutate
any artifact. It fails if hard balanced RC gates fail. It reports the current
community-wow target misses separately so the balanced RC can remain shippable
while quality work continues.

To regenerate the full heavy evidence path from fresh model-loading outputs,
against the accepted artifact, run:

```bash
scripts/glm45_air_rc.sh verify --overwrite --output-dir artifacts/rc/glm45-air-balanced-r4-fresh
```

That command executes the artifact audit, report/selection/holdout evals, paired
Lane S q2/candidate benchmarks, and the same focus/frontier/plan packet exports.
It is the closest current one-command reproduction path. It is expected to take
model time and may exit nonzero if fresh evidence fails a hard gate. In
particular, fresh Lane S acceptance requires memory-clean rows, timing relative
spread `<= 0.2` on the current three-row rerun path, and candidate speed within
the balanced RC ratio cap.

To reproduce the artifact path from the protected seed before verification, run:

```bash
scripts/glm45_air_rc.sh reproduce --overwrite --output-dir artifacts/rc/glm45-air-public-reproduction-r4
```

That command first trains the public rank-4 layer45 gate/up/down low-rank
residual recipe into
`artifacts/glm45-air-public-reproduction-low-rank-residual-r4`, then runs the
same audit, eval, paired Lane S, and packet export steps against the reproduced
artifact. It is intentionally heavy. The training step refuses an existing
artifact output directory, so use a fresh reproduction artifact path or remove
only a prior reproduction output you intentionally own before rerunning it.

The summary includes both split-level metrics and per-domain rows derived from
the eval prompt IDs. The per-domain table is where route/math/code top1 targets
are checked against the balanced RC quality target of `>= 0.80`. The generated
Markdown also includes a `Quality Focus` table for the lowest-top1 rows, highest
mean-KLD rows, highest token-KLD tails in each split, and a `Quality Frontier`
table that records balanced-versus-quality promotion evidence.

For a compact machine-readable tail export, run:

```bash
scripts/glm45_air_rc.sh focus --overwrite
```

This writes only the domain summaries and quality-focus rows, which is the first
tail-inspection handoff for a new quality run.

For a compact machine-readable promotion/frontier export, run:

```bash
scripts/glm45_air_rc.sh frontier --overwrite
```

This records the promoted preset, balanced hard targets, community-wow targets,
and any bounded quality candidates without requiring callers to parse the full
`rc-summary.json`.

For the next bounded quality slice, also write the machine-readable plan:

```bash
scripts/glm45_air_rc.sh plan --overwrite
```

The quality plan is model-free. It reads the same RC evidence, selects
report/selection row indices for a rank-4 layer45 route/math/instruction
low-rank residual run, and records holdout rows only as validation focus. Do not
train on holdout rows. The generated training command preserves the accepted
recipe constraints: `--trainable low_rank_residual`, `--low-rank 4`,
`--loss-scope final_layer_selected`, and the gate/up/down
`--surrogate-projections` setting.

The generated `rc-summary.json` and `RC_SUMMARY.md` also include a public
workflow command packet. That packet is the shortest current path through:

- environment preflight
- RC preflight
- host memory preflight
- full reproduction from protected seed
- full heavy RC verification
- calibration/imatrix collection
- dynamic-imatrix materialization
- rank-4 low-rank residual sidecar training
- report, selection, and holdout eval regeneration
- Lane S benchmark regeneration
- artifact audit
- focus export and token attribution
- balanced-vs-quality frontier export
- next quality-slice planning

Each command in that packet has an explicit run status. Cheap smoke/help paths
were run during this goal. Model-loading eval, benchmark, materialization, and
training commands are marked as heavy reruns unless they were exercised by the
accepted evidence that the summary reads.

When the wrapper executes a rerun command, the summary printed and written by
that invocation is rebuilt from the fresh deterministic output under
`--output-dir`: `report128.jsonl`, `selection128.jsonl`, `holdout128.jsonl`,
`lane-s-q2-control.jsonl`, or `lane-s-candidate.jsonl` as applicable. Passing
`--overwrite` removes that rerun JSONL before the lower-level append-only script
runs, which keeps fresh reports from accumulating stale rows. For Lane S, prefer
the paired `benchmark-lane-s` command so q2 and candidate evidence are regenerated
and summarized together.

## Regenerate Evaluation Evidence

The same wrapper prints the exact command packet in its JSON and Markdown
outputs. To run a fresh split:

```bash
scripts/glm45_air_rc.sh eval-report --overwrite
scripts/glm45_air_rc.sh eval-selection --overwrite
scripts/glm45_air_rc.sh eval-holdout --overwrite
```

Those commands call `benchmarks/eval_glm45_air_teacher_cache.py` with the clean
BF16/source teacher caches:

```text
artifacts/quality/glm45-air-teacher-cache-air-vq-ladder-report-v1-full-logits-route-trace-clean/metadata.jsonl
artifacts/quality/glm45-air-teacher-cache-air-vq-ladder-select-v1-full-logits-route-trace-clean/metadata.jsonl
artifacts/quality/glm45-air-teacher-cache-air-vq-ladder-holdout-v1-full-logits-clean-merged-20260701/metadata.jsonl
```

Each row must remain memory-clean. Dirty rows are not accepted as RC evidence.
After each command finishes, the generated `rc-summary.json` points that split
at the freshly written JSONL in the selected output directory.

## Inspect Tail Attribution

The summary and focus exports are the first stop for domain and tail metrics:

```bash
scripts/glm45_air_rc.sh focus --overwrite
```

For prompt/token attribution JSON that ranks prompt failures, token-KLD tails,
top-1 flips, and quality-floor violations for the accepted report, selection,
and holdout rows, run:

```bash
scripts/glm45_air_rc.sh attribute --overwrite
```

This writes:

```text
artifacts/rc/glm45-air-balanced-r4-20260701/report-attribution.json
artifacts/rc/glm45-air-balanced-r4-20260701/selection-attribution.json
artifacts/rc/glm45-air-balanced-r4-20260701/holdout-attribution.json
```

Use the split-specific commands when comparing a candidate JSONL against one
split:

```bash
scripts/glm45_air_rc.sh attribute-report --overwrite --candidate-jsonl candidate=artifacts/quality/<candidate-report>.jsonl
scripts/glm45_air_rc.sh attribute-selection --overwrite --candidate-jsonl candidate=artifacts/quality/<candidate-selection>.jsonl
scripts/glm45_air_rc.sh attribute-holdout --overwrite --candidate-jsonl candidate=artifacts/quality/<candidate-holdout>.jsonl
```

The `--output-dir` flag is consumed by the wrapper to choose the report location;
other flags are forwarded to `benchmarks/analyze_glm45_air_teacher_cache_attribution.py`.

## Regenerate Lane S

Lane S compares the RC artifact against the q2 routed control under the same
quiet-window methodology:

```bash
scripts/glm45_air_rc.sh benchmark-lane-s --overwrite
```

This paired command executes the q2 control first and the candidate second, then
rebuilds `rc-summary.json` from both freshly written files in the selected
`--output-dir`. Use the single-sided commands only for diagnosis:

```bash
scripts/glm45_air_rc.sh benchmark-q2 --overwrite
scripts/glm45_air_rc.sh benchmark-candidate --overwrite
```

The public wrapper uses one fresh-process warmup plus three measured
fresh-process repetitions, and it requires measured timing relative spread
`<= 0.2` for publication-grade reruns. Older accepted two-row Lane S evidence is
preserved as historical accepted evidence; the spread cap is enforced for the
current three-row rerun path. Warmups are not appended to the evidence JSONL, so
the accepted rows remain measured-only. The generated benchmark commands require
a host quiet-window preflight before each worker. If the host
cannot show flat pageout/swapout counters, the command exits before loading the
model instead of producing misleading dirty evidence. They also attach parent
`vm_stat` before/after snapshots and deltas to each measured row, and the
generated RC JSON/Markdown summaries compact those diagnostics into selected VM
delta totals for the candidate and q2 row sets. They also report accepted clean
timing, attempted timing over all measured rows, timing-stability pass/fail, and
Lane S failure reasons. This makes pageout or timing-stability rejections
diagnosable and avoids accepting a speed ratio only because one q2 control
repetition was unusually slow.

When candidate rows stay dirty despite parent pre-worker and child post-load
quiet windows, enable `--memory-phase-trace`. The target recipe enables this for
Lane S candidate runs. Phase labels show whether pressure came from model load,
tokenization, cache construction, measured prefill, decode, or metrics
collection, and each label includes MLX active/peak/cache bytes plus process
RSS. If `after_prefill` carries the pageout delta while `after_prefill_quiet`,
`before_measured`, and `after_cache` stay flat, the failure is inside measured
prefill rather than the launch/load quiet window. If the same pattern survives
`--mlx-cache-limit-gb 0`, target active/peak residency instead of persistent
MLX cache.

`--prefill-chunk-size` is available for diagnostic probes that split measured
prefill into sequential KV-cache updates. Chunked probes can show whether peak
activation pressure falls with smaller chunks, but they are not a Lane S gate
relaxation: measured rows must still be clean, stable, and within the q2 ratio.
The first 128/256-token chunk probes lowered peak only modestly while becoming
too slow and still dirty, so do not enable chunking in the target recipe without
a materially better implementation.

Use the header-only resident-byte audit to choose the next lower-residency
target before adding more Lane S probes. The target recipe includes this as the
diagnostic step `resident_byte_audit`, so a recipe-lineage rerun can execute
just the audit with:

```bash
uv run keep build recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml \
  --from resident_byte_audit \
  --until resident_byte_audit
```

The underlying standalone command remains useful for ad hoc artifact checks:

```bash
uv run python benchmarks/audit_glm45_air_resident_bytes.py \
  --artifact-dir artifacts/build/glm45air__dmx2p0__sc-l45__r26__20260701/steps/train_l45_gud_math8-afbbf149/out \
  --output-json artifacts/benchmarks/glm45-air-target23-nextcycle-r26-resident-byte-audit-20260702.json \
  --append-jsonl artifacts/benchmarks/glm45-air-target23-nextcycle-r26-resident-byte-audit-20260702.jsonl
```

For lower-residency candidates, compare against the current audited baseline in
the same diagnostic record before paying for Lane S:

```bash
uv run python benchmarks/audit_glm45_air_resident_bytes.py \
  --artifact-dir <candidate-artifact> \
  --baseline-json artifacts/build/glm45air__dmx2p0__sc-l45__r26__20260701/steps/resident_byte_audit-f2faba20/evidence/evidence_json.json \
  --min-total-byte-reduction <bytes> \
  --max-counted-resident-total-bytes <bytes> \
  --output-json <candidate-resident-byte-audit.json> \
  --append-jsonl <candidate-resident-byte-audit.jsonl>
```

When `--baseline-json` is present the evidence includes baseline totals,
candidate deltas, reduction bytes/ratio, and `resident_byte_check`. If threshold
parameters are supplied and the candidate misses them, the command writes the
failed check to evidence and exits nonzero. The `resident-byte-audit` recipe op
can render the same baseline and threshold flags through `baseline_evidence`,
`min_total_byte_reduction`, and `max_counted_resident_total_bytes`.

The `pipeline-source-views` op brings the upstream MLX pipeline source-view
materializer into recipe lineage. It wraps
`benchmarks/materialize_glm45_air_pipeline_views.py`, can either consume an
explicit source model directory plus optional safetensors index or resolve the
pinned `model_id`/`revision` from the local HF snapshot, writes rank-local
views to the step output directory, and renders split controls such as
`pipeline_size`, `layer_split`, `rank0_stop_after_layer`, route-trace pruning
switches, and `rank_budget_gb`. Keep it diagnostic for source-view preflights,
or mark it promotable when the source-derived views feed
`export-teacher-cache-cleanroom` and onward seed-training lineage. It does not
feed promotion gates by itself.

Pair source-view materialization with the `pipeline-view-load-probe` diagnostic
op when the recipe needs evidence that the generated views lazily load and match
the upstream MLX pipeline parameter plan. The op wraps
`benchmarks/probe_glm45_air_pipeline_view_load.py`, consumes
`view_dir: step:<pipeline-source-views-step>/artifact` plus either the same
explicit source/index inputs or the same pinned model/revision params, and
writes `evidence_json.json` containing `all_ranks_match_plan`, rank byte
budgets, layer ranges, key mismatches, and the explicit `did_eval_weights=false`
marker. The legacy accepted-r4 lineage recipe now starts with
`legacy_pipeline_source_views` and `legacy_pipeline_view_probe` diagnostic steps
before the promotable stream-convert root, so a `keep build` lineage replay
records the source-view preflight without tainting artifact promotion. This is
still a lazy-bind/source-view proof, not BF16 forward or teacher-cache
generation evidence.

The same legacy-lineage recipe also carries diagnostic
`legacy_jaccl_hostfile`, `legacy_report_cache_preflight`, and
`legacy_selection_cache_preflight` steps. These render the generated hostfile
plus generated rank-local source views into `export-teacher-cache-cleanroom`
with `--preflight-only`, `required_wired_mb=0`, and no custom
`mlx_wired_limit_gb` by default. They are a build-graph boundary for the
two-rank JACCL all-sum preflight, not a generated teacher cache. A full cache
replacement still requires the live Thunderbolt/RDMA checks in `AGENTS.md` to
pass before running exporter work. The 2026-07-04 read-only topology audit
`artifacts/quality/glm45-air-rdma-topology-audit-20260704-peer-restored-readonly.json`
records `rdma_link_down`: both hosts still map Thunderbolt 1 to `en1`, but the
interfaces are inactive, the direct `/30` IPs and ARP entries are absent, RDMA
IPv4 GIDs are missing, and UC pingpong/JACCL preflight/distributed cache
generation are not currently green.

This reads safetensors headers instead of loading the model. The current target
artifact counts about `37.06 GiB` resident tensor storage: `25.40GB` in VQ
artifact projections and `14.39GB` in source non-expert tensors. The largest
non-expert bucket is attention (`10.03GB`), followed by shared experts
(`1.56GB`), embeddings (`1.24GB`), and `lm_head` (`1.24GB`). Treat layer-46 MTP
missing-source shard warnings as non-resident for this audit; the resident
non-expert total is complete. The next Lane S memory work should target
attention/non-expert residency, the VQ artifact body, or a true lower-peak
prefill implementation rather than more MLX cache or naive chunking levers.

For cache-pressure troubleshooting, the wrapper also exposes a diagnostic-only
candidate command:

```bash
scripts/glm45_air_rc.sh benchmark-candidate-cache1 --overwrite
scripts/glm45_air_rc.sh benchmark-candidate-cache-sweep --overwrite
```

The first command runs one measured candidate row with `--mlx-cache-limit-gb 1`
and `--mlx-clear-cache-before-run`. The sweep command runs one measured row each
at `1`, `2`, and `4` GiB cache limits. These are not acceptance gates; use them
to compare memory policy behavior before rerunning the full Lane S benchmark.
Cache diagnostics are also replaced under `--overwrite`, but they do not replace
the accepted Lane S candidate summary because they are diagnostic-only row sets.

The balanced hard gates are:

- candidate and q2 control rows memory-clean
- three valid measured repetitions with timing relative spread `<= 0.2`
- effective routed bpw `<= 2.1`
- candidate/q2 prefill-1K median ratio `<= 1.15`
- no dense routed experts
- no unbound VQ experts
- non-expert dtype verified

## Artifact Audit

Run only the artifact audit with:

```bash
scripts/glm45_air_rc.sh audit --overwrite
```

The accepted balanced RC audit should show 135 routed projections, no
high-precision routed projection payloads, no Metal fallback layers for the
auto NAX-fast path, and seven continuous sidecars.

## Train A Rank-4 Sidecar

The workflow packet includes the accepted low-rank residual recipe for layer 45
gate/up/down:

```bash
scripts/glm45_air_rc.sh train-low-rank
scripts/glm45_air_rc.sh reproduce --overwrite --output-dir artifacts/rc/glm45-air-public-reproduction-r4
```

Run status: not rerun in this docs slice because these commands load the
resident model and use the heavy teacher caches. `train-low-rank` runs the
recipe only; `reproduce` runs that recipe and then verifies the reproduced
artifact. Their command shapes are covered by dry-run tests.

## Collect And Materialize Calibration Data

For a new materialization pass, collect routed projection calibration first:

```bash
scripts/glm45_air_rc.sh collect-imatrix
```

This writes the public calibration manifest:

```text
artifacts/imatrix/glm45-air-public-calibration/imatrix-manifest.json
```

Then materialize a bounded dynamic-imatrix sweep:

```bash
scripts/glm45_air_rc.sh materialize-sweep
```

The materializer consumes that public calibration manifest by default. To use a
different calibration run, pass:

```bash
scripts/glm45_air_rc.sh materialize-sweep --imatrix-manifest artifacts/imatrix/<run>/imatrix-manifest.json
```

If the selected manifest is missing, the wrapper exits before model work with an
actionable error pointing back to `collect-imatrix`.

Run status: not run in this docs slice because both commands require resident
model/source artifact work. Their wrapper command shapes are covered by dry-run
tests, and lower-level script `--help` entrypoints were smoke-tested.

## Extension Path For Another Model Family

Use `docs/new-model-family.md` as the reusable checklist for a
non-GLM-4.5-Air RC. For a new model family, add the same public surfaces before
publishing an RC:

```bash
scripts/glm45_air_rc.sh new-model-template
```

1. Model adapter that binds non-expert tensors without materializing routed
   dense experts.
2. Source tensor mapping from HF names to routed projection groups.
3. Manifest metadata for codebook, group size, projection storage, sidecars,
   and seed lineage.
4. Artifact audit that checks projection counts, precision tiers, and fallback
   paths.
5. Teacher-cache exporter or documented external teacher-cache source.
6. Eval wrapper that emits row-level clean-memory JSONL.
7. Benchmark wrapper with q-control comparison and pageout/swapout gates.
8. Tests for adapter binding, artifact loading, audit invariants, and summary
   thresholding.

The template expands those bullets into model identity, source tensor mapping,
manifest schema, audit rows, eval JSONL shape, benchmark gates, training and
calibration commands, summary outputs, and publication evidence. Its placeholder
commands are intentionally marked as not run until the family-specific wrappers
exist.

## Current Limitations

The balanced rank-4 RC passes the hard reproducibility gates, but it does not
yet hit the community-wow targets of mean KLD `<= 0.30`, p999 KLD `<= 3.0`, or
global top1 `>= 0.85` across report, selection, and holdout. The generated
domain table also marks any domain top1 below `0.80`. Route top1 remains the
main quality nuance to target next, and the `Quality Focus` table gives the
prompt rows and token positions to inspect before starting a new quality run.
The same content can be exported as `quality-focus.json` with the command above,
the balanced-vs-quality promotion state can be exported as
`quality-frontier.json`, and the first bounded rank-4 follow-up can be exported
as `quality-plan.json` without loading the model.

## Latest Quality Frontier Attempt

The first generated quality plan was executed into:

```text
artifacts/glm45-air-quality-r4-route-math-plan1
```

It is not promoted as a quality preset. It improves the full clean report,
selection, and repaired holdout splits versus the balanced RC, but its Lane S
speed evidence fails the current gate.

Quality evidence:

| Split | Clean rows | Mean KLD | Top1 | Mean PPL | Max PPL | p999 KLD | Compare |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| report | 128/128 | 0.314028 | 0.852442 | 0.964368 | 1.605935 | 3.581902 | RECOVER |
| selection | 128/128 | 0.319271 | 0.860780 | 0.977793 | 1.653569 | 3.581902 | RECOVER |
| holdout repaired | 128/128 | 0.304932 | 0.849860 | 0.954583 | 1.556347 | 3.581902 | RECOVER |

Speed evidence:

- Candidate Lane S attempt 1:
  `artifacts/benchmarks/glm45-air-quality-r4-route-math-plan1-lane-s-prefill1k-quiet.jsonl`
  was memory-clean but timing-unstable, median `3.276157707994571s`.
- Candidate Lane S retry:
  `artifacts/benchmarks/glm45-air-quality-r4-route-math-plan1-lane-s-prefill1k-quiet-retry2.jsonl`
  was memory-clean but timing-unstable, median `3.5266901875002077s`.
- Same-session q2 control:
  `artifacts/benchmarks/glm45-air-quality-r4-route-math-plan1-q2-control-prefill1k-quiet.jsonl`
  was clean and stable, median `0.9524510000046575s`.
- Same-session balanced-RC control:
  `artifacts/benchmarks/glm45-air-quality-r4-route-math-plan1-balanced-r4-control-prefill1k-quiet.jsonl`
  was not acceptable evidence because one row had pageouts.
- Diagnostic component profiles:
  `artifacts/benchmarks/glm45-air-quality-r4-route-math-plan1-component-profile-prefill1k.jsonl`
  and
  `artifacts/benchmarks/glm45-air-balanced-r4-component-profile-prefill1k-current.jsonl`
  both had `pageouts_delta=5`, so they are not benchmark-acceptance
  evidence. They do show that plan1 and the balanced RC have essentially the
  same routed projection component totals: plan1 gate/up/down
  `1.3513783760135993s`/`1.3488176649261732s`/`1.3913904200016987s`
  versus balanced
  `1.3617274550779257s`/`1.3393394149898086s`/`1.3923174579977058s`.
  Layer45 is likewise nearly identical.
- Sequential quiet-window rerun:
  `artifacts/benchmarks/glm45-air-quality-r4-route-math-plan1-lane-s-prefill1k-quiet-rerun3.jsonl`
  was clean and stable `2/2`, median `1.6227769590041135s`, with
  pageouts/swapouts `0/0`. Same-window q2 control
  `artifacts/benchmarks/glm45-air-quality-r4-route-math-plan1-q2-control-prefill1k-quiet-rerun3.jsonl`
  was also clean and stable `2/2`, median `0.9444605414901162s`, so
  plan1 remained `1.7182051422114344x` q2 and still failed Lane S.
  Same-window balanced-RC control
  `artifacts/benchmarks/glm45-air-balanced-r4-lane-s-prefill1k-quiet-rerun3.jsonl`
  was not acceptable evidence because row 1 had `pageouts_delta=72`.

Decision: keep `balanced` as the promoted RC. Treat `quality-r4-route-math-plan1`
as a useful RECOVER artifact for the next runtime/quality tradeoff slice, not as
a published quality preset. Do not attribute the current speed rejection to
value-dependent low-rank sidecar overhead; the clean rerun shows plan1 is stable
but too slow relative to q2, while the balanced-control rerun is still too noisy
to revise the promoted RC's accepted Lane S evidence.

## Commands Verified In This Slice

These cheap command surfaces were run while updating this workflow:

```bash
scripts/glm45_air_rc.sh help
scripts/glm45_air_rc.sh env-preflight
scripts/glm45_air_rc.sh preflight
scripts/glm45_air_rc.sh preflight-md
scripts/glm45_air_rc.sh memory-preflight
GLM45_AIR_RC_DRY_RUN=1 scripts/glm45_air_rc.sh reproduce --overwrite --output-dir artifacts/rc/glm45-air-public-reproduction-r4
GLM45_AIR_RC_DRY_RUN=1 scripts/glm45_air_rc.sh verify --overwrite --output-dir artifacts/rc/glm45-air-balanced-r4-fresh
GLM45_AIR_RC_DRY_RUN=1 scripts/glm45_air_rc.sh attribute --overwrite --output-dir artifacts/rc/glm45-air-balanced-r4-20260701
GLM45_AIR_RC_DRY_RUN=1 scripts/glm45_air_rc.sh benchmark-lane-s --overwrite --output-dir artifacts/rc/glm45-air-balanced-r4-fresh
GLM45_AIR_RC_DRY_RUN=1 scripts/glm45_air_rc.sh benchmark-candidate-cache-sweep --overwrite --output-dir artifacts/rc/glm45-air-balanced-r4-fresh
GLM45_AIR_RC_DRY_RUN=1 scripts/glm45_air_rc.sh collect-imatrix
GLM45_AIR_RC_DRY_RUN=1 scripts/glm45_air_rc.sh materialize-sweep
GLM45_AIR_RC_DRY_RUN=1 scripts/glm45_air_rc.sh materialize-sweep --imatrix-manifest artifacts/imatrix/custom/imatrix-manifest.json
GLM45_AIR_RC_DRY_RUN=1 scripts/glm45_air_rc.sh train-low-rank
scripts/glm45_air_rc.sh new-model-template
uv run python benchmarks/run_glm45_air_rc_pipeline.py --help
uv run python benchmarks/collect_glm45_air_imatrix.py --help
uv run python benchmarks/finetune_glm45_air_vq_continuous.py --help
uv run python benchmarks/materialize_glm45_air_dynamic_imatrix_sweep.py --help
uv run python benchmarks/eval_glm45_air_teacher_cache.py --help
uv run python benchmarks/bench_glm45_air_quant_compare.py --help
uv run python benchmarks/analyze_glm45_air_teacher_cache_attribution.py --help
uv run keep recipe-lineage --help
```
