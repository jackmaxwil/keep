# KEEP Quickstart

Trouble with setup, snapshots, memory, or a blocked run? See the [Troubleshooting guide](TROUBLESHOOTING.md).

This walkthrough is for a stranger with an Apple-silicon Mac who wants to see
what KEEP will do before launching expensive model work.

The fastest safe path is:

1. Install the environment.
2. Pick a model profile.
3. Check whether the Hugging Face snapshot is already staged.
4. Read the high-level recipe.
5. Dry-run the build plan.
6. Run the build only after the plan and local snapshot look right.

## Install

From the repo root:

```bash
uv sync --group dev
```

Check the CLI:

```bash
uv run keep --help
```

The `keep` CLI includes recipe commands (`validate`, `compile`, `build`) and
model commands (`models`, `get`).

## Pick A Model

List the model profiles that ship with the repo:

```bash
uv run keep models
```

Current profiles:

- `deepseek-v4-flash-0731`: DeepSeek-V4-Flash-0731 MoE family profile — FP4-in-I8 routed experts, FP8 e4m3 residents; measurement in progress.
- `glm45-air`: GLM-4.5-Air RC path.
- `glm52-reap-504b-v2`: pinned GLM-5.2-REAP source and guarded release path.
- `qwen36-35b-a3b`: Qwen3.6-35B-A3B diagnostic onboarding path.

Inspect a profile:

```bash
uv run keep models show qwen36-35b-a3b
```

The profile tells KEEP the Hugging Face model ID, pinned revision, architecture,
layer counts, routed-expert shape, default engine, recovery layer, and rough
BF16-equivalent unquantized-weight memory estimate when one is declared.

Profiles live in [`models/`](../models/). To add a model later, start by adding
a profile YAML there, then connect it to a converter/template.

## Stage Or Check The Model Snapshot

Before running a build, ask KEEP whether the local Hugging Face snapshot is
ready:

```bash
uv run keep get qwen36-35b-a3b --check-only
```

`--check-only` does not download missing files. It inspects the local HF cache
and reports:

- snapshot path
- `config.json` presence
- `model.safetensors.index.json` presence
- shard count
- total shard size
- whether the profile's rough BF16-equivalent unquantized-weight estimate fits
  single-host RAM
- final `ready: yes` or `ready: no`

If files are missing and you want KEEP to stage them, run the same command
without `--check-only`:

```bash
uv run keep get qwen36-35b-a3b
```

That can download many gigabytes.

## GLM-5.2-REAP Current Frontier

Inspect the pinned GLM-5.2-REAP profile without loading the model:

```bash
uv run keep models show glm52-reap-504b-v2
```

The profile's embedded `notes` field is frozen at an earlier checkpoint because
the profile bytes participate in the authenticated artifact identity. Its model
ID, revision, and architecture fields remain authoritative; use the current
readiness document linked below for live artifact and release status.

Check the local source snapshot without downloading missing content:

```bash
uv run keep get glm52-reap-504b-v2 --check-only
```

The current pinned snapshot resolves to revision
`6c9241aa05fb243a0edb7c804c213ec1cf5c920d`. The verified local source contains
63/63 shards and about 308.829 GB. Source readiness does not mean the release
pipeline is complete.

There is no BF16 REAP teacher. The source-relative reference is the
deterministically dequantized pinned FP4 source. The `teacher ~1008GB` line from
`keep get` is a rough BF16-equivalent unquantized-weight memory estimate from
the profile, not a cached teacher payload or proof of exact W4A4
activation/runtime parity.

The authenticated production checkpoint has all 225 routed groups and exactly
98,433,923,808 whole-main tensor-payload bytes. The routed component is
`1.03125` routed bits per weight (bpw); the composite is `1.593443277857996`
whole-model bpw. Complete production bind and one-token generation are proven
without dense routed experts, but only the warm steady-state interval is
memory-clean.

The public high-level file exists so the intended beginner surface is visible:

```bash
uv run keep compile examples/glm52-reap.yaml
```

That command intentionally exits `1`. The guard remains active because exactly
four release-evidence blockers remain:

- dequantized-source teacher cache;
- full-vocabulary source-relative eval;
- route/math diagnostics;
- same-machine pinned-FP4 benchmark.

Until those evidence lanes close, the family gate is not release-capable and
`uv run keep build examples/glm52-reap.yaml` is not a runnable release command.
KEEP does not yet expose working `doctor`, `report`, or `chat` release surfaces
for GLM52.

For the authenticated checkpoint and exact reproduction commands, read
[`docs/GLM52_PIPELINE_READINESS.md`](GLM52_PIPELINE_READINESS.md).

## Understand The Example YAML

Open [`examples/qwen36-a3b.yaml`](../examples/qwen36-a3b.yaml):

```yaml
model: qwen36-35b-a3b
quality: fast
output: runs/qwen36-fast
calibration: default
recovery: off
overrides: {}
```

The six high-level keys are:

| Key | Meaning |
| --- | --- |
| `model` | The model profile name from `uv run keep models`. |
| `quality` | One of `fast`, `balanced`, or `wow`. The compiler maps this to gate and benchmark policy. `fast` is the cheap onboarding setting. |
| `output` | The build output root. The compiled recipe's build root goes under this path. |
| `calibration` | Calibration prompt policy. `default` uses the profile/template default. |
| `recovery` | `auto` enables template recovery when a trainer exists; `off` disables it. Qwen v1 uses `off` because there is not a Qwen KD trainer yet. |
| `overrides` | Advanced escape hatch. Keys are low-level step ids; values merge into that step's params after compilation. |

The GLM example uses the same shape:
[`examples/glm45-air.yaml`](../examples/glm45-air.yaml). It sets
`model: glm45-air`, `output: runs/glm45-air-fast`, and `recovery: auto`.

## Compile The High-Level Recipe

Compile the beginner recipe into the low-level graph:

```bash
uv run keep compile examples/qwen36-a3b.yaml -o /tmp/keep-qwen36-compiled.yaml
```

The compiled file has:

- `schema_version`
- `name`
- `description`
- `build_root`
- `external_inputs`
- `steps`

This is the same graph shape used by files in [`recipes/`](../recipes/). The
high-level YAML is the community layer; low-level recipes are the advanced layer
for explicit step graphs. The product direction for that split is in
[`docs/COMMUNITY_PRODUCT_PLAN.md`](COMMUNITY_PRODUCT_PLAN.md).

## Dry-Run And Read The Plan

Print the plan without executing it:

```bash
uv run keep build examples/qwen36-a3b.yaml --dry-run --no-hash
```

Use `--no-hash` with dry-run when large external inputs may not exist or when
you only want command rendering. The CLI enforces that `--no-hash` is only valid
with `--dry-run`.

The plan starts with:

```text
recipe: qwen36-35b-a3b-fast
build root: runs/qwen36-fast/qwen36-35b-a3b-fast
```

Then each step shows:

- step id, class, op, key, and gate
- resolved inputs
- exact `argv` that would run

For Qwen, the rendered plan includes source audits, group materialization,
artifact audits, bind probes, tokenizer/logit probes, eval-row preparation,
latency probes, and final family gate checks.

## Run The Build

After `keep get --check-only` reports `ready: yes` and the dry-run plan looks
right:

```bash
uv run keep build examples/qwen36-a3b.yaml
```

This launches real model work. For Qwen, expect the build to read the local HF
snapshot and write conversion/eval/benchmark evidence under `runs/qwen36-fast/`.

## Where Outputs And Evidence Land

For `examples/qwen36-a3b.yaml`, the default build root is:

```text
runs/qwen36-fast/qwen36-35b-a3b-fast
```

Inside it:

```text
ledger.jsonl
steps/
  <step_id>-<step_key>/
    evidence/
      evidence.jsonl
      evidence_json.json
    out/
```

Some steps produce artifacts under `out/`; some produce only evidence. The
dry-run prints the exact output paths before anything runs.

## Resume Behavior

KEEP uses a content-addressed append-only ledger:

- A `completed` ledger entry with a matching step key and intact outputs is
  reused. The step is not re-run.
- `gate_failed` entries are immutable rejection records. The same key is not
  re-executed; use `--regate` to re-evaluate recorded evidence.
- Partial directories from interrupted or non-terminal runs are quarantined by
  rename, never deleted.
- `--from <step>` force-invalidates that step and its DAG descendants.

The source of truth is the build root's `ledger.jsonl`.

## Inspect Results

Start with the ledger:

```bash
tail -n 20 runs/qwen36-fast/qwen36-35b-a3b-fast/ledger.jsonl
```

Then inspect step evidence:

```bash
find runs/qwen36-fast/qwen36-35b-a3b-fast/steps -path '*/evidence/*' -maxdepth 4 -type f
```

Eval-like evidence is usually JSONL. For the Qwen family recipe, look for files
such as:

```text
runs/qwen36-fast/qwen36-35b-a3b-fast/steps/qwen_family_eval_rows-*/out/qwen_family_eval_rows.jsonl
runs/qwen36-fast/qwen36-35b-a3b-fast/steps/qwen_family_benchmark_rows-*/out/qwen_family_benchmark_rows.jsonl
```

Gate steps write compact JSON summaries under their `evidence/` directories.

## Go Deeper

Use `compile` whenever you want to see what a high-level recipe expands to:

```bash
uv run keep compile examples/glm45-air.yaml -o /tmp/keep-glm45-air-compiled.yaml
```

Use low-level recipes when you need direct control over step ids, external
inputs, gates, and recovery graph shape:

```bash
uv run keep build recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml --dry-run --no-hash
```

Use model profiles when adding a new family:

```text
models/glm45-air.yaml
models/glm52-reap-504b-v2.yaml
models/qwen36-35b-a3b.yaml
```

For GLM-4.5-Air release-candidate details, read
[`docs/GLM45_AIR_RC_PIPELINE.md`](GLM45_AIR_RC_PIPELINE.md).
For the current GLM-5.2-REAP release frontier, read
[`docs/GLM52_PIPELINE_READINESS.md`](GLM52_PIPELINE_READINESS.md).
