# KEEP Troubleshooting

Use this guide before retrying a model-loading command. KEEP keeps evidence and
resume state deliberately: a failed preflight, an incomplete snapshot, and a
valid gate rejection need different responses.

## Start With An Offline Check

Install the repository environment from the repository root, then run the
offline doctor. It checks the Apple-silicon/macOS platform, Python, MLX/Metal,
`uv`, disk space, RAM, memory pressure, the current wired-memory policy, and
the local Hugging Face cache. With `--model`, it also checks that model's local
snapshot and a conservative resident-size estimate; it does not make network
requests or download a model.

```bash
uv sync --group dev
uv run keep doctor --model qwen36-35b-a3b
```

`keep doctor` exits `0` when it has no failed checks (warnings do not change
that exit code) and `1` when a required check fails. A missing model snapshot,
missing `uv`, an unavailable MLX/Metal device, or a non-Apple-silicon platform
is a reason to stop and fix the preflight rather than start a build.

If `uv sync` fails, first make sure `uv` is installed and on `PATH`; the
project's public environment command is `uv sync --group dev`. If doctor says
MLX or Metal is unavailable, use an Apple-silicon Mac with a working Metal
device. Running a build repeatedly will not make a missing device available.

## Hugging Face Authentication, Cache, And Shards

Start with the local-only readiness check:

```bash
uv run keep get qwen36-35b-a3b --check-only
```

`--check-only` never downloads. It verifies `config.json`,
`model.safetensors.index.json`, and every unique shard named by the index in
the expected Hugging Face snapshot. It exits `0` only when that snapshot is
ready; it exits `1` when any required file or shard is missing. A `ready: no`
result is therefore not an authentication problem by itself--it is a local
cache result.

When the report identifies missing content and you have Hugging Face access,
stage the snapshot with the same model name, without `--check-only`:

```bash
uv run keep get qwen36-35b-a3b
```

This may download many gigabytes. Authentication or network errors belong to
that download step; resolve them with the Hugging Face credentials and network
policy appropriate for the source model, then rerun `keep get`. Do not start a
build from a partial snapshot: the cache check gives a direct, cheap answer.

## Disk, RAM, Metal, And Wired Memory

KEEP needs space both for the source snapshot and for build outputs beneath the
recipe's build root. `keep doctor` warns below 20 GB free on both the repository
filesystem and the Hugging Face cache filesystem; that is a warning threshold,
not a promise that a particular model will fit. Use `keep get --check-only`
and `keep doctor --model <profile>` before allocating a large build.

For GLM-5.2-REAP, the authenticated production artifact contains
98,433,923,808 bytes of tensor payload (about 98.4 GB). Doctor evaluates that
profile with an additional 10% runtime headroom, so RAM below roughly 108.3 GB
is warned as infeasible. This is only an offline capacity screen, not proof of
a clean source conversion, full evaluation, or same-machine benchmark.

Keep the normal macOS memory policy unless an explicitly approved diagnostic
requires otherwise: leave `GLM_MLX_WIRED_LIMIT_GB` unset. The doctor reports
`iogpu.wired_limit_mb=0` as the system default and warns when a wired-limit
override is active. Raising wired limits reduces macOS reclaimable headroom;
it is not a general remedy for an out-of-memory or memory-pressure result.

## Interrupted Builds, Resume, And The Heavy-Job Lock

Each build root has an append-only `ledger.jsonl` and per-step output under
`steps/`. Re-running the same recipe reuses a matching completed step only
when its declared outputs remain intact. KEEP does not rerun it merely because
the command was invoked again.

A partial step directory is normally quarantined by rename rather than
deleted. Some registered materializers support validated partial resume and
keep their atomic checkpoints; their completion markers are invalidated before
they resume. To deliberately rerun one step and its descendants, use the
actual step id from the dry-run plan:

```bash
uv run keep build examples/qwen36-a3b.yaml --dry-run --no-hash
uv run keep build examples/qwen36-a3b.yaml --from STEP
```

The executor serializes model-loading and artifact-producing subprocesses with
the repository-root `.keep-heavy-job.lock`. It is an advisory `flock`, held
only while the heavy subprocess runs. A leftover lock *file* after a crash is
not, by itself, a stuck lock: the process-held lock is released when that
process exits. Do not delete the file just because it exists, and do not start
another heavy build to work around a currently running one.

## Gate Failures Are Evidence, Not Necessarily Crashes

`gate_failed` means the step produced its declared evidence, but that evidence
did not satisfy its structured gate (for example, a dirty memory row). KEEP
records it immutably in `ledger.jsonl`; reports label it `VALID BLOCKED`. It is
not interchangeable with a `failed` ledger event, which covers a rejected
subprocess return code, missing declared output, or another execution error.

For a recorded gate rejection, do not rerun the identical step key. Inspect
the ledger and evidence first. If the only intended change is gate thresholds,
re-evaluate recorded evidence without model work:

```bash
uv run keep build examples/qwen36-a3b.yaml --regate
```

`--regate` can promote a threshold-only recorded rejection when the stored
subprocess exit was `0` and the operation permits regating. It cannot turn a
nonzero blocked result or a composite release gate into completed work; those
need a real rerun after the underlying issue is resolved.

Exit codes are intentionally conservative:

- `0` means the requested CLI operation completed successfully; a normal build
  has no unresolved gate failure.
- `1` means a CLI/preflight/build failure, an incomplete `--check-only`
  snapshot, or a `keep build` whose gate did not pass.
- `2` is reserved by the GLM-5.2 family evidence operation for a structured,
  valid blocked verdict. The executor accepts that operation-level result only
  with its required gate, records `gate_failed` when appropriate, and the
  enclosing `keep build` still exits `1`.

Use `--continue-on-gate-fail` only when you intentionally need later steps to
collect a supervised evidence packet. It continues past gate rejections but
still exits nonzero if any gate failed.

## Memory-Clean Evidence

Benchmark and evaluation evidence records `pageouts_delta` and
`swapouts_delta`. A memory-clean measured interval has both values equal to
zero; a nonzero value is valid evidence of memory pressure, but it is not clean
performance evidence. A warm interval can be clean while cold load or warmup
is not, so read the scope recorded with the evidence rather than generalizing
one clean row to the entire run.

For a clean rerun, stop competing memory-heavy applications and other KEEP
heavy jobs, let the host return to a healthy memory-pressure state, keep the
system wired-memory default, then re-run only the necessary measurement path.
Do not retry when the prerequisite remains absent: Metal is unavailable, the
snapshot is incomplete, disk/RAM feasibility is still inadequate, a heavy job
is already active, or the evidence is a legitimate quality/speed gate miss.
Those cases require an environment change, source staging, a different recipe
or threshold decision, or a new experiment--not another identical command.

## More Help

For the public onboarding flow, see [Quickstart](QUICKSTART.md). For the
current GLM-5.2 pipeline boundary and release evidence, see
[GLM-5.2 Pipeline Readiness](GLM52_PIPELINE_READINESS.md).
