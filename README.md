# KEEP + RAMP

Run a frontier-scale mixture-of-experts model on one Apple silicon Mac.

KEEP compresses a model's routed experts to about 2 bits per weight with
importance-weighted lattice vector quantization, and leaves attention, shared
experts, routers and embeddings near source precision. RAMP is the MLX runtime
that serves the result without ever expanding an expert back to a dense
matrix.

The current result: **DeepSeek-V4-Flash** (304 B parameters) compressed to a
**75.25 GB** routed payload at **2.031 bits per weight**, agreeing with the
uncompressed model on **89.9 %** of next-token predictions over a full
30-session evaluation split. Measured on one M5 Max, with no cloud spend.
Evidence and caveats are in
[`docs/deepseek-v4-flash/handoff.md`](docs/deepseek-v4-flash/handoff.md).

## Quickstart

### 1. Install

You need an Apple silicon Mac on macOS 14 or newer, [`uv`](https://docs.astral.sh/uv/),
and the Xcode command-line tools.

```bash
uv sync --group dev
uv run keep doctor
```

`keep doctor` checks the environment, memory and model cache without loading
any model.

### 2. Build the tensor-unit extension (M5 family)

The fast 2-bit expert kernels use the GPU tensor units that M5 chips have.
Build the extension once:

```bash
uv run --group dev cmake -S native/vq_nax_ext -B native/vq_nax_ext/build -DPython_EXECUTABLE="$(pwd)/.venv/bin/python3"
uv run --group dev cmake --build native/vq_nax_ext/build
uv run python -c "from mlx_vq.kernels import nax; print(nax.is_available())"
```

If it prints `False` on an M5, delete `native/vq_nax_ext/build` and configure
again. A stale cmake cache resolves MLX against the wrong Python. On M4 and
older chips it is always `False`, and the runtime falls back to a much slower
scalar kernel.

### 3. Look at a model profile

```bash
uv run keep models
uv run keep models show deepseek-v4-flash-0731
uv run keep get deepseek-v4-flash-0731 --check-only
```

A profile pins the Hugging Face revision and records the shapes, codebook and
group size KEEP uses. `--check-only` reports whether the snapshot is on disk
without downloading anything.

### 4. Plan a build before running one

```bash
uv run keep compile examples/qwen36-a3b.yaml -o /tmp/compiled.yaml
uv run keep build examples/qwen36-a3b.yaml --dry-run --no-hash
```

`examples/` holds short recipes (model, quality, output, calibration,
recovery). `recipes/` holds the explicit step graphs they compile to. Builds
resume from an append-only ledger instead of restarting.

### 5. Run the tests

```bash
uv run --frozen python -m pytest -q -p no:cacheprovider --continue-on-collection-errors --ignore=tests/test_nax_native.py --ignore=tests/test_glm45_air_projection_kernels.py
```

Many tests and quality tools need model data and prompt sets that are not in
the repo, so a clean checkout has known failures. The list is in
[`artifacts/baselines/`](artifacts/baselines/). Compare against it before
blaming your change.

## Compress a model end to end

The DeepSeek-V4-Flash path, in the order it runs. Every step is resumable.

```bash
# 1. Reference answers from the uncompressed model, streamed one layer at a time.
uv run python benchmarks/produce_dsv4_teacher_cache.py run --mode calibration --split calibration --out-dir ~/keep-artifacts/dsv4-teacher-calibration
uv run python benchmarks/produce_dsv4_teacher_cache.py run --mode logits --split report --out-dir ~/keep-artifacts/dsv4-teacher-logits-eval

# 2. Measure the rate ladder on one layer before committing to all of them.
uv run python benchmarks/pilot_dsv4_vq_materialization.py sweep

# 3. Fit every expert, then check that one block binds and matches its fit.
uv run python benchmarks/materialize_dsv4_vq.py run --out-dir ~/keep-artifacts/dsv4-vq-e8p-g512 --calibration-dir ~/keep-artifacts/dsv4-teacher-calibration
uv run python benchmarks/materialize_dsv4_vq.py verify-roundtrip --out-dir ~/keep-artifacts/dsv4-vq-e8p-g512 --blocks layers.0

# 4. Score against the teacher, and always run the uncompressed control.
uv run python benchmarks/eval_dsv4_vq_teacher_cache.py run --split report
uv run python benchmarks/eval_dsv4_vq_teacher_cache.py control --split report
uv run python benchmarks/eval_dsv4_vq_teacher_cache.py summarize
```

The control matters. It scores the uncompressed model against the same teacher
cache, which proves the measurement pipeline is exact and turns a raw
perplexity ratio into a compression cost.

Heavy jobs take `.keep-heavy-job.lock` at the repo root so only one runs at a
time. Leave `GLM_MLX_WIRED_LIMIT_GB` unset.

## Models

| Model | Status | Docs |
| --- | --- | --- |
| DeepSeek-V4-Flash-0731 | Compressed and evaluated. Quality gate passed on the report split | [`docs/deepseek-v4-flash/`](docs/deepseek-v4-flash/) |
| GLM-5.3-Flash | Adapter matches the reference model. Teacher run not started | [`docs/glm-5.3-flash/`](docs/glm-5.3-flash/) |
| GLM-4.5-Air | Earlier release candidate. The pipeline still runs | [`docs/glm45-air/rc-pipeline.md`](docs/glm45-air/rc-pipeline.md) |
| Qwen3.6-35B-A3B | Small onboarding example for new families | [`models/qwen36-35b-a3b.yaml`](models/qwen36-35b-a3b.yaml) |
| GLM-5.2-REAP | Retired. Library code remains because the build system still uses it | |

To add a family, follow [`docs/new-model-family.md`](docs/new-model-family.md).

## Repo layout

| Path | What it holds |
| --- | --- |
| `src/keep/` | KEEP, the compression method: codebooks, quantizers, converters, quality checks |
| `src/ramp/` | RAMP, the runtime: model adapters, the family registry, kernels |
| `src/mlx_vq/` | The original package. `keep` and `ramp` re-export from it while code moves |
| `native/vq_nax_ext/` | The tensor-unit kernel extension |
| `models/` | One pinned profile per model family |
| `examples/`, `recipes/` | Short recipes and the step graphs they compile to |
| `benchmarks/` | Command-line tools for each pipeline stage, named `<verb>_<family>_<thing>.py` |
| `scripts/` | Setup, download, census and checking tools |
| `tests/` | Unit and parity tests, with small fixtures in `tests/fixtures/` |
| `artifacts/` | Small evidence files the docs cite. Everything else under it is gitignored |
| `docs/` | One folder per model family for decisions, plans and research, plus the guides below |

## Docs

- [`docs/troubleshooting.md`](docs/troubleshooting.md): setup, cache, memory and build recovery.
- [`docs/new-model-family.md`](docs/new-model-family.md): the checklist for onboarding a model.
- [`docs/naming.md`](docs/naming.md): how runs and published artifacts are named.
- [`docs/deepseek-v4-flash/handoff.md`](docs/deepseek-v4-flash/handoff.md): DeepSeek state, environment traps and next steps.
- [`docs/glm-5.3-flash/2026-10-06-plan.md`](docs/glm-5.3-flash/2026-10-06-plan.md): the GLM-5.3-Flash plan.

## Known limitations

- The DeepSeek full runnable size (82.5 to 85.1 GB) is accounting. Only the
  routed payload exists as bytes so far.
- Production dispatch does not yet pick the fastest 2-bit kernels, and decode
  speed has not been measured for DeepSeek.
- Timing on these machines varies by up to 1.9x between runs. Quote ratios
  against a control, not absolute times.
