# KEEP + RAMP

Run a huge mixture-of-experts language model on one Apple silicon Mac.

**KEEP** shrinks a model's experts to about 2 bits per weight. **RAMP** runs
the shrunken model on the Mac's GPU with MLX.

What it has done so far: **DeepSeek-V4-Flash** (304 B parameters) squeezed to
a **75.25 GB** expert payload at **2.031 bits per weight**. The compressed
model picks the same next token as the original **89.9 %** of the time,
measured over a full 30-session test set on one M5 Max. Proof and caveats are
in [`docs/deepseek-v4-flash/handoff.md`](docs/deepseek-v4-flash/handoff.md).

---

## Part 1. Check that it works on your Mac (5 minutes, downloads no models)

Do these in order. Each step says what you should see. If you see something
else, stop and look at [If something goes wrong](#if-something-goes-wrong).

### Step 1. Install the three tools you need

You need macOS 14 or newer on an Apple silicon Mac (M1 or later). Open
Terminal and run these one at a time. Skip any you already have.

```bash
xcode-select --install
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close Terminal and open it again so it finds `uv`. Then check:

```bash
uv --version
```

You should see a version number such as `uv 0.12.1`.

### Step 2. Get the code

```bash
git clone https://github.com/jackmaxwil/keep.git
```

```bash
cd keep
```

Every command from here on is run inside this `keep` folder.

### Step 3. Install the Python packages

```bash
uv sync --group dev
```

This takes a minute or two the first time. It ends with a list of installed
packages, or with almost nothing if they were already installed. Either is
fine.

### Step 4. Check your machine

```bash
uv run keep doctor
```

You should see lines that start with `PASS`. A line that starts with `WARN`
about disk or memory is a heads-up, not a failure. Nothing is downloaded and no
model is loaded.

### Step 5. List the models KEEP knows about

```bash
uv run keep models
```

You should see five lines, one per model, starting with
`deepseek-v4-flash-0731`.

### Step 6. Ask whether a model is already on disk

```bash
uv run keep get deepseek-v4-flash-0731 --check-only
```

On a fresh Mac this prints `ready: no` and finishes with an error code. **That
is expected.** It only checks, it never downloads. A line saying the teacher
"needs off-box/distributed" is also expected: the uncompressed model is bigger
than any Mac's memory, which is why KEEP streams it one layer at a time.

### Step 7. Plan a build without running it

```bash
uv run keep build examples/qwen36-a3b.yaml --dry-run --no-hash
```

You should see a list of steps (`step source_audit ...`,
`step source_payload_audit ...` and so on). Nothing runs. This shows what a
real build would do.

### Step 8. Run the tests (optional, about 5 minutes)

```bash
uv run --frozen python -m pytest -q -p no:cacheprovider --continue-on-collection-errors --ignore=tests/test_nax_native.py --ignore=tests/test_glm45_air_projection_kernels.py
```

**Some tests fail on every machine**, because they need model files and
prompt sets that are not in this repo. That is normal. Your result is fine if
every failing test is listed in
[`artifacts/baselines/test-baseline-20261007.txt`](artifacts/baselines/test-baseline-20261007.txt).

If you got this far, your setup works.

---

## Part 2. Turn on the fast kernels (M5 chips only)

**Skip this part unless your Mac has an M5, M5 Pro or M5 Max.** On an M4 Max
the extension builds, but 36 of its 94 kernel tests return wrong answers. So
RAMP refuses to use it on anything older than M5 and falls back to slower
kernels that are correct.

On an M5:

```bash
uv run --group dev cmake -S native/vq_nax_ext -B native/vq_nax_ext/build -DPython_EXECUTABLE="$(pwd)/.venv/bin/python3"
```

```bash
uv run --group dev cmake --build native/vq_nax_ext/build
```

```bash
uv run python -c "from ramp.kernels import nax; print(nax.is_available())"
```

On an M5 the last command prints `True`. On M1 to M4 it prints `False`, which
is correct. If an M5 prints `False`, delete the folder
`native/vq_nax_ext/build` and run the three commands again.

---

## Part 3. Compress DeepSeek-V4-Flash yourself (days, very large downloads)

Only start this if all of these are true:

- You have an M5 Max (or M5 Ultra) with 128 GB of memory.
- You have **at least 240 GB of free disk**: 163 GB for the original model and
  75 GB for the compressed experts, plus room for the teacher caches.
- You have a calibration corpus at
  `~/models/teich/dsv4-coding-agent-v1-20260811.json`. **This file is not part
  of the repo.** The campaign used private coding-agent sessions. Its format is
  pinned by [`recipes/dsv4_teich_split_manifest_v1_20260811.json`](recipes/dsv4_teich_split_manifest_v1_20260811.json).
  Without it, steps 2, 3 and 6 cannot run.

Every step below resumes where it stopped if you run it again. Long steps hold
a lock file, `.keep-heavy-job.lock`, so only one runs at a time. **Never run
two of them at once.** Two model-sized jobs can run a 128 GB Mac out of memory.

**1. Download the original model** (163 GB, saved to `~/models/DeepSeek-V4-Flash-0731`):

```bash
uv run python scripts/download_dsv4_flash_source.py
```

**2. Record how the original model behaves on calibration text.** This took
5.1 hours on an M5 Max.

```bash
uv run python benchmarks/produce_dsv4_teacher_cache.py run --mode calibration --split calibration --out-dir ~/keep-artifacts/dsv4-teacher-calibration
```

Check progress from a second Terminal window:

```bash
uv run python benchmarks/produce_dsv4_teacher_cache.py monitor --out-dir ~/keep-artifacts/dsv4-teacher-calibration
```

**3. Record the original model's answers on the test text.**

```bash
uv run python benchmarks/produce_dsv4_teacher_cache.py run --mode logits --split report --out-dir ~/keep-artifacts/dsv4-teacher-logits-eval
```

**4. Compress the experts.** The full sweep took 1.8 hours on an M5 Max.

```bash
uv run python benchmarks/materialize_dsv4_vq.py imatrix
```

```bash
uv run python benchmarks/materialize_dsv4_vq.py sweep
```

```bash
uv run python benchmarks/materialize_dsv4_vq.py manifest
```

The compressed experts land in `~/keep-artifacts/dsv4-vq-e8p-g512`.

**5. Prove the compressed experts load correctly.**

```bash
uv run python benchmarks/materialize_dsv4_vq.py verify-roundtrip --block layers.0
```

**6. Score the compressed model against the original.**

```bash
uv run python benchmarks/eval_dsv4_vq_teacher_cache.py run --split report
```

```bash
uv run python benchmarks/eval_dsv4_vq_teacher_cache.py control --split report
```

```bash
uv run python benchmarks/eval_dsv4_vq_teacher_cache.py summarize
```

The `control` run scores the *uncompressed* model the same way. It should come
out almost perfect. If it does not, the measurement is broken, not the
compression, so do not trust the `run` numbers until `control` passes.

---

## If something goes wrong

| What you see | What to do |
| --- | --- |
| `uv: command not found` | Close Terminal, open it again, and retry. If it persists, rerun the `curl` install line from Step 1. |
| `Failed to query Python interpreter ... Permission denied` | Run `uv python install --reinstall`, then retry. |
| `quality prompt set '...' is empty` | That tool needs GLM-4.5-Air prompt data that is not in the repo. It does not affect Part 1 or the DeepSeek steps. |
| `nax.is_available()` prints `False` on an M5 | Delete `native/vq_nax_ext/build` and redo Part 2. |
| A long job says another heavy job holds the lock | Another long job is running. Wait for it, or stop it, before starting a new one. |
| The Mac becomes slow, or memory pressure turns red | Stop the job with `Ctrl-C`. Make sure only one long job runs at a time. |

More fixes: [`docs/troubleshooting.md`](docs/troubleshooting.md).

---

## Reference

### Models

| Model | Status | Docs |
| --- | --- | --- |
| DeepSeek-V4-Flash-0731 | Compressed and evaluated. Quality gate passed | [`docs/deepseek-v4-flash/`](docs/deepseek-v4-flash/) |
| GLM-5.3-Flash | In progress. The adapter matches the reference model. Not runnable yet | [`docs/glm-5.3-flash/`](docs/glm-5.3-flash/) |
| GLM-4.5-Air | Earlier release candidate. The pipeline still runs | [`docs/glm45-air/rc-pipeline.md`](docs/glm45-air/rc-pipeline.md) |
| Qwen3.6-35B-A3B | Small example used for onboarding new models | [`models/qwen36-35b-a3b.yaml`](models/qwen36-35b-a3b.yaml) |
| GLM-5.2-REAP | Retired. Its library code remains because the build system uses it | |

To add a model, follow [`docs/new-model-family.md`](docs/new-model-family.md).

### Repo layout

| Path | What it holds |
| --- | --- |
| `src/keep/` | KEEP, the compression method: codebooks, quantizers, converters, quality checks, the `keep` command |
| `src/ramp/` | RAMP, the runtime: model adapters, the model registry, GPU kernels |
| `native/vq_nax_ext/` | The M5 tensor-unit kernel extension from Part 2 |
| `models/` | One settings file per supported model |
| `examples/`, `recipes/` | Short build recipes and the detailed step lists they expand into |
| `benchmarks/` | One command-line tool per pipeline stage, named `<verb>_<model>_<thing>.py` |
| `scripts/` | Setup, download and checking tools |
| `tests/` | Tests, with small fixtures in `tests/fixtures/` |
| `artifacts/` | Small evidence files the docs cite. Everything else there stays out of git |
| `docs/` | One folder per model for decisions, plans and research, plus the guides below |

### Docs

- [`docs/troubleshooting.md`](docs/troubleshooting.md): setup, cache, memory and build problems.
- [`docs/new-model-family.md`](docs/new-model-family.md): checklist for adding a model.
- [`docs/naming.md`](docs/naming.md): how runs and published models are named.
- [`docs/deepseek-v4-flash/handoff.md`](docs/deepseek-v4-flash/handoff.md): DeepSeek state and next steps.
- [`docs/glm-5.3-flash/2026-10-06-plan.md`](docs/glm-5.3-flash/2026-10-06-plan.md): the GLM-5.3-Flash plan.

### Known limitations

- The DeepSeek full runnable size (82.5 to 85.1 GB) is an estimate. Only the
  compressed experts exist as files so far.
- RAMP does not yet pick the fastest kernels automatically, and DeepSeek
  decode speed has not been measured.
- Timings on these Macs vary by up to 1.9x between runs. Compare against a
  control run, never against an absolute time.
