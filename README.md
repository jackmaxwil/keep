# KEEP + RAMP

Run a frontier-scale routed-expert MoE model resident on one Apple silicon Mac,
by compressing the experts hard and keeping everything else honest.

Primary target: **`deepseek-ai/DeepSeek-V4-Flash-0731`** (304 B parameters,
pinned at revision `7872f01b`). The compressed artifact is **75.25 GB** of
routed experts at **2.031 bits per weight**, and it agrees with the source
model on **89.9 %** of next-token argmaxes across a full 30-session evaluation
split. Every number on this page was measured on one M5 Max; cloud spend on the
campaign is **$0.00**.

- Decision record: [`keep-deepseek-v4-flash-pivot-20260811.md`](keep-deepseek-v4-flash-pivot-20260811.md)
- Live state and next steps: [`docs/HANDOFF.md`](docs/HANDOFF.md)
- Earlier GLM-4.5-Air release candidate (superseded as the headline, pipeline
  still supported): [`docs/GLM45_AIR_RC_PIPELINE.md`](docs/GLM45_AIR_RC_PIPELINE.md)

---

## 1. What is KEEP

**KEEP is the compression method.** It turns the routed experts of a
mixture-of-experts model into low-bit vector-quantized artifacts, and it treats
evidence as part of the deliverable.

In a sparse MoE model the routed experts dominate size — for DeepSeek-V4-Flash,
277 B of the 304 B parameters. KEEP compresses those and leaves attention,
shared experts, routers, and embeddings in source precision or a mild affine
quantization, because damaging them costs far more quality per byte saved.

The method, in order:

1. **Measure importance.** Forward real calibration traffic through the source
   model and record per-column activation statistics for every routed
   projection. Columns that carry more activation energy get protected.
2. **Choose the rate.** Codebook family and group size are re-derived per model
   family, never inherited. On this family the codebook choice dominates: 8-bit
   E8 lattice codes measured a 0.56 block cosine (unusable), while 16-bit E8P
   codes reached 0.954 at the same group size.
3. **Fit the codes.** Importance-weighted lattice VQ per expert projection,
   with a named uniform-importance fallback for any expert the calibration
   corpus never routed to (exactly one existed model-wide).
4. **Recover, optionally.** Train a low-rank or output-bias sidecar against
   teacher logits to repair the hardest rows, without mutating the protected
   seed artifact.
5. **Prove it.** Compare against a captured teacher cache on non-overlapping
   report / selection / holdout splits, with a paired uncompressed control so
   the measurement pipeline itself is validated.

Artifacts carry a manifest with source identity and revision, per-layer fit
metrics, codebook hashes, and every file's SHA-256. Builds resume from an
append-only ledger instead of restarting.

**Where it lives:** `src/keep/` — `vq/`, `quant/`, `io/`, `quality/`,
`convert/`, `validate/`.

## 2. What is RAMP

**RAMP is the runtime.** KEEP's artifacts are useless unless something can
multiply by them fast, and a compressed expert is not a matrix — it is codes
plus scales plus a codebook that has to be decoded inside the matmul.

RAMP provides:

- **Model adapters** that bind quantized experts into a real model without ever
  materializing a dense routed tensor. The DeepSeek-V4-Flash adapter carries
  DSpark sparse attention, hyper-connections, the clamped SwiGLU, and the
  three-block MTP drafter.
- **Routing operations** that gather only the experts a token actually selected,
  with a dispatch layer that picks a kernel from shape, route count, code width,
  and activation dtype.
- **Metal and NAX kernels** — a hand-written Metal family plus a native Apple
  tensor-unit extension (`native/vq_nax_ext/`) carrying ~50 E8P variants.
- **Benchmark surfaces** with quiet-window discipline, fresh-process rows, and
  pageouts treated as an acceptance gate rather than a footnote.

**Where it lives:** `src/ramp/` — `models/`, `nn/`, `ops/`, `kernels/`,
`benchmark/`.

> `mlx_vq` is the legacy package. It still works; new code goes in `keep` and
> `ramp`, which re-export through it during the cutover.

### The speed thesis

The reason RAMP exists as a peer to KEEP rather than a footnote:

```
tok/s  ≈  (bytes per token)⁻¹  ×  (tokens accepted per verify pass)
```

Compression shrinks the left factor. The model's built-in MTP drafter grows the
right one. Both together is the goal, and nobody has shipped it on Apple
silicon. **This is currently unproven** — see §5.

## 3. Using KEEP and RAMP

```bash
uv sync --group dev

# Build the native tensor-unit extension. Without this, every E8P fast path is
# silently unreachable and the runtime falls back to a scalar kernel ~8x slower.
uv run --group dev cmake -S native/vq_nax_ext -B native/vq_nax_ext/build \
  -DPython_EXECUTABLE="$(pwd)/.venv/bin/python3"
uv run --group dev cmake --build native/vq_nax_ext/build
uv run --group dev python -c "from mlx_vq.kernels import nax; print(nax.is_available())"
```

If that prints `False`, wipe `native/vq_nax_ext/build` and configure again — a
stale cmake cache will resolve MLX against the wrong interpreter.

### Orientation

| Goal | Command |
| --- | --- |
| List model profiles | `uv run keep models` |
| Inspect a profile | `uv run keep models show deepseek-v4-flash-0731` |
| Check a local snapshot without loading it | `uv run keep get deepseek-v4-flash-0731 --check-only` |
| Compile a high-level recipe | `uv run keep compile examples/qwen36-a3b.yaml -o /tmp/compiled.yaml` |
| Dry-run without model work | `uv run keep build examples/qwen36-a3b.yaml --dry-run --no-hash` |

Recipes come in two layers: `examples/*.yaml` is the small public surface
(`model` / `quality` / `output` / `calibration` / `recovery` / `overrides`);
`recipes/*.yaml` is the explicit step graph with ids, gates, and external
inputs. Start with the former.

### Training and materialization

Reference answer keys come first — compression is calibrated and judged against
the uncompressed model's own outputs. The teacher runner streams the full model
layer-sequentially, so a 304 B model runs on a 128 GB machine:

```bash
# Activation statistics for importance-weighted fitting.
uv run python benchmarks/produce_dsv4_teacher_cache.py run \
  --mode calibration --split calibration --out-dir ~/keep-artifacts/dsv4-teacher-calibration

# Top-2048 teacher logits for the quality gate.
uv run python benchmarks/produce_dsv4_teacher_cache.py run \
  --mode logits --split report --out-dir ~/keep-artifacts/dsv4-teacher-logits-eval

# Progress on a live run.
uv run python benchmarks/produce_dsv4_teacher_cache.py monitor
```

Both modes are resumable per session with atomic writes, and they stamp the
prefill chunk size into every artifact — chunked prefill is **not**
bit-identical across chunk sizes, so a cache is only comparable to a run that
used the same chunk.

Then fit the artifact. Measure one layer before committing to all 46:

```bash
# Rate ladder and per-layer wall-clock, on a stratified expert sample.
uv run python benchmarks/pilot_dsv4_vq_materialization.py sweep
uv run python benchmarks/pilot_dsv4_vq_materialization.py layer-fit
uv run python benchmarks/pilot_dsv4_vq_materialization.py schedule

# Full materialization: resumable per block, atomic, manifest + audit.
uv run python benchmarks/materialize_dsv4_vq.py run \
  --out-dir ~/keep-artifacts/dsv4-vq-e8p-g512 \
  --calibration-dir ~/keep-artifacts/dsv4-teacher-calibration
uv run python benchmarks/materialize_dsv4_vq.py monitor
uv run python benchmarks/materialize_dsv4_vq.py manifest
```

### Recovery

Recovery trains a small trainable surface against teacher logits to buy back
quality the codebook cannot reach at a given bit rate. For the GLM-4.5-Air path
this is a rank-4 low-rank residual with a route-local surrogate loss:

```bash
scripts/glm45_air_rc.sh train-low-rank
uv run python benchmarks/finetune_glm45_air_vq_continuous.py --help
```

For DeepSeek-V4-Flash the intended surface is the **MTP drafter head**, trained
against captured drafter logits and hidden states, because acceptance rate
converts directly into tokens per second. The teacher runner supports
`--mode mtp-targets`; that split has not been generated yet.

Recovery never mutates a protected seed artifact — it writes a sidecar, and the
audit proves the seed is unchanged.

### Inference

```bash
uv run python benchmarks/materialize_dsv4_vq.py verify-roundtrip \
  --out-dir ~/keep-artifacts/dsv4-vq-e8p-g512 --blocks layers.0
```

Roundtrip verification binds a real block through the adapter's own loader and
checks three things: zero unbound experts, zero dense routed experts, and
forward output matching the fit's reconstruction. It also runs a
**gate/up-swapped control** — a mis-named artifact still scores a 0.999 cosine
and produces finite output, so the discrimination ratio is what proves the
projection naming, not the agreement number.

For programmatic use, bind through the family registry rather than importing an
adapter directly:

```python
from ramp.models.registry import resolve_family
family = resolve_family("deepseek_v4_flash")
```

### Benchmarks and evaluation

```bash
# Quality against a captured teacher cache.
uv run python benchmarks/eval_dsv4_vq_teacher_cache.py plan
uv run python benchmarks/eval_dsv4_vq_teacher_cache.py run     --split report --stratify 8
uv run python benchmarks/eval_dsv4_vq_teacher_cache.py control --split report --stratify 8
uv run python benchmarks/eval_dsv4_vq_teacher_cache.py summarize
```

Always run `control`. It measures the **uncompressed** model against the same
teacher cache, and without it a divergence number is uninterpretable. On this
family the control scored a KL divergence of 3.5e-10 and top-1 agreement of
1.000, which both validated the pipeline and revealed that a raw perplexity
ratio of 1.088 appears even for a bit-equivalent model — an artifact of targets
falling outside the teacher's captured top-K. Subtracting the control is what
turns a raw ratio into a compression cost.

```bash
# Kernel selection on a real artifact, not synthetic weights.
uv run python benchmarks/bench_glm45_air_projection_kernels.py \
  --variant nax_e8p_fp16_sorted_steel_m32n64_raw \
  --artifact-dir ~/keep-artifacts/dsv4-vq-e8p-g512 --artifact-block-module ffn \
  --vq-code-bits 16 --vq-preferred-group-size 512 --experts 256 --top-k 6

# Byte-exact verify-row parity, and the verify-window dispatch benchmark.
uv run python -m pytest tests/test_gather_vqmm_verify_rows.py -q
uv run python benchmarks/bench_gather_vqmm_verify_rows.py --include-dispatch-flip
```

This machine shows up to **1.8x within-process** and **1.9x across-run** timing
variance. Quote ratios with controls, not absolute microseconds. The
quiet-window harness that would fix this is still unbuilt.

Tests: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest <paths> -q`.
Nineteen failures in the GLM-4.5-Air and GLM-5.2 suites are pre-existing;
check a clean worktree before attributing a failure to your change.

## 4. Current DeepSeek-V4-Flash-0731 figures

Source model: 304.18 B logical parameters measured across all 48 shards,
163 GB on disk. Routed experts ship as FP4 packed two-per-byte in `I8` with
E8M0 group-32 scales; residents (attention **and** shared experts) as `F8_E4M3`
with 128×128 block scales. The MTP drafter is three full MoE blocks.

### Artifact

| | |
| --- | ---: |
| Blocks materialized | 46 (43 backbone + 3 drafter) |
| Rate | 2.031 bpw, E8P 16-bit codes, group size 512 uniform |
| Routed payload on disk | **75.25 GB** (138 files, all SHA-verified) |
| Full runnable model | 82.5–85.1 GB *(residents priced, not yet built)* |
| Versus source | **1.94x smaller** than an already-4-bit release |
| Headroom on a 128 GB Mac | 43–45 GB |
| Materialization wall-clock | 1.77 h |
| Audit | zero dense routed experts, zero unbound experts |

### Quality — full report split, 30 sessions, ~316 K supervised positions

| Metric | Value | Threshold | |
| --- | ---: | --- | --- |
| Mean KL divergence | 0.1487 (upper bound 0.1763) | ≤ 0.30 | **pass** |
| Top-1 agreement | 0.8993 | ≥ 0.85 | **pass** |
| Top-5 / top-10 agreement | 0.9841 / 0.9926 | — | |
| Raw perplexity ratio | 1.156 | — | |
| Paired source control | KL 3.5e-10, top-1 **1.000** | — | pipeline exact |

Compression-attributable perplexity cost is **~3.3 %** on the paired
8-session comparison; the full-30 equivalent is unknown because the control has
only been run on 8. A stratified 8-session subset read top-1 0.9335 — do not
quote subset figures as split figures. The 37-session holdout is **sealed**
until the final release-candidate gate.

KL divergence uses a tail-corrected estimator over the shared support, using the
teacher's stored `logsumexp` and `tail_mass` rather than pretending the
discarded probability mass is zero. It reports a bracket, and both ends pass.

### Speed — the open question

An end-to-end paired comparison measured the compressed model **2.4x slower**
than the FP4 source (25.6 vs 61.6 tok/s) while streaming 48 % *fewer* bytes.
Decomposition ([analysis](docs/research/dsv4-vq-forward-throughput-analysis-20260819.md))
put metrics at ~2 %, I/O as a modest *win*, and the entire gap in the expert
projection kernel.

Root cause: `native/vq_nax_ext` had been compiled against Python 3.14 while the
environment ran 3.12, so all ~50 E8P tensor-unit kernels were unreachable and
the 16-bit artifact fell through to a scalar path. Rebuilt, the A/B on the real
artifact (layer 0, 256 experts, top-6, 1024 tokens, total ms across gate/up/down
— `artifacts/benchmarks/dsv4-e8p-kernel-ab-20260819.jsonl`):

| Kernel | total ms | vs shipped path |
| --- | ---: | ---: |
| `nax_e8p_fp16_sorted_steel_m32n64_raw` | 30.24 | **8.36x** |
| `nax_e8p_fp16_sorted_steel_raw` | 31.87 | 7.93x |
| `nax_e8p_packed_rhs_sorted_tiled_raw` | 34.46 | 7.33x |
| `vq_e1` — what production took | 252.63 | 1.00x |

So the "compressed model is slower" verdict was measured on a broken path. With
~8x back on the term that *was* the whole gap, the compressed model should be
faster per token than the source. **That is an expectation, not a measurement.**
Production dispatch does not select these kernels yet, and decode at M=1 has
never been measured for this family.

## 5. Next steps and roadmap

Ordered. Each gates the next. [`docs/HANDOFF.md`](docs/HANDOFF.md) carries a
copy-paste prompt with the full detail.

1. **Wire production dispatch to the E8P kernels.** `route_strategy="auto"`
   still sends 16-bit codes to the scalar path. Gate the change on
   byte-exactness or a documented tolerance, pin the decision table with tests,
   fix the profile's stale `default_engine`, and make an unavailable NAX
   extension **loud** — that silence cost days and produced two wrong
   conclusions.
2. **Re-measure end-to-end, prefill and decode.** Decode at M=1 is the regime
   the speed claim sells and has never been measured here. If the compressed
   model is now faster per token, report the number; if not, say so and stop
   before building on it.
3. **Build the residents.** Turn the 82.5–85.1 GB accounting into bytes on disk.
4. **MTP verify runtime and the headline measurement.** Wire the merged drafter
   to the verify kernels with acceptance-rate instrumentation, then run
   compressed + speculative against the autoregressive baseline under
   quiet-window rules. The drafter has no numerical reference yet, so treat a
   surprisingly good acceptance rate as a suspect result.
5. **Release candidate.** Touch the sealed holdout once, package per
   [`docs/KEEP_NEW_MODEL_FAMILY_TEMPLATE.md`](docs/KEEP_NEW_MODEL_FAMILY_TEMPLATE.md).

Queued behind those: generate the 120-session MTP training split (~154 GB) for
drafter recovery; build the quiet-window benchmark harness, deferred twice; move
the drafter's dense bf16 experts (~39 GB) to native mxfp4 (~10 GB); port the
remaining NumPy hot paths in the fit to MLX.

### Known limitations

- The full runnable size is accounting, not bytes — residents are unbuilt.
- No MTP binder exists, so drafter artifact *discovery* is unproven (naming is
  verified by roundtrip).
- The drafter is a faithful transcription with no numerical reference.
- Machine timing variance is 1.8–1.9x; the quiet-window harness is unbuilt.
- Evidence under `artifacts/` is gitignored. Force-add anything load-bearing —
  an earlier wave reported committing its benchmark JSON and silently lost it.

## Requirements

- Apple silicon Mac, macOS 14 or newer.
- [`uv`](https://docs.astral.sh/uv/) for the environment and the `keep` CLI.
- Local disk for model snapshots: DeepSeek-V4-Flash-0731 is 163 GB; the
  compressed artifact adds 75 GB; Qwen3.6-35B-A3B is about 72 GB.
- Xcode command-line tools for the native extension.

## Documentation map

- [`docs/HANDOFF.md`](docs/HANDOFF.md) — current state, traps, next steps.
- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — beginner walkthrough.
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — setup, cache, memory, build recovery.
- [`docs/KEEP_NEW_MODEL_FAMILY_TEMPLATE.md`](docs/KEEP_NEW_MODEL_FAMILY_TEMPLATE.md) — onboarding another family.
- [`docs/GLM45_AIR_RC_PIPELINE.md`](docs/GLM45_AIR_RC_PIPELINE.md) — the earlier GLM-4.5-Air RC pipeline.
- [`docs/research/`](docs/research/) — analyses, model-card drafts, publication evidence.
- [`WORK_LOG.md`](WORK_LOG.md), [`DISCOVERY.md`](DISCOVERY.md) — run ledger and durable facts.
