# Task 2 report — real-artifact prefill and M=1 gate

Status: DONE — STOP

Baseline: `e4b47523e5997458a05607ab252106520731400e` on local `main`.

## Gate verdict

**STOP Tasks 3–7. Do not continue the campaign.**

The decision quantity for the clean M=1 component benchmark is
`source FP4 projection time / production VQ E8P projection time`, which is the
compressed-path speed relative to source. Across nine clean fresh processes,
the median paired ratio is **0.0934906005x** (IQR **0.0884375729–0.0972339658x**;
range **0.0866981820–0.0985019441x**). The production VQ path is therefore
approximately **10.70x slower** than native FP4 for the measured layer-0 M=1
expert projections. It was faster in **0/9** clean rows.

The full-model streaming forward/prefill rerun also observed a non-passing
ratio: **27.9380 tok/s / 62.7579 tok/s = 0.4451716951x** against the exact
eight-session FP4 source control. However, every candidate and source row
recorded pageouts and several recorded large swapout deltas, so this ratio is
completeness evidence, not a portable or precise speed claim. The independent
clean M=1 result is sufficient to force STOP without leaning on contaminated
prefill timing.

No Task 3 work was started.

## Full-model streaming forward/prefill

### Command and lifecycle

The real-artifact evaluator used this command under the normal cooperative
`.keep-heavy-job.lock`, launched from a persistent shell with `nohup` and
`disown`:

```bash
env -u GLM_MLX_WIRED_LIMIT_GB \
  -u GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB \
  UV_CACHE_DIR=/tmp/keep-uv-cache \
  uv run --group dev python \
  benchmarks/eval_dsv4_vq_teacher_cache.py run \
  --split report --stratify 8 \
  --checkpoint /Users/jack.mazac/models/DeepSeek-V4-Flash-0731 \
  --artifact-dir /Users/jack.mazac/keep-artifacts/dsv4-vq-e8p-g512 \
  --teacher-dir /Users/jack.mazac/keep-artifacts/dsv4-teacher-logits-eval \
  --pack /Users/jack.mazac/models/teich/dsv4-coding-agent-v1-20260811.json \
  --output /Users/jack.mazac/keep-artifacts/dsv4-task2-20260819/prefill-vq
```

The operator paused the run twice to free the laptop. Both pauses terminated
only the active evaluator, preserved durable rows, and released the flock. The
first resume authenticated and skipped two rows; the second authenticated and
skipped five rows. The final process (`uv` PID 96442, evaluator PID 96449)
terminated normally with:

```text
{"rows": 8, "phase": "complete", "sessions_done": 3,
 "sessions_skipped": 5, "prompt_id": null,
 "tokens_per_s": 23.4, "eta_h": null}
```

Terminal audit: evaluator PIDs absent, flock acquirable, status
`phase=complete`, exactly 8 rows and 8 unique prompt IDs. No completed row was
duplicated. Both custom wired-limit variables remained unset. The 37-session
holdout was not read or summarized.

Run directory:
`/Users/jack.mazac/keep-artifacts/dsv4-task2-20260819/prefill-vq`

### Stable identities

- Checkpoint: `/Users/jack.mazac/models/DeepSeek-V4-Flash-0731`
- Revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- Checkpoint config SHA-256:
  `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- Checkpoint index SHA-256:
  `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- VQ artifact manifest SHA-256:
  `a996c4bedc514f49c54fe15d5d9b2373f839678e55b241861f2fda60cb4bf8b3`
- VQ file inventory SHA-256:
  `4bda0cf1a635123d08f711cd38ad4bb3f716b41714883495ec91cfd71540e843`
- E8P codebook SHA-256:
  `efc2c03c60acd955dc812ff2bade9ec6cc31259807e48473506161f3a9a230ce`
- Teacher generation config SHA-256:
  `c9d68af1c4939b9281571f5fae69ea51c482928f6ea0c9a7cd2d398131d240f1`
- Split `report`, position slice 256, prefill chunk 1024, 256 experts,
  top-6, code bits 16, group size 512.

The exact ordered prompt IDs and token counts match the prior FP4 source
control. The retained source run payload has config/index hashes, checkpoint
path, pack path, teacher hash, split, position slice, prefill chunk, and all
eight selected-session records identical to the candidate's corresponding
stable fields. Expected measured-path fields differ: engine, expert mode,
expert bytes/span, artifact metadata, and resident bind time.

### Volatile run-config hash limitation

The candidate rows contain three run-config hashes across the interrupted and
resumed process lifetimes:

```text
8d8520e5d82158d3b2b5f5e941fc48472a4fce73f14b6bde905f3671cb865c3b
fdd9f5aa722eac3670889f5a01e81bc5cad053a00b50e43f83f0b7c6551d9e8a
99f75386306f0306a61b36a5135f5b2494ce06dace83331fcbc52236c9d9e0b5
```

`resident_bind_seconds` is included inside the hashed `identities` object, and
the runner overwrites `run-vq_e8p_streamed.json` on resume. Only the final
payload (`99f753...`, resident bind 9.68 seconds) remains. Therefore the exact
byte-level difference for the earlier two hashes cannot be proven after the
fact and is not claimed. No hash or evidence was rewritten. Instead, the
checkpoint, artifact, teacher, and ordered-session identities above are
authenticated directly from every durable row plus the retained run payload.

### NAX production-path proof

The evaluator bind proof records a real `QuantizedVQSwitchGLU`, no dense routed
expert parameters, 16-bit E8P, group size 512, 256 experts, and
`route_backend=gather_vqmm_auto`.

A separate short real-artifact layer-0 dispatch proof used the evaluator's
1,024-token prefill chunk and top-6 routing under the heavy lock. It observed
the production auto selector choose `nax_e8p_m32n64`; all three projection
calls reached that implementation and the actual native kernel wrapper was
called three times:

| projection call | geometry | routes | implementation |
|---|---:|---:|---|
| gate | 4096→2048 | 6144 | `nax_e8p_m32n64` |
| up | 4096→2048 | 6144 | `nax_e8p_m32n64` |
| down | 2048→4096 | 6144 | `nax_e8p_m32n64` |

The BF16 output was finite. This proves dispatch only; its diagnostic duration
is not a speed claim.

### Eight-session results

| row | prompt ID (suffix abbreviated only in this table) | tok/s | mean KLD | top-1 | pageouts | swapouts | clean |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `teich_claude_agent-ab6dc2ab37c3c2c34` | 38.7 | 0.124404 | 0.927554 | 1,610 | 0 | no |
| 2 | `...07-11T18-14-54...` | 38.6 | 0.123011 | 0.903285 | 7,684 | 0 | no |
| 3 | `...07-10T14-23-17...` | 38.1 | 0.099531 | 0.924511 | 3,045 | 58,409 | no |
| 4 | `...07-06T09-36-52...` | 33.5 | 0.141047 | 0.932022 | 5,615 | 244,092 | no |
| 5 | `...05-27T21-21-35...` | 32.9 | 0.122931 | 0.953453 | 3,616 | 73,484 | no |
| 6 | `...07-02T12-37-58...` | 25.0 | 0.097812 | 0.947272 | 11,400 | 128,988 | no |
| 7 | `...06-19T10-50-52...` | 22.7 | 0.223430 | 0.923636 | 10,807 | 328,448 | no |
| 8 | `...07-03T15-31-42...` | 23.8 | 0.112698 | 0.932992 | 7,946 | 10,044 | no |

Aggregate formula: `361,366 tokens / 12,934.56 forward seconds = 27.9380
tok/s`. Row tok/s median is 33.2, IQR 24.7–38.225, range 22.7–38.7.

The named broken-path exact-eight aggregate is 25.8138485 tok/s, so the
observed rerun ratio is 1.0822881x. This apparent 8.2% improvement is not
decision-usable as a precise speedup because the candidate rows span zero to
328,448 swapouts and all have pageouts. In particular, row 3's +58,409
swapouts is a major timing limitation, not noise; rows 4, 6, and 7 are worse.

### Source control and quality

The prior exact FP4 control was reused rather than rerun:

- Run directory: `/Users/jack.mazac/keep-artifacts/dsv4-quality-gate`
- Engine: `source_mxfp4_streamed`
- Run config SHA-256:
  `cf7150bcf7954a7b61a53ffa8071181bc32d0c262da626631b0d39d88fd03e86`
- Aggregate: `361,366 / 5,758.10 = 62.7578542 tok/s`
- Exact ordered session and token-count match: yes
- Stable checkpoint/config/index/teacher/run-shape identity match: yes
- Clean source rows: 0/8

It was not rerun because both full-model sides are contaminated and therefore
completeness-only, while the clean M=1 result independently forces STOP.

Candidate quality is position-weighted mean KLD **0.1409330**, top-1 agreement
**0.9339517**, and pooled p99.9 KLD **10.5493102**. The inherited gates pass
mean KLD (`<=0.3`) and top-1 (`>=0.85`) but fail p99.9 KLD (`<=3.0`). Versus
the prior broken-path exact eight, mean KLD improves from 0.1429486 and top-1
improves from 0.9334717, while pooled p99.9 KLD worsens from 10.2843871 to
10.5493102 (+2.58%). Thus aggregate quality remains close, but Task 2 does not
claim full quality acceptance or a cleared tail gate.

## M=1 real-artifact expert projections

### Minimal benchmark rail

`benchmarks/bench_dsv4_m1_expert_projections.py` reuses the existing real
checkpoint shard reader/source expert loader, real VQ artifact loader, metric
counters, and cooperative heavy lock. Each `run` invocation is one fresh
process. It verifies DSV4 geometry from `config.json`, loads pinned layer-0 FP4
weights through native `mx.gather_qmm:mxfp4`, loads all three real VQ
projections, executes production `route_strategy=auto`, counts calls to the
actual `nax_e8p_fp16_sorted_steel_m32n64` wrapper, and interleaves source/VQ
orders within every projection.

Representative row command (run index, order, and seed varied across nine
fresh processes):

```bash
env -u GLM_MLX_WIRED_LIMIT_GB \
  -u GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB \
  UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONPATH=.:src \
  uv run --group dev python \
  benchmarks/bench_dsv4_m1_expert_projections.py run \
  --append-jsonl artifacts/benchmarks/dsv4-task2-20260819/m1-rows.jsonl \
  --run-index 1 --order vq_first --seed 20260820 \
  --iterations 21 --warmup 3 --inner-loops 5
```

The nine rows use nine unique PIDs, 5 `vq_first` and 4 `source_first` starts,
21 interleaved samples per projection, 5 inner calls, 3 warmups, and 324
observed production NAX wrapper calls per process. All nine rows have zero
pageouts and zero swapouts.

Stable identity:

- Checkpoint revision/config/index: same values as full-model evidence.
- Pinned layer-0 source shard:
  `model-00002-of-00048.safetensors`, 3,566,321,192 bytes, revision
  `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, etag
  `77b26c939a0e25b3113c8d6bb04e1901a748bd4a7d2589e3bfdaabdf1e9bba14`.
- Artifact manifest, inventory, and codebook hashes: same values above.
- Kernel: `nax_e8p_fp16_sorted_steel_m32n64`.
- Source backend: `mx.gather_qmm:mxfp4`.
- Geometry: gate/up 4096→2048, down 2048→4096, 256 experts, top-6, M=1.

### Result

| projection | median VQ ms | median source ms | median paired source/VQ | paired IQR | paired range |
|---|---:|---:|---:|---:|---:|
| gate | 10.9214 | 1.0320 | 0.0913730x | 0.0877598–0.0987163 | 0.0714290–0.1022601 |
| up | 11.1853 | 1.0619 | 0.0947615x | 0.0903037–0.0979224 | 0.0853358–0.1011655 |
| down | 10.7870 | 0.9292 | 0.0948563x | 0.0923483–0.0982987 | 0.0765283–0.0989030 |
| total | 32.7062 | 2.9881 | **0.0934906x** | **0.0884376–0.0972340** | **0.0866982–0.0985019** |

The total uses the paired per-process sum of the three projection medians, not
a ratio of independently pooled medians. The summary command intentionally
returns exit 2 for `gate=STOP`; revalidation observed that exact expected exit.

This is **component-level layer-0 M=1 expert-projection evidence only**.
Residents do not exist yet, so it is not a full-model decode, end-to-end decode,
tokens-per-second, or user-visible latency claim.

## RED → GREEN

Initial module RED, before benchmark implementation:

```text
ModuleNotFoundError: No module named
'benchmarks.bench_dsv4_m1_expert_projections'
```

The first implementation GREEN was:

```text
2 passed
```

Self-review found that the summary authenticated the kernel identity string but
did not fail closed on a zero call count. A regression row with
`kernel_calls=0` was added first. Focused RED:

```text
.F
FAILED test_summary_fails_closed_on_identity_drift_or_a_non_faster_total
Failed: DID NOT RAISE ValueError
1 failed, 1 passed, 2 warnings in 4.29s
```

Minimal GREEN added one validator: every row must record
`kernel_calls > 0`. Fresh focused result:

```text
2 passed, 2 warnings in 1.64s
```

The warnings are existing SWIG deprecations for `SwigPyPacked` and
`SwigPyObject`; there were no task warnings.

Final self-review then challenged the declarative `fresh_process=true` field.
A duplicated-PID regression failed first with `DID NOT RAISE ValueError`
(1 failed, 2 passed), after which the summary was made to require a distinct
integer PID for every row. Fresh GREEN: `3 passed, 2 warnings in 0.89s`.

## Verification

Focused plus relevant production-dispatch tests:

```bash
env -u GLM_MLX_WIRED_LIMIT_GB \
  -u GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB \
  UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONPATH=.:src \
  uv run --group dev pytest -q \
  tests/test_dsv4_m1_expert_projections.py \
  tests/test_vq_switch_nax_e8p_auto.py \
  tests/test_switch_routing.py
```

```text
32 passed, 2 warnings in 1.14s
```

Formatting and lint:

```text
ruff format --check: 2 files already formatted
ruff check: All checks passed!
```

Evidence validation:

```text
jq empty artifacts/benchmarks/dsv4-task2-20260819/*.json  # exit 0
prefill-vq-rows.jsonl: 8 rows
m1-rows.jsonl: 9 rows
M=1 summarize: gate STOP, expected exit 2
```

## Evidence

Committed evidence root:
`artifacts/benchmarks/dsv4-task2-20260819/`

- `prefill-vq-rows.jsonl` — exact eight durable rows.
- `prefill-vq-run.json` — retained final-resume run payload.
- `prefill-vq-summary.json` — evaluator summary and quality metrics.
- `prefill-vq-bind-proof.json` — full-model VQ bind proof.
- `prefill-dispatch-proof.json` — real-artifact 1,024-token NAX observation.
- `prefill-control-comparison.json` — stable identity authentication, source
  control, ratios, memory limits, quality comparison, and hash limitation.
- `prefill-source-control-identity.json` — immutable raw source identity,
  run-shape, session, and provenance extract.
- `prefill-source-control-run.json` — exact copy of the reused FP4 run payload.
- `prefill-source-control-rows.jsonl` — exact eight FP4 rows extracted from the
  pre-existing mixed-engine rows file.
- `m1-rows.jsonl` — nine clean fresh-process paired rows.
- `m1-summary.json` — robust projection distributions and STOP verdict.

The exact external run directories remain preserved. New ignored evidence was
force-added explicitly; no prior evidence was deleted.

## Self-review and concerns

- The benchmark imports existing loaders, source-reader logic, memory counters,
  artifact identity, and lock semantics. It adds no dependency or generalized
  framework.
- Row validation fails closed on record type, distinct fresh-process PIDs, production kernel
  identity and positive call count, native FP4 source backend, all three
  projections, stable identities, minimum clean rows, and both interleaving
  orders.
- The benchmark records every sample and checksum, but checksums are diagnostics,
  not a numerical-equivalence gate between different quantizations.
- The full-model result is explicitly not promoted to a precise ratio because
  swapping dominates. The source control is identity-matched but also dirty.
- The evaluator quality verdict is not green: inherited p99.9 KLD remains above
  threshold. STOP is therefore conservative on both speed and tail quality.
- The volatile run-config hashes cannot be fully reconstructed because resume
  overwrote earlier payloads. This is reported rather than silently treating
  them as identical.
- The M=1 evidence is not promoted to a model-level decode claim.
- No production model/runtime code changed. No cloud or external system was
  mutated. No push was performed. The holdout remained sealed.

Concern: the decisive clean component result is dramatically negative, but it
does not diagnose whether future resident/full-model work could amortize other
costs. That question belongs to a new explicitly authorized campaign; it does
not weaken this task's required STOP boundary.

## Fix Round 1 — reconstructible source identity and empty-env guard

Review found that the initial `prefill-control-comparison.json` asserted one
shared identity object. It named hashes of the external source files, but a
reviewer could not reconstruct the source side using only the committed Task 2
packet. Review also found that the wired-limit guard used environment-value
truthiness, so a forbidden variable present with an empty value bypassed it.

### Immutable source-control extract

No heavy evidence was rerun. The exact existing FP4 control was mechanically
extracted and copied:

```bash
jq -c 'select(.engine == "source_mxfp4_streamed")' \
  /Users/jack.mazac/keep-artifacts/dsv4-quality-gate/rows.jsonl \
  > artifacts/benchmarks/dsv4-task2-20260819/prefill-source-control-rows.jsonl

cp /Users/jack.mazac/keep-artifacts/dsv4-quality-gate/run-source_mxfp4_streamed.json \
  artifacts/benchmarks/dsv4-task2-20260819/prefill-source-control-run.json
```

Observed provenance:

| file | SHA-256 |
|---|---|
| external mixed rows file | `3c7b91e354e49433cb6b84f324dff338e1d0ef8e117ddb7e516cdf1bf5b8a22c` |
| committed exact-eight source extract | `12e1f7db2753dca0b00621dc4bb1fc0a56fa89514c492e3803b4f86f428312b6` |
| external source run payload | `995182195ae136526358c9d0f606a027122854d3d542e69b6045647998cd47ac` |
| committed source run copy | `995182195ae136526358c9d0f606a027122854d3d542e69b6045647998cd47ac` |
| pinned `_KEEP_DOWNLOAD_COMPLETE.json` | `68d973a8b79801a90f888c4edc5d7739721aaf2e5dc3698787964d64467cf4eb` |

`prefill-source-control-identity.json` separately records the source's raw
checkpoint path/revision/config/index, teacher path/hash, pack/split/run shape,
run-config hash, ordered full selected-session records, and both external and
committed source-file hashes. The exact-eight source extract has 8 rows, one
engine, one run-config hash, one checkpoint-config hash, one teacher hash, and
the expected ordered token counts.

The historical FP4 run payload did not itself emit checkpoint revision. That
limitation remains explicit: revision is read from the pinned checkpoint's
hashed `_KEEP_DOWNLOAD_COMPLETE.json`; the historical run payload separately
matches that checkpoint's config and index hashes. No stronger contemporaneous
revision claim is made.

`prefill-control-comparison.json` now contains two independent raw extracts,
`stable_identity_extracts.candidate` and
`stable_identity_extracts.source`. Each has its own committed evidence hashes,
checkpoint fields, teacher fields, run shape, selected-session hash, ordered
prompt IDs, and ordered token counts. `stable_identity_matches` is checkable by
directly comparing those two sides rather than trusting a shared assertion.

The verification reconstructed the source extract from its committed run and
row files, then derived all 15 match fields and `all_required_fields_match`
from the two separate comparison objects. Both checks returned `true`:

```text
candidate/source raw-field equality derivation: true
source run/rows reconstruction: true
identity-reconstruction: PASS
```

This strengthens identity auditability only. Both full-model timing sides
remain dirty, and the 0.4451716951x observed prefill ratio remains
completeness-only.

### Wired-limit RED → GREEN

The regression sets `GLM_MLX_WIRED_LIMIT_GB` to the empty string and calls the
real row entry point. It must reject the environment before touching model
arguments. RED before the guard change:

```text
FAILED test_run_rejects_an_empty_custom_wired_limit
AttributeError: 'NoneType' object has no attribute 'checkpoint'
1 failed, 2 warnings in 1.71s
```

That failure proved the empty variable bypassed the guard. The minimal change
uses membership (`name in os.environ`) rather than value truthiness. GREEN:

```text
1 passed, 2 warnings in 0.87s
```

The guard now rejects either forbidden variable whenever it is present,
including an empty value. The normal benchmark and all evidence runs continue
to require both variables to be absent.

### Fix-round verification

Focused and relevant tests after the code change:

```text
33 passed, 2 warnings in 1.45s
```

The warnings remain the existing SWIG deprecations. Final verification also
reported `2 files already formatted`, `All checks passed!`,
`tests-lint-json-hashes-diff: PASS`, and
`source-and-paired-identity-reconstruction: PASS`. No heavy job, Task 3 work,
cloud work, push, or holdout access occurred in this fix round.
