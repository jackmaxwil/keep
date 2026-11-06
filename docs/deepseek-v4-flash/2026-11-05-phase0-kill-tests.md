# Phase 0 kill tests: toward DeepSeek-V4.1-Flash on one 128 GB Mac

Date: 2026-11-05. Target machine: the M4 Max (40-core GPU, 128 GB; 546 GB/s
theoretical, 474-477 GB/s measured MLX read on 2026-11-05). Phase 0 is the work that needs no V4.1 download and can be proven
on the V4-Flash artifact. Every runtime change below is **opt-in**; defaults
are unchanged until the kill test that guards it passes on the target Mac.

Run everything that needs no model download in one go:

```bash
scripts/phase0_kill_tests.sh
```

It writes JSON evidence to `artifacts/benchmarks/phase0-<host>-<date>/`. That
path is gitignored, so force-add it (`git add -f`).

## What each test decides

| # | Change | Opt-in switch | Kill test | KILL when | If it passes |
|---|---|---|---|---|---|
| 1 | FP8 residents kept as source bytes (weights-only mxfp8, incl. `wo_a`) | `resident_format="mxfp8"`; `--resident-format mxfp8` on the headline harness | `benchmarks/bench_dsv4_mxfp8_residents.py` | any weight differs from the bf16 decode; mxfp8 < 1.5x bf16 at M=1; > 1.1x affine-8's time | bind the composite with mxfp8 and rerun the headline series |
| 3 | Single-token E8P kernels for 16-bit codes on GPUs without NAX | `RAMP_E8P_M1=1` | `benchmarks/bench_e8p_m1.py` | outputs disagree, or < 100 GB/s effective per routed FFN, 43 chained FFNs per sync | >= 200 GB/s: make it the M4 default. 100-200: tune (decoding each codeword once was tried and is slower) |
| 5 | Headline rows also gate memory coming back in (schema v2) | always on for new rows; v1 rows keep their rule | the next headline series | — | quote only v2 rows |
| 6 | One-dispatch hyper-connection sinkhorn | `RAMP_HC_SINKHORN=metal` | `benchmarks/bench_hc_sinkhorn.py` | < 3x faster at decode shape, or > 2e-6 relative difference | set it for the headline |
| R3 | REAP saliency from the real gate, in the calibration pass | always collected in `calibration` mode | `produce_dsv4_teacher_cache.py reap-select --keep 128` on V4-Flash stats | median pruned-set Jaccard vs route frequency >= 0.9 (REAP adds nothing) | use REAP order for the prune and the hot set |
| 7 | Expert hot set in RAM, rest on SSD | none (simulation only) | `benchmarks/replay_dsv4_expert_residency.py` | best fetching policy misses > 5% of routed slots at the hot-set size you'd ship | worth building the SSD tier |

Tests R3 and 7 need a calibration run, because the per-session REAP statistics
and recorded routes are new. Rerun the V4-Flash calibration split (5.11 h last
time) into a **new** output directory. The runner resumes by skipping sessions
that already exist, so the old directory would keep sessions without the new
arrays, and `finalize-reap` refuses those. Then:

```bash
uv run python benchmarks/produce_dsv4_teacher_cache.py finalize-reap --out-dir ~/keep-artifacts/dsv4-teacher-calibration-reap
```

```bash
uv run python benchmarks/produce_dsv4_teacher_cache.py reap-select --stats ~/keep-artifacts/dsv4-teacher-calibration-reap/reap-stats.npz --keep 128
```

```bash
uv run python benchmarks/replay_dsv4_expert_residency.py --calibration-dir ~/keep-artifacts/dsv4-teacher-calibration-reap --reap-stats ~/keep-artifacts/dsv4-teacher-calibration-reap/reap-stats.npz --hot 96,128,160 --expert-mb 6.39
```

The replay's default `--expert-mb` is V4.1's 8.99 MB. A V4-Flash expert is
6.39 MB (3 x 2,129,920 B).

## What was verified where

On Linux with CPU MLX 0.31.2:

- the mxfp8 byte copy dequantizes bit-identically at all eight V4-Flash
  resident shapes, and `mx.quantize(mode="mxfp8")` does not (608 of 98,304
  values changed);
- REAP statistics match a float64 reference, and slicing the router equals
  masking it;
- the residency replay reproduces hand-counted LRU and decayed-count misses;
- the sinkhorn reference reproduces the model's compiled path.

Everything Metal (kernel parity, every speed number) is untested until the
script above runs on the Mac.

## Known gaps

- `benchmarks/bench_dsv4_mtp_headline.py` validates speculative rows against
  `EXPECTED_VERIFY_IMPLEMENTATION = "nax_e8p_m32n64"`, so MTP headline rows
  cannot validate on an M4 until the verify window has a non-NAX E8P kernel.
- Decoding each E8P codeword once was tried on the M4 Max on 2026-11-05 and
  made the kernel slower (113 vs 154 GB/s, kernel alone, 256 calls per sync).
  The compiler already lifts the per-codeword work out of the 8-dimension loop.
- `bench_e8p_m1.py` v1 synced after every FFN and called KILL at 52.8 GB/s,
  mostly CPU-GPU round trips. v2 (`ramp_e8p_m1_kill_test_v2`) chains 43
  dependent FFNs per `mx.eval`, as one decode step does. On the M4 Max it
  gives **TUNE**: 122-126 GB/s best (8-16 rows per threadgroup), 2.4-2.5x
  production's 50.5 GB/s, outputs within 6.4e-4. Quote only v2 rows.
- The tuned kernel (`src/ramp/kernels/e8p_m1.metal`, 2026-11-05) then
  **PASSES** v2, narrowly: 201.9-205.5 GB/s best (4 rows per simdgroup), 3.97x
  production, outputs unchanged at 6.4e-4. Each simdgroup owns several rows,
  lanes load 8 codewords at once, and the abs grid is decoded once per
  threadgroup into half2 pairs, with signs applied by XOR. A loads-only stub
  of the same kernel reads at ~410 GB/s, so decode ALU is still half the
  time. Since 2026-11-05 it is the default on any Metal GPU without NAX
  (`RAMP_E8P_M1` unset or `auto`); `RAMP_E8P_M1=0` turns it off, `=1` forces
  it on. It clears the bar by only 1-3%, measured while other jobs were
  loading the machine, so rerun the kill test on an idle machine.
- `bench_dsv4_mxfp8_residents.py` v1 also synced after every call. v2
  (`dsv4_mxfp8_resident_kill_test_v2`) chains 43 dependent calls per
  `mx.eval`. On the M4 Max the verdict flips from KILL to **PASS** in two runs:
  mxfp8 is 1.66x bf16 and takes 1.01-1.05x affine-8's time, still bit-exact.
  The large tensors reach 350-450 GB/s.
