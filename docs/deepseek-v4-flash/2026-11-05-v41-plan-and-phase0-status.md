# DeepSeek-V4.1-Flash on one 128 GB Mac: plan and Phase 0 status

Date: 2026-11-05. Branch `jack/hopeful-cori-iszsfq`, draft PR jackmaxwil/keep#3.
Target machine: M4 Max, 40-core GPU, 128 GB. Memory bandwidth is 546 GB/s
theoretical; MLX reads 474-477 GB/s (1 GiB reductions) and a bf16 matvec
458 GB/s, measured 2026-11-05. The 409.4 GB/s used for the estimates below is
an older, lower measurement, so those estimates are conservative.

## The plan in brief

This synthesises three earlier analyses (oMLX's oQ quantization, oMLX's V4.1
Engram/offload work, and REAP pruning) and checks them against this repo.

- **Memory and speed come from different weights.** Each routed expert kept in
  every layer costs 0.359 GB of RAM (V4.1, 2.031 bpw) but adds no per-token
  reads. Resident weights and kernels set decode speed. So: prune to fit, fix
  residents and kernels to go fast, and give MTP whatever memory is left.
- **Speed is a kernel problem first.** The measured V4-Flash decode floor is
  270–390 ms/token (roadmap). On an M4 Max, 16-bit E8P experts ran on the
  scalar kernel, because NAX is M5-only and the M=1 fast paths were 8-bit-only.
- **Fit forces a ~50% prune**, about 176–192 of 384 experts per layer. That is
  where REAP's published results lose most outside code (Kimi-K2 at 50%: −17.6%
  multiple choice, −54% multi-turn tool use). KEEP's quality gate is in-domain,
  so an out-of-domain teacher split must decide the expert count.
- **Plausible end state (estimate):** ~31 tok/s autoregressive and 37–53 with
  MTP, at 81–87 GB in flight. Engram's 203 GB stays on SSD.

### The top 20, by execution order

Required for V4.1 to exist at all:
- **R1:** port V4.1 with reference parity.
- **R2:** read Engram from SSD.
- **R3:** REAP done right (real-gate saliency plus router slicing).

Ranked levers:
1. FP8 residents as mxfp8 byte copies
2. Out-of-domain teacher split to choose the calibration mix and N
3. Non-NAX 16-bit E8P kernels
4. Standard benchmarks compared with oMLX and GGUF builds
5. Quiet-window harness that gates on inbound paging
6. Decode-floor fusion (sinkhorn, descriptors, compile)
7. REAP-ordered fast/exact residency (SSD tier)
8. MTP stack
9. Recovery distillation
10. Ship V4-Flash first
11. Persistent prefix-KV cache
12. Prune-vs-bits frontier
13. Resident trims (head and embedding to 8 bits)
14. Two-Mac RDMA calibration
15. 4-bit attention "turbo"
16. 4-bit Engram tables
17. Fit polish

## Phase 0: done in this branch (no V4.1 download needed)

| Plan item | Commit | Status |
|---|---|---|
| 1 mxfp8 residents (incl. `wo_a`) | `3c6d9f1` | Built, opt-in. Bit-exact on CPU at all 8 V4-Flash shapes. **Speed untested** |
| 5 inbound-paging gate (row schema v2) | `450c925` | Built. Unit-tested |
| R3 REAP saliency + router slicing | `bd7fd54` | Built. Matches a float64 reference; slicing equals masking |
| 7 residency replay (SSD-tier kill test) | `33a5681` | Built. Hand-checked miss counts |
| 6 fused sinkhorn | `5baed30` | Built, opt-in. **Metal untested** |
| 3 single-token E8P on M1–M4 | `00580c4` | Built, opt-in. **Metal untested** |
| Kill-test runner and runbook | `7deda77` | `scripts/phase0_kill_tests.sh`, `2026-11-05-phase0-kill-tests.md` |

Regression check (Linux, CPU MLX): on the 33 test files that import changed
modules, the failure set is identical to `main`'s (308 failures needing Metal
or Mac-local files), plus 22 newly passing tests.

## Next actions, in order

1. On the M4 Max, run `scripts/phase0_kill_tests.sh`. Each test is PASS, KILL
   or TUNE per the runbook.
2. Rerun V4-Flash calibration into a **new** directory, then run
   `finalize-reap`, `reap-select --keep 128` and the residency replay.
3. If mxfp8 passes, bind the composite with `--resident-format mxfp8` and rerun
   the headline series. Quote v2 rows only.
4. Fix CI. `validate-recipes` fails on `main` because strict recipes refuse
   `--no-hash`, so the pytest step never runs. The proposed workflow patch is
   in the PR description.
5. Then Phase 1 (V4.1 port, Engram, two-Mac calibration), which waits on your
   decisions below.

## Open decisions (yours)

- **Serving machine.** M4 Max (this plan) or M5 Max (decision D2, 2026-10-06).
- **Scope.** Coding-agent specialist or generalist. This sets the calibration
  mix and the prune depth.
- **Disk and download.** The V4.1 source is 510 GB; the M4 had 73 GiB free on
  Oct 6.
- **Exact mode.** In v1 or later. It adds 69 GB to the download.

## Known gaps

- MTP headline rows require the M5-only `nax_e8p_m32n64`, so they won't
  validate on an M4.
  On a synthetic V4-Flash FFN (256 experts, top-6, 43 chained FFNs per sync)
  the same kernel now also serves MTP verify windows (<= 32 tokens): 2.6-5.5x
  the previous M4 path at 2-21 tokens and 9x at 32, outputs within 5.7e-4.
  It still is not traced as a verify implementation, so the MTP headline
  harness, which expects `nax_e8p_m32n64`, needs updating before M4 rows can
  validate.
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
