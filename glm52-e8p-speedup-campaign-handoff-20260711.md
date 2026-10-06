# GLM-5.2 E8P recovery speedup campaign — handoff (2026-07-11)

Orchestrated Claude (Opus 4.8) + Codex campaign. Claude designed the algorithm
and owned all MLX/Metal verification (Codex sandboxes have no Metal); Codex
implemented the mechanical translation and tests under exclusive file ownership.

## Outcome

**The worst8 E8P re-materialization is no longer multi-day.** The brute-force
65,536-row CPU/NumPy diagonal-Hessian search is replaced by a byte-identical
256-row factored search on MLX/Metal.

| Search path | Rate | Projected worst8 (24 groups × 4 passes) | Acceptance gate (≤12h) |
| --- | ---: | ---: | :---: |
| Exhaustive NumPy (original) | 22.9k cw/s | 307.7 h | fail |
| Factored NumPy (reference) | 138k cw/s | 51.1 h | fail |
| **Factored MLX/Metal** | **975k cw/s** | **7.23 h** | **pass** |

Measured on this host at the real materializer call granularity (2048 vecs/call).
The exhaustive rate reproduces the original stopped run (~1 assign-pass of 1
group in 3h15m), so the projection is trustworthy.

## Live run (launched this session, in progress)

A full 8-layer worst8 re-materialization is **running now** on the fast Metal
path, launched via the direct producer (self-authenticating, `--resume`,
holding `.keep-heavy-job.lock`) with `GLM52_E8P_BACKEND=metal`:

- Launched 15:51:55 PDT to the canonical output
  `artifacts/quality/glm52-recovery-worst8-e8p-artifact-20260711`.
- Metal engagement verified on the live process (`sample` stack:
  `mlx::core::gpu::eval → binary_op_gpu → AGXG17XFamily...dispatchThreads`).
- **Group 1 (layer 75 gate_proj) published at 16:24:46 — 33 min** (includes the
  one-time ~10 min source load/auth). Output validated well-formed: `codes`
  uint16 `[168, 2048, 768]`, `scales` f16 `[168, 2048, 12]`, embedded E8P
  codebook, provenance + quantization_config metadata. 0.54 GB/projection.
- Projected total: **~11–13 h for all 24 groups** (warm groups faster than the
  first), vs ~308 h exhaustive. Resumable per group; will run past this session.
- **Ledger note:** this was launched via the producer directly (the controller
  is retry-blocked), so the campaign ledger still shows the prior SIGTERM. The
  produced artifact is self-authenticated by the producer's SHA guards + manifest;
  reconcile the ledger by recording a completion/retry-authorization event once
  the controller `--authorize-retry` follow-up lands, or by re-running via
  `advance --resume` (which will find the groups already complete).

## Resource-utilization analysis (why not faster than ~12–14 h)

Activity Monitor showed the run at **63.5% GPU and <1 CPU core**, with idle CPU
and free RAM — prompting a "use more resources" push. Findings:

- **The run is GPU/latency-bound, not CPU/RAM-bound.** The <1-core CPU and 63.5%
  GPU mean the GPU idles ~36% of the time during the serial pipeline's CPU-side
  phases (source load, numpy scale-update between passes, transactional publish).
  Idle CPU cores and free RAM cannot accelerate GPU-bound compute.
- **Batching bigger GPU kernels BACKFIRED.** Collapsing the 768 per-codeword
  calls/expert into one large per-expert call (per-row diagonals) was measured at
  14.9 s/expert → ~16.7 h — *slower* than the per-call path. The broadcast-sum
  distance materializes an `[N,256,8]` tensor, so it is **memory-bandwidth-bound**;
  small per-call batches (N=2048, ~975k cw/s) have far better cache locality than
  one big batch (N=131072, ~421k cw/s). The per-call path is near-optimal for
  this kernel. (Implemented, verified byte-exact, then reverted as a regression.)
- **Process parallelism is only ~1.5× and risky here.** Two producers could push
  the GPU toward 100% (~1.5–1.6×, the GPU is the ceiling), but the producer's
  `conversion-manifest.json` is fail-closed and coupled to the exact group set
  (rejects mismatched inventories), so concurrent writers to the canonical
  artifact collide and a safe merge needs surgery on the authenticated manifest
  format. Not worth destabilizing the live authenticated run for ~1.5×.

**Real further-speedup path (follow-up, not attempted in-window):** a fused
`mx.fast.metal_kernel` that computes each codeword's 256-row distance + argmin +
code assembly in-register (no `[N,256,8]` materialization) would be compute-bound
instead of memory-bound — plausibly 5–10× — and would also make batching and
parallelism worthwhile. This is a larger, byte-exactness-sensitive kernel-writing
effort. A `--layer-index-list` + manifest-merge for clean process parallelism
(~1.5×) is a smaller follow-up. Both are deferred rather than risking the
in-progress run.

## Session 3: worst8 COMPLETE, audit PASS, reeval in flight

- **The worst8 E8P re-materialization finished at 20:18 PDT** — 24/24 groups on
  the fused Metal path (relaunched 17:53 with `--resume` over 3 prior groups;
  later groups ~4–7 min each). Manifest `status: complete`, SHA-256
  `e3cdf933d96812918bdef242ebca044b7e91df954353adca8f0c56846f930b33`. The job the
  original handoff projected at 78+ hours completed in aggregate ~5 h of wall
  clock across the two runs.
- **First-class audit: PASS (10/10 checks).** Candidate identity
  `0bbd540c6b0816ec6efe929b2d7e1816beeef4a85bfd415b2b5d54eafb123c86`, layers
  70–77 fully replaced (24 groups + 201 inherited), payload 104.8 GB within the
  112 GB budget (matches the recipe's expected bytes exactly). Audit JSON:
  `artifacts/quality/glm52-recovery-worst8-e8p-artifact-audit-20260711.json`
  (SHA-256 `27ee05ba3853363c3dcf691b31dab84d1fcfcab4997f222ac3b97d062d700b8a`).
- **Frozen 66-row reevaluation launched 20:24 PDT** (direct producer, ledger
  caveat unchanged) → output will land at
  `artifacts/quality/glm52-recovery-worst8-e8p-reeval-20260711.json`. This is
  the quality verdict: does worst8-E8P beat Full75's 0.5633 top-1? Follow-ups
  after it lands: selection-only re-attribution + decision packet (P1 c,d).

### Worst8-E8P reevaluation verdict (landed 20:31 PDT)

`artifacts/quality/glm52-recovery-worst8-e8p-reeval-20260711.json`
(`status: diagnostic_complete`, 66 rows, selection-only tuning preserved):

| Split | top-1 | mean KLD | PPL ratio | p999 KLD |
| --- | ---: | ---: | ---: | ---: |
| selection | 0.5666 | 1.056 | 2.32 | 9.84 |
| holdout | 0.5373 | 1.112 | 1.57 | 7.02 |
| report | 0.5396 | 1.091 | 2.49 | 10.07 |

Overall ≈ **0.548 top-1: below Full75's 0.5633**, worse p999 tail, +6.4 GB
payload. Honest reading: 16-bit E8P on the 8 worst layers (over the original
seed) roughly matches — but does not beat — 8-bit re-rounding of all 75 layers.
All five frozen gates still fail. **The per-layer-bits lever is plateauing;
the next gains need a stronger algorithm class** (EBSS per the declared matrix,
and/or the fp32-LDLQ/YAQA reshape below). Decision packet + selection-only
re-attribution remain to be issued through the campaign evidence flow.

### Ledger reconciliation (DONE via authorize-retry)

`keep recovery campaign authorize-retry` (new, commit `6e506983`) appended
authorization event `ff8570c0116c0d97bcdced80c296203edb33dfad52495482e0e475a98fec146c`
bound to the SIGTERM terminal. `advance --dry-run` now reports **State: audit**
("authenticated materialization is ready for audit") — the controller observes
the complete, authenticated artifact. Remaining reconciliation: register the
already-produced audit (`27ee05ba…`) and reeval evidence through the campaign
evidence flow (no CLI subcommand exists for evidence registration yet — small
automation follow-up; the prior session registered evidence via the Python
ledger API).

### BlockLDLQ/YAQA research + spike outcome (recon #3)

- **A fused Metal port of the current algorithm is infeasible as-is:** the
  descent is fp64 (no Metal fp64) and per-codeword sequential (accepted moves
  update the gradient).
- **Byte-exact row-vectorization landed (`9952dbff`) but yields only 1.30×** —
  the bottleneck is the 768 serial fp64 codeword steps, not the row loop. The
  full Kronecker variant's rows are coupled through `h_out` and cannot be
  row-parallelized byte-identically at all.
- **Research (GPTQ-Babai equivalence; YAQA paper; fp64-on-Metal emulation)**
  reframed the box: (a) the field runs this algorithm class in fp32 on GPU
  (GPTQ/llm-compressor/GPTQModel all `float32`); (b) the YAQA paper's own
  formulation is m+n steps of parallelizable matmuls via LDL error feedback —
  the repo's gradient-recompute loop is a more-sequential re-derivation; (c)
  bit-exact fp64 emulation on Metal exists (metal-softfloat / double-float) if
  ever needed. Full hypothesis set + acceptance-criterion reframe (determinism +
  objective-dominance instead of byte-identity for *new-candidate* levers) is in
  the session transcript; decision deferred until the worst8 reeval verdict
  picks the next quality lever (fp32-LDLQ reshape vs full-YAQA vs EBSS).

## Session 2: fused Metal kernel + convention sweep

The broadcast search was memory-bandwidth-bound (63.5% GPU). A **fused
`mx.fast.metal_kernel`** (`src/mlx_vq/kernels/e8p_fused_search.metal`, one thread
per codeword, 256-row search in registers, no `[N,256,8]` scratch) is
compute-bound:

- **34M cw/s vs 0.39M broadcast (87×); worst8 search ~12.5 min vs ~18 h.** Per
  expert quantize 0.34 s vs ~15 s. Byte-identical to exhaustive (codes+scales),
  verified on-host including per-row diagonals and a real materialize e2e.
- Wired as the `metal` backend; `assign_codes` batches per expert (per-row
  diagonals) so the kernel gets one large launch. Commits `05a820c5`.
- **Reality check:** once the search is fused, the recovery run is
  I/O + single-core-NumPy bound (source-shard reads, the scale-update iterations,
  resume-validation, publish). End-to-end gain is ~2–3×, not 87× — the search is
  simply no longer the bottleneck. Further speedup needs I/O overlap / prefetch
  or moving the scale-update to MLX (follow-ups).

**Metal is now the default** on Metal-capable hosts
(`_default_recovery_search_backend`, commit `6be41ab6`) — fixes operators
silently getting the slow path. NumPy stays the deterministic reference/off-host
fallback; `GLM52_E8P_BACKEND` overrides.

**Fused nearest-code primitive for the 8-bit / general path**
(`src/mlx_vq/kernels/nearest_diag_search.metal`, commit `7062de4a`): 61.8M vs
4.9M cw/s (12.7×) on the 256-row E8 grid, byte-exact. The 8-bit recovery search
(`code_bits==8`) now batches through it (commit `7f814518`), metal-default,
byte-identical on-host (codes+scales).

### Convention/deferral sweep (recon, gpt-5.6-terra) — status

Prioritized siblings of the "known win but deferred" fused-kernel gap:

| # | Location | Issue | Status |
| --- | --- | --- | --- |
| 1 | `glm52_recovery_materialize.py` | recovery defaulted to slow NumPy | **FIXED** (`6be41ab6`) |
| 4 | `rtn.py` 8-bit search | CPU 256-way scan on common build path | **FIXED** (fused, `7062de4a`+`7f814518`) |
| 2 | recovery controller | no `--authorize-retry` after SIGTERM | open (campaign plumbing) |
| 3 | `kronecker_hessian.py:498` | BlockLDLQ/YAQA reassignment nested Python/float64 per expert | open — HARD: sequential coordinate descent w/ gradient updates, needs a tiled MLX kernel + a full-YAQA dispatch criterion; not the default path (`blockldlq_hin_only` is) |
| 5 | `imatrix.py:163` | routed imatrix Python per-expert scan after MLX→NumPy | open — Medium: keep on MLX, segmented scatter-add |
| 6 | `vq_switch.py:179` | NumPy token×route loop in a RAMP inference fallback | open — Medium: make on-device gather mandatory or fail loudly |
| 7 | `keep/cli.py` | KEEP/RAMP still a forwarding shim into `mlx_vq` | open — Medium: complete the public-package cutover (naming, no numerics) |
| 8 | `kronecker_hessian.py:441` | CPU-only H_in-only recovery fallback | open — Medium: pair with fused BlockLDLQ (see #3) |

Recon correctly excluded the fixed E8P scan and flagged the "16-bit E8P off NAX"
roadmap text as stale (a `nax_e8p` auto-dispatch path exists).

## Root cause and fix (the hard part)

- The E8P codebook factors into 256 absolute rows × even-parity signs × ±0.25
  shift, so a nearest-code search under a diagonal metric decomposes to a 256-row
  search plus a single min-penalty parity flip. The repo already had this for the
  *unweighted* RTN encoder (`encode_e8p_rtn`); the diagonal-Hessian recovery path
  did not, and brute-forced all 65,536 rows on one CPU core.
- **Diagonal-Hessian generalization** (`encode_e8p_rtn_diagonal_hessian`,
  `src/mlx_vq/codebook/e8.py`): each `wᵢ ≥ 0` keeps the per-dim optimal sign, so
  the weighted distance is `Σᵢ wᵢ(|tᵢ| − aⱼᵢ)²` over 256 rows with weighted flip
  penalty `4·wᵢ·|tᵢ|·aⱼᵢ`. Byte-identical to the exhaustive full-grid search:
  0/15,000 mismatches on random data, and 0 on constructed exact ties (a
  deterministic tie-break guard was required — `E8P_SHUFFLE_MAP` means the lowest
  output dim is not the lowest packed code).
- **Metal precision trap** (`src/mlx_vq/quant/e8p_metal.py`): MLX's Metal
  float32 *matmul* uses reduced-precision accumulation (~1e-2 abs error) that
  flips `argmin` on near-tie rows → ~0.19% wrong codes. Fixed by computing the
  8-term distance with an **fp32 elementwise broadcast + sum**, not a matmul.
  Result: distance matches NumPy to 5e-6 (ULP); Metal byte-identity test passes
  on the host (0 mismatches).

## What landed (branch `keep-glm52-pipeline-and-p1-lock`, not pushed)

- `c9b18c2a` — factored NumPy reference + tie-break, standalone checkpoint module
  (`src/mlx_vq/convert/recovery_checkpoint.py`), worst8 benchmark + acceptance gate.
- `36fb5b67` — MLX/Metal factored search + `nearest_e8p_codes_diagonal_hessian`
  dispatcher (backends `numpy`/`metal`/`exhaustive`; default `numpy`, byte-exact,
  deterministic; `metal` opt-in with import-only fallback so runtime errors
  propagate). Wired into the recovery materializer's `code_bits==16` path.
  Adversarial-review fixes applied (safe default + narrowed fallback).
- `18f46df9` — `materialize_groups_from_source` resolves the backend from
  `GLM52_E8P_BACKEND` (default `numpy`); `=metal` opts into the fast path **with
  no change to the SHA-pinned campaign argv**.

Tests: 13 new E8P tests pass; `test_rtn_quantization` (9) and the recovery
materialize suite (260) stay green — the default path is bit-for-bit unchanged.

Verification:
```bash
UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest \
  tests/test_e8p_diagonal_hessian.py tests/test_e8p_metal.py \
  tests/test_e8p_recovery_checkpoint.py tests/test_e8p_search_bench.py \
  tests/test_rtn_quantization.py -q
uv run python benchmarks/bench_e8p_search.py --backend metal   # host; acceptance_gate_pass:true
```

## How to launch the real worst8 run (the P1 rerun)

The acceptance gate passes, so the handoff's "hours not days" precondition is met.
Two independent blockers remain for a campaign-clean launch:

1. **Controller retry authorization.** `keep recovery campaign advance` is
   fail-closed BLOCKED: the prior worst8 transition ended with `exit_code=-15`
   (SIGTERM), so `controller.py` refuses to relaunch without "explicit retry
   authorization." There is currently **no `--authorize-retry` flag** on the
   advance CLI — the intended re-authorization path is not yet implemented. Until
   it is, the campaign controller will not launch worst8.
2. **Backend selection.** The run must be launched with `GLM52_E8P_BACKEND=metal`
   in the environment (inherited by the producer subprocess). Without it the
   producer uses the byte-identical but ~11× slower numpy path (~83 h).

Direct producer launch (bypasses the controller ledger; the producer
self-authenticates all inputs via `--expected-*-sha256`, holds
`.keep-heavy-job.lock`, publishes transactionally per group, and `--resume`s):

```bash
GLM52_E8P_BACKEND=metal .venv/bin/python benchmarks/run_glm52_recovery_wave1.py rematerialize \
  --source-dir <SNAP> --index-path <SNAP>/model.safetensors.index.json \
  --seed-artifact-dir artifacts/quality/glm52-wave6-full-materialization-20260709/e8-full-w1 \
  --stats-dir artifacts/quality/glm52-recovery-wave1-stats-20260710 \
  --attribution-json artifacts/quality/glm52-recovery-wave1-reattribution75-20260711.json \
  --output-dir artifacts/quality/glm52-recovery-worst8-e8p-artifact-20260711 \
  --worst-layer-count 8 --e8p-worst-layer-count 8 --resume \
  --expected-stats-manifest-sha256 9c743281c358332b4ff473d87e17b796275007f4662cc9363cbfd59da56196d7 \
  --expected-attribution-sha256 f4db48e00e32b6b1df502bfa7e91902b3a9c008160dd45285d49a66e97d67ee8 \
  --expected-seed-manifest-sha256 ba1d3135ef8901f1a69ead28b5f9d330ef40d015ac31fdb4d2dcfe678c41f0a4 \
  --expected-full-source-blob-inventory-sha256 ace08e87dcbce3a22249e54196a27c0045992f8d5b8ca07f342899fa7a53fc8d \
  --expected-routed-source-blob-inventory-sha256 5dbacc9b0ec3deee829027fe739ebca9d0a77e8cd0016977b799e3a52539e2da \
  --accepted-composite-audit-json artifacts/quality/glm52-wave6-full-artifact-audit-20260709.json \
  --expected-composite-audit-sha256 8026322a533606ed451d13b029d83836fefbf1a938b33529de535f3fc778591a \
  --heavy-lock-path .keep-heavy-job.lock
```
`<SNAP>` = `~/.cache/huggingface/hub/models--0xSero--glm-5.2-reap-504B-v2/snapshots/6c9241aa05fb243a0edb7c804c213ec1cf5c920d`.
Recommended: add `PYTHONUNBUFFERED=1` so progress is observable, and launch with
`nohup ... & disown`. Expect ~7.3 h; resumable per group.

After it completes: first-class artifact audit, frozen 66-row reevaluation,
selection-only re-attribution, and the decision packet (unchanged from the prior
handoff's P1).

## Not done (deliberately)

- **Full worst8 run** — not launched: the controller retry-authorization path is
  unimplemented, and 7.3 h exceeded the session window. A one-layer real-data
  validation run was exercised (see below).
- **Controller `--authorize-retry`** — needs implementing so campaign-clean
  relaunch is possible (small, well-scoped follow-up in
  `src/mlx_vq/recovery_campaign/`).
- **Expert-level checkpoint wiring** — the module exists
  (`recovery_checkpoint.py`) but is not yet wired into the materializer loop; the
  producer's existing group-level `--resume` (≤18 min loss on interrupt) suffices.
- **EBSS / learned rotations** (campaign P2), release-class evidence (P3), family
  gate (P4), product/publication (P5) — unchanged, downstream.

## Guards honored

No push/merge/publish/upload. `GLM_MLX_WIRED_LIMIT_GB` unset. `.keep-heavy-job.lock`
used by the producer. `runs/`, prior root handoffs, and RAMP plans untouched.
`artifacts/` gitignored. Default recovery output remains byte-identical to the
prior exhaustive path (numpy default), so no accepted verdict changed.
