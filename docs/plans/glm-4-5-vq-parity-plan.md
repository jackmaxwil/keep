# GLM-4.5-Air VQ Routed-Expert Kernel Parity Plan

## Summary And Pass Gates

Goal remains full parity with MLX q2:
- **Decode `M=1`: primary target.** VQ must reach or beat fresh same-run MLX q2 on Air `gate_up` and `down` microbench rows, then on resident `decode_128`.
- **Prefill `M=1024,4096`:** VQ must be `<= 1.5x` fresh same-run MLX q2 on Air `gate_up` and `down`; true parity is stretch.

Execution uses `superpowers:subagent-driven-development`: one implementer subagent per milestone, then spec-review and code-quality-review subagents. Each loop records commands, target, best achieved, and next action in `WORK_LOG.md`; every durable fact goes in `DISCOVERY.md`.

Verified local fact:
- PyObjC Metal reports `Apple M5 Max` and `maxThreadgroupMemoryLength = 32768`, so the threadgroup-memory budget is exactly 32 KiB.

Public interface additions:
- Add benchmark-only decomposition modes to `benchmarks/bench_glm45_air_projection_kernels.py`: `vq_current`, `vq_decode_only`, `vq_matmul_only`, and `vq_decode_direct_candidates`.
- Add `--profile-components` to `benchmarks/bench_glm45_air_quant_compare.py` for both VQ and MLX q2 engines; extend the MLX q2 resident runner to emit the same `component_timings` / `component_summary` schema already used by VQ.
- Add `route_strategy: Literal["auto","direct","sorted_tiled"] = "auto"` to `gather_vqmm()` and `QuantizedVQSwitchLinear`; keep existing callers working.

## Phase N: NAX Prefill

`docs/PLAN_PREFILL_NAX.md` is the active successor for the prefill side of this plan. It supersedes further decoded-B simdgroup cwdecode tuning as the primary route to GLM-4.5-Air prefill parity.

NAX pass gates:
- N0 must capture the actual runtime MLX q2 kernel names before any q2 parity row is trusted. Local `mlx.metallib` inventory proves q2 NAX kernels exist at `bm64_bn64_bk64_wm2_wn2`, but MLX 0.31.2 predates the `gather_qmm_nax` dispatcher-name fix and may still request nonexistent `bk32` names. Use `benchmarks/audit_mlx_q2_nax_runtime.py` to emit the exact capture matrix and opt-in `MTL_CAPTURE_ENABLED=1` commands. Current Air gate/up and down target shapes are cleared by LLDB logs and command-line xtrace shader-list exports: they request/export `affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2`, with no `bk32`.
- N1/N2 add benchmark-only `mlx_gather_mm_bf16` and `vq_decode_to_scratch_gather_mm` probes. These are diagnostic routes for proving the NAX engine ceiling, not resident defaults.
- N3-N5 move to a C++ `mx::Primitive` path: predecoded FP16/BF16 NAX first, fused E8 decode in the B-fragment load second, resident integration only after projection rows reach `<=1.5x` sorted MLX q2 at `M=1024,2048,4096`.
- 2026-06-25 status: `nax_e8_fp16_sorted_steel` passes the N4 projection gate with all target rows `<=1.5x` sorted q2, and the public resident engine `vq_e1_routed` now defaults to `prefill_engine="auto"` so NAX is used when available with Metal fallback. Fresh default rows pass the original N5 gate: `prefill_1k` median `1.6776839164958801s` versus q2 median `1.1985174789988378s` (`1.40x`), `decode_128` median `4.428020296505565s`, and clean fixed-prompt quality rows for all five prompts.
- Publication phase: `docs/research/NAX_PARITY_RESULTS.md` is rendered from `artifacts/benchmarks/glm45-air-nax-publication.jsonl`, `artifacts/benchmarks/glm45-air-nax-publication-projections.jsonl`, `artifacts/benchmarks/glm45-air-nax-toolchain-audit.jsonl`, and `artifacts/benchmarks/glm45-air-nax-int8-raw-ceiling.jsonl`. The initial full publication matrix has clean 5/5 resident rows for all engines and scenarios, including clean `prefill_4k`. Pinned NAX is the steadier publication row at `1.44x` q2 for 1K and `1.48x` q2 for 4K; default auto records higher fresh-process variance and a clean median `1.52x` in the generated table.
- N6 INT8 is currently stopped by the raw ceiling gate. The 64x64x64 native E8 probe records prequantized INT8 at `0.97x` FP16 and INT8 with A8 quantization at `0.81x` FP16, below the `>=1.7x` proceed threshold. FP16-NAX remains the public default.
- The banked M=1 decode win remains protected. Any future `prefill_engine` flag must affect prefill only and must re-run `decode_128` after integration.
- Down `group_size=352` has its own N3.5 correctness fork before down performance tuning: either specialize the 352 tail or explicitly re-export down with an aligned group size such as 128.

Resident acceptance:
- Run VQ and MLX q2 in fresh same-run comparisons.
- Require at least 2 clean resident repetitions for accepted `decode_128` and `prefill_1k` rows.
- Any row with nonzero pageouts or swapouts is invalid and must be repeated.
- 4K MLX peak must stay `<= 1.05x` current clean VQ resident baseline.

## Step 0 Root-Cause And Feasibility Gate

Add diagnostics only; do not retile production kernels.

Decode-step resident profile:
```bash
uv run python benchmarks/bench_glm45_air_quant_compare.py --engine vq_e1_routed --scenario decode_128 --repetitions 1 --profile-components --append-jsonl artifacts/benchmarks/glm45-air-decode-component-profile.jsonl
uv run python benchmarks/bench_glm45_air_quant_compare.py --engine mlx_q2_routed_g128 --scenario decode_128 --repetitions 1 --profile-components --append-jsonl artifacts/benchmarks/glm45-air-decode-component-profile.jsonl
```

Decode bottleneck gate:
- Proceed with decode kernel work only if VQ-vs-MLX routed projection component deltas explain a material share of the `decode_128` tok/s gap.
- If attention, cache update, router, shared expert, sampling, or Python/dispatch overhead explains most of the gap, stop and revise toward that measured bottleneck before touching `vq_qmv`.

Projection decomposition:
```bash
uv run python benchmarks/bench_glm45_air_projection_kernels.py --projection down --tokens 1 --tokens 1024 --tokens 4096 --top-k 8 --experts 128 --iterations 5 --warmup 2 --decompose-vq --append-jsonl artifacts/benchmarks/glm45-air-vq-decomposition.jsonl
uv run python benchmarks/bench_glm45_air_projection_kernels.py --projection gate_up --tokens 1 --tokens 1024 --tokens 4096 --top-k 8 --experts 128 --iterations 5 --warmup 2 --decompose-vq --append-jsonl artifacts/benchmarks/glm45-air-vq-decomposition.jsonl
uv run pytest tests/test_gather_vqmm.py tests/test_switch_routing.py tests/test_vq_qmv.py -q
```

Diagnostic kernels:
- `vq_decode_only`: same current route/output/codeword decode loop, no activation multiply, checksum output to prevent dead-code elimination.
- `vq_matmul_only`: same current loop, scale/activation/reduction/output work, but cheap deterministic weight expression instead of codebook decode.
- `vq_decode_direct_candidates`: M=1-only direct kernels using constant-space or single-copy codebook, no duplication-8, lower barrier count, quad/simd reductions, and route/output-row packing candidates.
- `vq_current`: unchanged current path.

Feasibility gates:
- Decode floor: compute M=1 bytes read per Air projection and a measured bandwidth floor. If the VQ bandwidth floor is not faster than fresh MLX q2, decode parity is physically unlikely in this custom-op path and must be escalated or renegotiated before kernel days are spent.
- Decode attempt gate: if candidate kernels consume the milestone budget and best achieved remains above MLX q2, record best achieved vs target and stop for escalation rather than continuing into resident propagation.
- Prefill gate: if `vq_decode_only` grows with `M`, proceed with sorted tiled prefill; if not, revise.
- Prefill ceiling: if `vq_matmul_only` at `M=4096` is already `> 1.5x` MLX q2, prefill parity depends on simdgroup MMA. A non-MMA prefill milestone cannot claim parity.

## Kernel Design

Decode track, `M=1`:
- Optimize the direct `vq_qmv` / small-route path, not sorted tiled GEMM.
- Use constant-space or single-copy codebook staging for M=1; do not use register-loaded full codebooks or duplication-8.
- Candidate changes: fewer barriers, quad/simd reductions, multiple output rows or routes per threadgroup, and lower per-route dispatch/reduction overhead.
- Decode microbench gate: Air `gate_up M=1` and `down M=1` VQ `<=` fresh same-run MLX q2.

Prefill track, `M>=1024`:
- Flatten routes; build `flat_rhs`, `flat_lhs`, `order = mx.argsort(flat_rhs)`, `sorted_rhs`, `sorted_lhs`, and `inverse = mx.argsort(order)` on-device.
- Do **not** materialize sorted activations. Pass `sorted_lhs` into the kernel and gather `x[sorted_lhs]` into threadgroup memory inside the tiled kernel.
- Output `[routes, output_dims]`; scatter back with `inverse` and reshape to `[tokens, top_k, output_dims]`.
- Normal route tiles decode one expert weight tile once and reuse it across the tile. Mixed boundary tiles use direct fallback by default; switch to padding expert blocks to `M_TILE` if measured mixed-tile work exceeds 8%.

Initial prefill tiles:
- Down first: `K=1408`, `N=4096`, `group=352`, `K_TILE=352`, `N_TILE=8`, `M_TILE=16`.
- Gate/up second: `K=4096`, `N=1408`, `group=512`, `K_TILE=256`, `N_TILE=8`, `M_TILE=16`.
- Gate/up scale index is `scale_idx = k_global / 512`; the same scale spans two consecutive K tiles and must not be re-based per tile.

Threadgroup-memory budget:
- Codebook duplication 8: `8192 B`.
- Down: decoded weights `5632 B`, activations `11264 B`, scratch about `1024 B`, total about `26112 B`.
- Gate/up: decoded weights `4096 B`, activations `8192 B`, scratch about `1024 B`, total about `21504 B`.
- Both fit verified `32768 B`; tune `M_TILE=8/16` and duplication `4/8` if occupancy suffers.

Simdgroup requirement:
- Prefill parity requires simdgroup MMA. If `simdgroup_matrix` / `simdgroup_multiply_accumulate` cannot be compiled and run through `mx.fast.metal_kernel`, prefill parity is not reachable in this custom-op plan; escalate to the deferred C++/MLX-GEMM path or rescope prefill.

## Dependency-Ordered Milestones

| Milestone | Depends On | Deliverable | Gate | Effort | Risk |
|---|---:|---|---|---:|---|
| 0. Decode/Pipeline Diagnosis | none | Component profile for VQ and MLX q2 `decode_128`; projection decomposition rows; bandwidth-floor analysis | Confirms routed VQ projection is the decode bottleneck and floor is winnable | 1-2d attempt | High |
| 1. Capability Probes | 0 | Verify simdgroup compile/run, dtype support, Metal 4 custom-kernel access; record 32 KiB fact | If simdgroup unavailable, prefill parity escalates/rescopes | 0.5d attempt | Medium |
| 2. Decode Direct Track | 0-1 | M=1 `vq_qmv` / direct-route candidate kernel(s) | `gate_up/down M=1 <= MLX q2`; record best achieved each loop | 2-4d attempt | High |
| 3. Sorted Compaction Contracts | 0 | `route_strategy`, on-device sort/inverse flow, in-kernel activation indexing contract | sorted/direct/scatter equality; no activation materialization | 1d attempt | Medium |
| 4. Down Tiled Correctness | 1,3 | Down sorted-tiled kernel with fp32 accum, scalar or provisional inner loop | cosine `>=0.99999`; fallback/padding fraction recorded; no parity claim | 1-2d attempt | High |
| 5. Down Simdgroup Prefill | 4 | Down tiled kernel with simdgroup MMA and tile sweep | down `M=1024,4096 <= 1.5x MLX q2`; memory/pageouts clean | 2-4d attempt | High |
| 6. Gate/Up Simdgroup Transfer | 5 | Gate/up sorted-tiled simdgroup path with half-group scale handling | gate/up `M=1024,4096 <= 1.5x MLX q2` | 1-3d attempt | Medium |
| 7. Auto Dispatch Integration | 2,5,6 | `auto`: decode direct for small route counts, sorted-tiled for prefill | all microbench regime gates pass in same-run MLX q2 comparison | 1d attempt | Medium |
| 8. Resident Propagation | 7 | Resident comparison and quality runs | 2 clean repetitions each for `decode_128` and `prefill_1k`; no dense params; pageouts/swapouts `0/0` | 1-2d attempt | High |

## Tests And Open Probes

Focused tests:
- `tests/test_vq_qmv.py`: M=1 candidate kernels match dense-dequant reference, cosine `>=0.99999`.
- `tests/test_gather_vqmm.py`: `direct`, `sorted_tiled`, mixed boundary tiles, padded expert blocks, explicit `lhs_indices`, scatter-back equality.
- `tests/test_switch_routing.py`: `route_strategy="auto"` uses decode direct below threshold and sorted-tiled at/above threshold.
- Benchmark tests: MLX q2 and VQ resident comparison rows accept `--profile-components` and emit matching component schemas.

Open probes before production kernel coding:
- Decode component attribution: confirm routed VQ projections explain enough of `decode_128` loss to make kernel work meaningful.
- Decode bandwidth floor: confirm VQ’s M=1 lower bound beats fresh same-run MLX q2 before spending the decode milestone budget.
- `simdgroup_matrix` / `simdgroup_multiply_accumulate` compile and run through `mx.fast.metal_kernel`, with half/bfloat16 inputs and fp32 accumulation.
- Metal 4 cooperative tensor/matmul callable path from MLX custom kernels; optional only.
- Keep `artifacts/glm-4.5-air-vq` untouched; write only new benchmark/quality rows and future variants to new paths.
