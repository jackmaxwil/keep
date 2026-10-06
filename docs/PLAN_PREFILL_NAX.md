# GLM-4.5-Air NAX Prefill Parity Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task by task. Keep the Ralph loop in `WORK_LOG.md` and durable facts in `DISCOVERY.md`.

**Goal:** Move GLM-4.5-Air VQ routed-expert prefill onto Apple M5 NAX / Metal 4 TensorOps through a C++ `mx::Primitive`, reaching `<=1.5x` resident-style sorted MLX q2 at `M=1024,2048,4096`, then lock publication-grade evidence and attempt INT8 only behind a hard raw-ceiling gate.

**Architecture:** Treat q2 as the engine reference and VQ as a data-path problem. First audit the local MLX q2 runtime names, then use pure MLX `gather_mm` and decode-to-scratch probes to confirm the NAX engine ceiling, then build an out-of-tree FP16 NAX primitive that eventually decodes E8 codes inside the B-fragment load.

**Tech Stack:** Python 3.14 via `uv`, MLX 0.31.2, mlx-lm 0.31.3, Apple M5 Max (`applegpu_g17s`), Metal 4 / MPP TensorOps, CMake 3.27+, nanobind 2.12+, Objective-C++ C++17.

---

## Current State And Invariants

- `docs/PLAN.md` is intentionally tracked-deleted in the current docs migration. The active parity plan is `docs/plans/glm-4-5-vq-parity-plan.md`; do not resurrect `docs/PLAN.md`.
- Existing resident q2 engine remains `mlx_q2_routed_g128`.
- Existing M=1 row-pair/direct decode is banked and must not regress. Any `prefill_engine` flag affects prefill only.
- Baseline comparisons must use resident-style sorted q2: `_gather_sort(..., sorted_indices=True)`.
- Every accepted benchmark row requires `pageouts_delta == 0`, `swapouts_delta == 0`, and at least two clean resident repetitions for headline numbers.
- Correctness gate precedes performance: projection cosine `>=0.99999` vs dequant-then-dense reference. Engine-swap probes also compare against prior VQ output to catch route/scatter mistakes.
- The VQ storage contract stays stable unless Phase N3.5 explicitly chooses a down-projection group-size re-export fork: codes `[E,out,in/8]`, scales `[E,out,in/group]`, and one global E8 codebook.

## Public Interfaces And File Changes

- Add benchmark-only projection variants in `benchmarks/bench_glm45_air_projection_kernels.py`:
  - `mlx_gather_mm_bf16`
  - `vq_decode_to_scratch_gather_mm`
  - `nax_predecoded_fp16`
  - `nax_e8_fp16`
  - `nax_e8_fp16_steel`
  - `nax_e8_fp16_sorted_steel`
- Add an N0 audit helper in `src/mlx_vq/benchmark/nax_audit.py` that reports NAX kernel inventory from the installed `mlx.metallib`.
- Add `benchmarks/audit_mlx_q2_nax_runtime.py` to emit the exact q2 capture matrix and, when explicitly requested, run sorted q2 under `mx.metal.start_capture()` using direct packed q2 tensors rather than dense pre-quantized weights.
- Add `benchmarks/audit_mlx_q2_nax_lldb.py` as the reproducible LLDB fallback when xtrace is unavailable. It logs `newFunctionWithName:` requests for the same q2 capture matrix and writes JSONL summaries plus full logs.
- Add resident comparator engines only when they are runnable without hidden dense routed-expert materialization:
  - future `mlx_gather_mm_bf16_probe`
  - future `vq_decode_to_scratch_gather_mm_probe`
  - `vq_e1_routed_nax_e8`
- Add runtime flag; normal resident loading defaults to `auto` after N5, while named comparison engines can still pin `vq_metal` or `nax_e8`:
  `prefill_engine: Literal["auto","vq_metal","nax_e8"]`.
- Add native extension under `native/vq_nax_ext/`, with wrapper in `src/mlx_vq/kernels/nax.py`, built with `python -m mlx --cmake-dir`, `nanobind>=2.12`, and `cmake>=3.27`.
- Add publication tooling:
  - `benchmarks/bench_glm45_air_quant_compare.py --publication-matrix`
  - `benchmarks/bench_glm45_air_projection_kernels.py --publication-matrix`
  - `benchmarks/audit_nax_toolchain.py`
  - `benchmarks/bench_nax_int8_raw_ceiling.py`
  - `benchmarks/render_nax_publication_results.py`
  - `docs/research/NAX_PARITY_RESULTS.md`

## Phase Plan

| Phase | Objective And Smallest Slice | Validation Gate | Evidence And Commands | GO/STOP |
| --- | --- | --- | --- | --- |
| N0 Git Hygiene And q2 Runtime Audit, 0.5-1d | Create a branch or commit before N-phase work, then audit actual q2 runtime kernels for prefill shapes. Confirm whether sorted q2 launches `gather_qmm_rhs_nax` or `gather_qmm_t_nax`, and whether any lookup requests nonexistent `bk32`. | Baseline branch or commit exists; q2 kernel capture identifies actual names for gate/up and down at `M=1024,2048,4096`; no trusted parity math until this passes. | `git status --short`; `uv run python - <<'PY' ... mlx_metallib_nax_inventory() ... PY`; `uv run python benchmarks/audit_mlx_q2_nax_runtime.py --projection all --tokens 1024 --tokens 2048 --tokens 4096`; LLDB fallback via `benchmarks/audit_mlx_q2_nax_lldb.py`; xtrace capture plus `--summarize-trace` shader-list export when Xcode `xctrace` is available. | GO if q2 is proven on existing `bk64` NAX kernels. STOP and patch/upgrade MLX if dispatcher requests `bk32` or falls off NAX unexpectedly. |
| N1 Pure MLX `gather_mm` Engine Probe, 0.5-1d | Add and run `mlx_gather_mm_bf16` using the same sorted route metadata as q2. | `mx.gather_mm` bf16 `<=2.0x` sorted q2 while current VQ remains `3-5x`; cosine `>=0.99999` vs dense/reference. | `uv run python benchmarks/bench_glm45_air_projection_kernels.py --projection all --variant nax_audit --tokens 1024 --tokens 2048 --tokens 4096 --iterations 3 --warmup 1 --append-jsonl artifacts/benchmarks/glm45-air-nax-gather-mm-audit.jsonl`. | GO if the engine diagnosis holds. STOP if `gather_mm` is also slow. |
| N2 Decode-To-Scratch Probe, 1d | Add benchmark-only active-expert scratch: decode active VQ experts to bf16/fp16 scratch, remap sorted expert ids, call `mx.gather_mm`, scatter back. | Correctness cosine `>=0.99999` vs dequant-then-dense and prior VQ. Resident `prefill_1k <=1.5x` q2 makes it a candidate interim; `1.5-2.5x` proceeds to fused. | `uv run pytest tests/test_glm45_air_projection_kernels.py tests/test_gather_vqmm.py -q`; append `artifacts/benchmarks/glm45-air-decode-scratch-gather-mm.jsonl`. | GO to N3 unless N1/N2 disprove NAX benefit. Do not ship if scratch creates memory pressure or misses `<=1.5x`. |
| N3 FP16-NAX Primitive Scaffold, 2-4d | On the Mac build host, verify macOS `>=26.2`, Xcode/Metal 4 SDK, `cmake>=3.27`, and `nanobind>=2.12`. Build `native/vq_nax_ext` and a predecoded fp16/bf16 sorted-gather primitive that calls FP16 `matmul2d`. | Build smoke passes on the Mac; predecoded primitive `<=1.6x` sorted q2 for gate/up and down at target M; cosine `>=0.99999`; memory pressure clean. | `uv run python -m mlx --cmake-dir`; `uv run python -m nanobind --cmake_dir`; `cmake --version`; append `artifacts/benchmarks/glm45-air-nax-predecoded-fp16.jsonl`. | GO if real NAX primitive access is proven. STOP and read cider/local MLX primitive wiring if still `>2x`. |
| N3.5 Down-352 Correctness Fork, 1-2d | Decide down `group_size=352`: specialize the 352 K-tail or re-export down with an aligned group size such as 128. | Either specialized down-352 tail path passes correctness, or the aligned-group re-export dry-run is chosen and documented. | Tail-specific tests for down `1408->4096, gs=352`; optional conversion dry-run for down `group_size=128`; append decision to `DISCOVERY.md`. | GO only after the down fork is chosen and correctness-passed. |
| N4 Fused E8 FP16-NAX, 1-2w | Replace B-fragment fp16 load with inline E8 decode: `w = (codebook[code] - 8) * 0.5 * scale`. Use direct device reads; no fp16 scratch and no weight threadgroup staging. | Gate/up and down projection rows `<=1.5x` sorted q2 for `M=1024,2048,4096`; cosine `>=0.99999`; clean memory metrics. | Add focused tests for gate/up `4096->1408, gs=512` and chosen down path; append `artifacts/benchmarks/glm45-air-nax-e8-fused-fp16.jsonl`. | GO to resident integration when all projection gates pass. If only down fails, return to N3.5. |
| N5 Resident Integration And Tile Sweep, 3-5d | Wire `prefill_engine="nax_e8"` into shared-sort GLU path. Sweep `BM/BN/BK`, route tile, small/large-M variants, and swizzle. Use q2 references: gather q2 is `64^3/wm2/wn2`; dense fused is `128/128/512`. | Two clean resident `prefill_1k` repetitions: VQ `<=1.5x` `mlx_q2_routed_g128`; no dense routed params; `decode_128` still passes or matches the banked win. | `uv run python benchmarks/bench_glm45_air_quant_compare.py --engine vq_e1_routed --scenario decode_128 --repetitions 2 --append-jsonl artifacts/benchmarks/glm45-air-nax-resident.jsonl`; same for `prefill_1k`; same q2 rows; focused tests and `git diff --check`. | GO to default only after resident prefill passes and decode is protected. |
| N5.5 Publication Benchmarks, 0.5-1d | Lock publication-grade resident and projection evidence before further tuning. Run default auto, pinned NAX, pinned Metal, and audited sorted q2 across `decode_128`, `prefill_1k`, and `prefill_4k` with fresh processes. | Every headline row has `>=5` clean reps and reports median/min/p90/stddev; 4K is published only if clean, otherwise the failed attempts and memory reason are documented. | `uv run python benchmarks/audit_nax_toolchain.py --try-xcode-install --append-jsonl artifacts/benchmarks/glm45-air-nax-toolchain-audit.jsonl`; `uv run python benchmarks/bench_glm45_air_quant_compare.py --publication-matrix --repetitions 5 --append-jsonl artifacts/benchmarks/glm45-air-nax-publication.jsonl`; projection matrix; render `docs/research/NAX_PARITY_RESULTS.md` from JSONL. | GO if clean 4K exists or its absence is explained. Publication rows do not change the default. |
| N6 Optional INT8-NAX, gated 1w | Beat q2 only if a raw 64x64x64 INT8 TensorOps ceiling is worth pursuing. Weight mapping is exact: `q_w=codebook[code]-8`, `s_w=0.5*scale`; activation A8 and rescale overhead are the risks. | Raw native INT8 prequantized speedup must be `>=1.7x` FP16 before resident optimization. If it passes, INT8 must beat FP16-NAX on gate/up and down at `M=1024,2048,4096`, be `<1.0x` sorted q2 on at least gate/up, and show one-layer PPL degradation `<1%` vs FP16-NAX VQ. | `uv run python benchmarks/bench_nax_int8_raw_ceiling.py --append-jsonl artifacts/benchmarks/glm45-air-nax-int8-raw-ceiling.jsonl`; then optional artifacts `artifacts/quality/glm45-air-nax-int8-one-layer.jsonl` and `artifacts/benchmarks/glm45-air-nax-int8.jsonl`. | STOP INT8 if raw ceiling is not `>=1.7x`; keep FP16-NAX as default unless speed and quality gates both pass. |

## Execution Notes

- 2026-06-25: N4 FP16 fused E8 first became correct with codeword-staged B tiles, not scalar direct B-fragment decode. That path was useful as scaffolding but stalled around `2x` sorted q2.
- 2026-06-25: The original no-threadgroup-staging instinct is falsified for E8 FP16 on this shape. Staging is valuable because one E8 codeword expands eight K values; scalar cooperative-fragment decode reloads/extracts too often.
- 2026-06-25: A decoded E8 table in threadgroup is useful only when amortized by long-K gate/up. Keep it conditional on `K >= 4096`; unconditional table setup regresses down `K=1408`, and per-block scale staging was measured slower.
- 2026-06-25: N4 passes with `nax_e8_fp16_sorted_steel`, which sorts/gathers activations into contiguous route order and uses MLX Steel `NAXTile` / `tile_matmad_nax` for the TensorOps loop. Paired hybrid projection ratios are gate/up `1.42x/1.32x/1.26x` and down `1.42x/1.30x/1.38x` at `M=1024/2048/4096`, with pageouts/swapouts `0/0`.
- 2026-06-25: N5 is wired into resident loading through `prefill_engine="auto"`. Auto selects NAX only when the native extension is available and the projections are 8-bit/8D-aligned; otherwise it falls back to the Metal VQ path. The main `vq_e1_routed` comparison engine follows this default; `vq_e1_routed_vq_metal` and `vq_e1_routed_nax_e8` provide explicit pins.
- 2026-06-25: Resident evidence passes the default gate on this host. The public default engine `vq_e1_routed` now uses `prefill_engine=auto`; two clean `prefill_1k` repetitions have median `1.6776839164958801s` versus q2 median `1.1985174789988378s` (`1.40x`), and two clean `decode_128` repetitions have median `4.428020296505565s`. All accepted rows have pageouts/swapouts `0/0`.
- 2026-06-25: Full default-auto quality matrix is clean. The public default engine generated the expected fixed-prompt outputs for all five prompts, all accepted rows have pageouts/swapouts `0/0`, finite logits, no dense routed params, and no unbound VQ experts.
- 2026-06-25: First N6 INT8 prototype is correct against a quantized-A reference but not fast enough. Full gate/up M=1024 is `10.36ms`, and prequantized native-only is about `9.83ms`, slower than staged FP16 native-only at about `8.97ms`.
- 2026-06-25: N0 is cleared for Air target prefill shapes by CLT LLDB runtime-name logs. Gate/up and down at `M=1024,2048,4096` all request `affine_gather_qmm_rhs_nax_nt_float_gs_128_b_2_bm_64_bn_64_bk_64_wm_2_wn_2`; no `bk32` request observed. Re-run N0 after MLX/Metal changes or for new q2 shapes.
- 2026-06-25: Native runtime compilation can now expand installed MLX Metal headers. `nax_fp16_matmul.metal` includes `mlx/backend/metal/kernels/steel/gemm/nax.h`; the accepted sorted-A path uses that header directly instead of vendoring Steel.
- 2026-06-25: Publication matrix landed with clean 5/5 resident rows for default auto, pinned NAX, pinned Metal, and q2 across `decode_128`, `prefill_1k`, and `prefill_4k` in the initial full matrix. Pinned NAX is the steadier headline path at `1.44x` q2 for 1K and `1.48x` q2 for 4K; default auto has higher fresh-process variance and recorded `1.52x` clean medians in the report. A targeted default-auto 4K rerun was rejected due pageouts and is ledgered as dirty evidence, not a headline row.
- 2026-06-25: N6 raw INT8 ceiling stops current INT8 optimization. On the 64x64x64 native E8 tile, prequantized INT8 is `0.97x` FP16 and INT8 including A8 quantization is `0.81x` FP16, below the `>=1.7x` proceed gate.
- 2026-06-25: Xcode 26.4.1 is installed and selected at `/Users/jack.mazac/Applications/Xcode-26.4.1.app/Contents/Developer`; the license is accepted; MetalToolchain `17E188` is installed; `xcrun xctrace version`, `xcrun metal --help`, and `xcrun --find metallib` work. The toolchain audit also passes a tiny static `metal -c` plus `metallib` compile/link smoke. The separately installed Metal Shader Converter 4.0 beta 1 provides `/usr/local/bin/metal-shaderconverter` (`metal-irconverter version: 4.0.0`) but remains separate from the Xcode `metal`/`metallib` compiler tools.

## Parallel Quality Track

- Run the cheap kill experiment first: single-layer least-squares scale fit to Q8, frozen E8 codebook, output cosine vs Q8 on calibration activations.
- If viable, implement PV-tuning separately: P-step updates per-group scales, V-step reassigns codes, KL distill from Q8, and expert-balanced sampling for cold experts.
- Treat sub-2-bit fixed-lattice sparse MoE as research risk. Quality recovery cannot excuse failed prefill gates.

## Risk And De-Risking Ledger

- `gather_qmm_nax` PR #3632: local metallib has `bk64` NAX kernels and no `bk32` NAX gather kernels; LLDB runtime logs and command-line xtrace shader-list exports prove the current Air target shapes request `bk64` RHS NAX. Re-audit after MLX/Metal updates or new q2 shapes.
- Sorted q2 path ambiguity: current target shapes fire `gather_qmm_rhs_nax`, not general `gather_qmm_t_nax`; this defines the real target for N4/N5.
- Down `group_size=352`: explicit N3.5 fork prevents hidden tail bugs.
- Mac build host: extension prerequisites must be verified on the real Mac, not inferred from a sandbox.
- Xcode tooling: `xctrace` works and corroborates the q2 NAX runtime path; static `metal`/`metallib` now pass through MetalToolchain `17E188`, including a tiny compile/link smoke. Metal Shader Converter is installed but is a separate tool, not a substitute for Xcode `metal`/`metallib`.
- Publication variance: distinguish clean headline rows from cold/runtime-cache variance and rejected memory-pressure reruns. The JSONL is the evidence authority; the Markdown report renders the latest fully accepted clean summary per cell.
- Cooperative tensor uniform control flow: all lanes participate; mask stores, not TensorOps participation.
- Register spilling: prove predecoded FP16 first, then inline only B-fragment E8 decode.
- Pipeline-cache staleness: cache by source hash, projection, dtype, MLX version, and kernel directory.
- Small/cold expert underfill: record expert-count histograms and keep route-tile variants.
- Cider execution risk: clone and read cider's real `eval_gpu` and `.metal` inner loop locally before implementing the primitive.
