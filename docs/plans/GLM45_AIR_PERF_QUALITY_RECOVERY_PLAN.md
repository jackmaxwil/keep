# GLM-4.5-Air Performance Optimization And Quality Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize GLM-4.5-Air resident VQ inference performance and recover quantization quality while preserving the proven no-dense-routed-expert resident baseline.

**Architecture:** Keep the current GLM-4.5-Air resident adapter as the stable baseline, add instrumentation before changing runtime behavior, then optimize routed MoE kernels and only expand context after 4K behavior is characterized. Quality work runs on a separate ladder: first record Air-only metrics, then use source-shard and artifact-oracle probes to test calibration, scale policy, activation-aware, and selective resident-safe precision variants.

**Tech Stack:** Python 3.14 through `uv`, MLX 0.31.2, mlx-lm 0.31.3, MLX custom Metal kernels through `mx.fast.metal_kernel`, safetensors byte-range source readers, Hugging Face cached GLM-4.5-Air shards, pytest.

---

## Scope

GLM-4.5-Air is the only active target for this plan.

GLM-5.2 remains parked. Do not resume GLM-5.2 resident VQ-only generation from this plan; it exceeded the 120 GB resident gate and needs a separate heavy-quantization or paging plan.

This plan is plan-only. It defines implementation slices, files, gates, risks, and fallbacks. It does not implement any code.

## Proven Baseline To Preserve

The current Air proof is the baseline, not a stepping stone to overwrite:

- Full resident bind loaded `645` non-expert params.
- Full resident bind bound `45` VQ sparse layers.
- Full resident bind skipped `17,280` dense raw routed expert tensors and `404` MTP tensors.
- No dense routed expert tensors exist in `model.parameters()`.
- Memory-only full bind MLX active/peak: `27,297,052,168` bytes.
- Memory-only full bind RSS: `30,554,669,056` bytes.
- Short smoke prompt `The capital of France is`, `max_new_tokens=8`, generated exactly `The capital of France is Paris. Paris is the capital of France`.
- Short smoke MLX peak: `27,368,638,640` bytes.
- Short smoke pageouts/swapouts: `0` / `0`.
- 4K context gate passed with finite logits and no observed pageouts/swapouts.
- 4K prefill is slow: `320.41s` prefill, `0.42s` decode.
- 4K MLX peak: `31,172,609,556` bytes.
- 4K RSS: `30,715,658,240` bytes.
- Broad suite passed: `84 passed`.

## Hard Invariants

- Do not materialize dense routed expert tensors in the resident inference path.
- Do not overwrite `artifacts/glm-4.5-air-vq`; all quantization variants use new artifact directories.
- Do not regress the proven short-smoke text without recording before/after evidence and deciding whether the regression is expected.
- Do not run larger context gates until 4K performance and memory behavior are benchmarked with repeatable harnesses.
- Do not add `mx.eval`, cache clearing, or dense-dequant calls to production model paths for instrumentation convenience.
- Dense/source-oracle materialization is allowed only in bounded validation code that reads selected layers/projections, never in the resident generation path.
- GLM-5.2 stays parked until a separate heavy-quantization or paging plan exists.

## Evidence And Source Facts

Local source inspection:

- `GLM45AirVQModel` binds non-experts normally and sparse routed experts as `QuantizedVQSwitchGLU`.
- `bind_glm45_air_non_expert_weights()` streams source tensors entry by entry and skips routed expert and MTP tensors.
- `QuantizedVQSwitchLinear.__call__()` currently calls `vq_switch_qmv()`.
- `vq_switch_qmv()` is scalar over tokens/routes and calls `vq_qmv()` inside Python loops.
- `gather_vqmm()` and `gather_vqmm_kernel()` already exist, but the Air switch path does not currently call them.
- `QuantizedVQLinear.__call__()` already dispatches batched inputs to `vq_qmm()`.
- `vq_qmv`, `vq_qmm`, and `gather_vqmm` Metal kernels are row-parallel fallback kernels with one output row per threadgroup and a threadgroup-staged codebook.

External research through Exa:

- MLX custom kernel docs say `mx.fast.metal_kernel()` should be constructed once and reused to avoid repeated Metal library/JIT overhead.
- MLX `gather_qmm` is the official quantized equivalent of `gather_mm`, accepts `lhs_indices`/`rhs_indices`, and exposes `sorted_indices` as a faster-path hint.
- `mlx-lm` `QuantizedSwitchLinear` calls `mx.gather_qmm(...)` directly for routed quantized experts.
- Apple Metal MPP guidance for GEMM emphasizes threadgroup tile size, simdgroup tile size, walk order, synchronization, static tensor extents, and fused-kernel designs.
- QuIP# supports E8 lattice codebooks plus randomized Hadamard incoherence and optional fine-tuning for low-bit quality.
- GPTQ supports Hessian-aware one-shot quantization; AWQ and SmoothQuant support activation-aware scale transforms; LLM.int8-style methods support selective precision for outlier-sensitive dimensions.

## File Map

Likely performance files:

- `benchmarks/bench_glm45_air_vq.py`: new resident benchmark harness for load, short decode, 1K/4K prefill, and token throughput.
- `src/mlx_vq/benchmark/metrics.py`: new stdlib-only helpers for MLX memory, RSS, `resource` usage, and optional macOS `vm_stat` deltas.
- `src/mlx_vq/benchmark/glm45_air.py`: new reusable Air load/generation/prefill benchmark routines.
- `src/mlx_vq/models/glm45_air_vq_adapter.py`: optional profiling hooks around attention, MoE, and layer calls.
- `src/mlx_vq/nn/switch_linear.py`: route token-level switch calls to fused gather when safe.
- `src/mlx_vq/ops/vq_switch.py`: make `gather_vqmm` cover per-route/down-projection shapes without scalar fallback when safe.
- `src/mlx_vq/kernels/gather_vqmm.py`, `src/mlx_vq/kernels/gather_vqmm.metal`: optimize gather path and support route-compacted forms if needed.
- `src/mlx_vq/kernels/vq_qmm.py`, `src/mlx_vq/kernels/vq_qmm.metal`: benchmark and possibly retile qmm.
- `tests/test_glm45_air_vq_adapter.py`, `tests/test_switch_routing.py`, `tests/test_gather_vqmm.py`: no-dense and fused-path regression coverage.

Likely quality files:

- `benchmarks/eval_glm45_air_quality.py`: new fixed-prompt/logit/perplexity quality harness.
- `src/mlx_vq/quality/glm45_air.py`: new Air-only quality ladder utilities.
- `src/mlx_vq/quality/prompts.py`: fixed prompt corpus and exact metadata.
- `src/mlx_vq/validate/glm45_air_vq.py`: extend selected layer/projection source-oracle probes.
- `src/mlx_vq/quant/rtn.py`: scale estimator variants after baseline quality metrics exist.
- `src/mlx_vq/convert/stream_convert.py`: artifact variant naming, manifest fields, scale-policy metadata.
- `tests/test_rtn_quantization.py`, `tests/test_stream_convert_b2.py`, `tests/test_glm45_air_vq_validation.py`: quality variant and manifest coverage.

Likely documentation/log files:

- `WORK_LOG.md`: one Ralph-loop entry per implementation slice with commands, results, evidence, and next slice.
- `DISCOVERY.md`: durable facts from benchmark/local source/research findings.
- `docs/GLM45_AIR_PERF_QUALITY_RECOVERY_PLAN.md`: this plan.

## Phase Dependency Map

- P0 preserves the Air baseline and gives every later phase a comparison anchor.
- P1 instrumentation must land before any performance or quality changes.
- P2 bottleneck localization depends on P1.
- P3 and P4 performance optimizations depend on P2.
- P5 context expansion depends on P1/P2 and at least one stable optimized 4K benchmark, not merely correctness.
- Q1 quality baseline depends on P1 metrics plumbing but can run before P3/P4 optimization.
- Q2 source-oracle quality probes depend on existing artifact/source validators and Q1.
- Q3/Q4/Q5 quality recovery candidates depend on Q1/Q2 before changing artifacts.
- Q6 optional RHT/QuIP-style work depends on earlier quality candidates failing clear gates.

## Phase P0: Baseline Preservation And Plan Handoff

**Objective:** Freeze the known-good Air proof in documentation and artifact metadata before any runtime or quantization changes.

**Smallest first slice:** Add a plan-only work-log entry that points to this document and states no implementation was performed.

**Files likely touched:**

- Modify: `WORK_LOG.md`
- Modify: `DISCOVERY.md`
- Create: `docs/GLM45_AIR_PERF_QUALITY_RECOVERY_PLAN.md`

**Focused verification:**

- Run `git diff -- docs/GLM45_AIR_PERF_QUALITY_RECOVERY_PLAN.md WORK_LOG.md DISCOVERY.md`.
- Expected: documentation-only changes; no `src/`, `tests/`, `scripts/`, or artifact edits for this plan-only slice.

**Phase gate:** A reviewer can start implementation from this document without re-reading the entire prior loop history.

**Expected risk:** Low. Risk is stale or ambiguous plan language that allows drift back to GLM-5.2 or dense routed expert loading.

**Fallback if gate fails:** Tighten scope language and add explicit first-slice commands before any implementation starts.

## Phase P1: Repeatable Air Benchmark And Smoke Harnesses

**Objective:** Add repeatable, appendable Air benchmark/smoke harnesses for resident load, short decode, 1K prefill, 4K prefill, and routed MoE kernel timing.

**Smallest first slice:** Add stdlib-only metric helpers and a synthetic test for memory/RSS/pageout payload schema without loading the model.

**Files likely touched:**

- Create: `src/mlx_vq/benchmark/__init__.py`
- Create: `src/mlx_vq/benchmark/metrics.py`
- Create: `tests/test_benchmark_metrics.py`
- Create later: `src/mlx_vq/benchmark/glm45_air.py`
- Create later: `benchmarks/bench_glm45_air_vq.py`

**Harness output schema:**

```json
{
  "model_id": "zai-org/GLM-4.5-Air",
  "artifact_dir": "artifacts/glm-4.5-air-vq",
  "scenario": "short_decode|prefill_1k|prefill_4k|moe_kernel",
  "prompt_id": "capital_france",
  "context_tokens": 4096,
  "max_new_tokens": 1,
  "elapsed_seconds": 320.41,
  "prefill_seconds": 320.41,
  "decode_seconds": 0.42,
  "prefill_tokens_per_second": 12.78,
  "decode_tokens_per_second": 2.38,
  "mlx_active_bytes": 29358880276,
  "mlx_peak_bytes": 31172609556,
  "mlx_cache_bytes": 0,
  "rss_bytes": 30715658240,
  "pageouts_delta": 0,
  "swapouts_delta": 0,
  "generated_text": "..."
}
```

**Focused verification:**

- `uv run pytest tests/test_benchmark_metrics.py -q`
- Expected: metric schema serializes, MLX memory keys exist, RSS is positive, missing `vm_stat` fields degrade to `null` rather than failing.

**Phase gate:**

- `uv run python benchmarks/bench_glm45_air_vq.py --scenario short_decode --max-new-tokens 8 --append-jsonl artifacts/benchmarks/glm45-air-baseline.jsonl`
- `uv run python benchmarks/bench_glm45_air_vq.py --scenario prefill_1k --append-jsonl artifacts/benchmarks/glm45-air-baseline.jsonl`
- `uv run python benchmarks/bench_glm45_air_vq.py --scenario prefill_4k --append-jsonl artifacts/benchmarks/glm45-air-baseline.jsonl`
- Each run records MLX active/peak/cache, RSS, pageouts/swapouts, elapsed prefill/decode, and token throughput.
- Each run can emit a markdown snippet suitable for `WORK_LOG.md`.

**Expected risk:** Medium. Resident Air load is expensive, and repeated 4K prefill is slow.

**Fallback if gate fails:** Keep the metric helper and run only synthetic plus short decode until 4K can be isolated; do not optimize without at least one reproducible slow 4K number.

## Phase P2: Bottleneck Localization

**Objective:** Determine whether 4K prefill slowness is dominated by attention, qmm, switch routing, Python loops, cache handling, qmm dispatch shape, or Metal kernel throughput.

**Smallest first slice:** Add a test that proves the current Air switch path uses scalar `vq_switch_qmv` rather than fused `gather_vqmm`, then record that as the first bottleneck fact.

**Files likely touched:**

- Modify: `tests/test_switch_routing.py`
- Modify later: `src/mlx_vq/models/glm45_air_vq_adapter.py`
- Create later: `src/mlx_vq/benchmark/glm45_air.py`
- Modify later: `benchmarks/bench_glm45_air_vq.py`

**Focused verification:**

- `uv run pytest tests/test_switch_routing.py::test_air_switch_path_reports_scalar_route_backend -q`
- Expected before optimization: current backend report is scalar route qmv for `QuantizedVQSwitchLinear.__call__()`.

**Bottleneck probes:**

- Add optional timing hooks around `GLM45AirVQDecoderLayer.__call__()`, attention, MoE gate, VQ gate/up/down projections, score-weighted reduce, shared expert, and cache creation/update.
- Time with `mx.eval(...)` only inside benchmark/profiling harnesses.
- Record `mx.get_peak_memory()` around each scenario with `mx.reset_peak_memory()`.
- Run microbenchmarks at Air shapes:
  - gate/up: `tokens in {1, 8, 128, 1024, 4096}`, `top_k=8`, `in_dim=4096`, `out_dim=1408`, `experts=128`, `group_size=512`.
  - down: flattened route count `tokens * top_k`, `in_dim=1408`, `out_dim=4096`, `group_size=352`.
  - attention-only or dense-layer-only approximations if hook evidence suggests attention dominates.
- Search for unnecessary barriers:
  - `rg -n "mx\\.eval|clear_cache|depends|np\\.array|decode_weight_matrix" src scripts benchmarks tests`
  - Classify each occurrence as benchmark/test/oracle/runtime.

**Phase gate:**

- `DISCOVERY.md` contains a durable bottleneck entry with per-component timings for short decode, 1K prefill, and 4K prefill.
- The entry explicitly answers whether Air prefill reaches fused `gather_vqmm`, falls back to scalar per-route qmv, or spends most time outside routed VQ.

**Expected risk:** Medium-high. MLX lazy evaluation can make naive timers lie.

**Fallback if gate fails:** Keep only coarse whole-call timings and microbenchmarks; do not modify kernels until the dominant layer family is clear.

## Phase P3: Route Fused Gather Into Air MoE

**Objective:** Ensure `QuantizedVQSwitchGLU` uses batched/fused paths for prefill where possible, without changing the resident artifact format.

**Smallest first slice:** Make token-level `QuantizedVQSwitchLinear(x, indices)` call `gather_vqmm(... implementation="metal")` when `x` is `[tokens, input_dims]` and `indices` is `[tokens, top_k]`.

**Files likely touched:**

- Modify: `src/mlx_vq/nn/switch_linear.py`
- Modify: `src/mlx_vq/ops/vq_switch.py`
- Modify: `tests/test_switch_routing.py`
- Modify: `tests/test_gather_vqmm.py`

**Focused verification:**

- Add monkeypatch coverage proving token-level gate/up projections call `gather_vqmm`.
- Run:
  - `uv run pytest tests/test_switch_routing.py tests/test_gather_vqmm.py tests/test_quantized_vq_switch_linear.py -q`

**Phase gate:**

- Real Air short smoke still generates exactly the baseline text or a logged, accepted equivalent.
- No dense expert parameters appear in `model.parameters()`.
- `benchmarks/bench_glm45_air_vq.py --scenario moe_kernel` shows fused path used for gate/up and records speedup or no-regression.

**Expected risk:** Medium. Gate/up can use fused token routing directly, but down projection receives per-route hidden states and needs careful shape handling.

**Fallback if gate fails:** Keep a feature flag defaulting to scalar route qmv; land tests and instrumentation first, then optimize down projection separately.

## Phase P4: Per-Route Down Projection And Sorted Expert Batching

**Objective:** Remove the remaining Python route loops from down projection and evaluate sorted expert batching/route compaction for better locality.

**Smallest first slice:** Extend `gather_vqmm()` so explicit `lhs_indices` no longer forces scalar `vq_switch_qmv`; flatten selected route inputs, call `gather_vqmm_kernel()` as `[routes, 1, out]`, and reshape.

**Files likely touched:**

- Modify: `src/mlx_vq/ops/vq_switch.py`
- Modify: `src/mlx_vq/kernels/gather_vqmm.py`
- Modify: `tests/test_gather_vqmm.py`
- Modify: `src/mlx_vq/nn/switch_linear.py`
- Modify: `benchmarks/bench_gather_vqmm.py`

**Focused verification:**

- `uv run pytest tests/test_gather_vqmm.py::test_gather_vqmm_explicit_lhs_uses_fused_kernel -q`
- Expected: explicit `lhs_indices` route path matches scalar oracle and calls fused kernel.

**Phase gate:**

- `benchmarks/bench_gather_vqmm.py` reports Air-shaped gate/up/down projection timings for scalar, direct fused, and sorted/compacted fused variants.
- Sorted and unsorted outputs are numerically identical after scatter.
- Air 1K and 4K prefill benchmark improves or `DISCOVERY.md` records why MoE is not the dominant bottleneck.

**Expected risk:** Medium-high. Sorting can add overhead for small decode batches and may not help unless repeated experts are common.

**Fallback if gate fails:** Use unsorted direct fused path for prefill and keep scalar qmv for decode or small route counts; record route-count thresholds.

## Phase P5: Kernel Tiling And Runtime Memory Optimization

**Objective:** Retile or specialize `gather_vqmm`/`vq_qmm` only after P2/P3/P4 prove kernel throughput is the bottleneck.

**Smallest first slice:** Add benchmark parameters that sweep `threadgroup`, codebook duplication, and batch sizes for Air gate/up/down shapes without changing kernel code.

**Files likely touched:**

- Modify: `benchmarks/bench_gather_vqmm.py`
- Modify: `benchmarks/bench_vq_qmv.py`
- Create: `benchmarks/bench_vq_qmm.py`
- Modify later: `src/mlx_vq/kernels/gather_vqmm.metal`
- Modify later: `src/mlx_vq/kernels/vq_qmm.metal`

**Focused verification:**

- `uv run python benchmarks/bench_gather_vqmm.py --tokens 128 --top-k 8 --experts 128 --in-dim 4096 --out-dim 1408 --group-size 512 --iterations 5 --warmup 2`
- `uv run python benchmarks/bench_gather_vqmm.py --tokens 1024 --top-k 8 --experts 128 --in-dim 1408 --out-dim 4096 --group-size 352 --iterations 3 --warmup 1`

**Optimization candidates:**

- Avoid Python loops in hot prefill paths.
- Avoid unnecessary `mx.eval` and cache-clearing barriers outside benchmark code.
- Evaluate tile sizes beyond the current one-row-per-threadgroup fallback.
- Evaluate simdgroup reductions versus threadgroup reductions.
- Evaluate codebook/threadgroup memory duplication only if it improves measured throughput.
- Consider an M5/Metal 4 tensor-ops prototype only behind capability detection and only after fallback paths are stable.

**Phase gate:**

- At least one kernel/runtime change has an Air-shaped before/after benchmark and full focused regression test.
- The change does not increase 4K MLX peak by more than 5 percent unless explicitly justified.
- The change does not add dense routed expert params.

**Expected risk:** High. Kernel changes can pass small correctness tests but hurt full Air due to shape, dtype, or memory-layout effects.

**Fallback if gate fails:** Revert to P3/P4 fused dispatch without retile; keep benchmark findings in `DISCOVERY.md`.

## Phase P6: Context Expansion Gates

**Objective:** Keep 4K as the baseline and only run larger context gates after 4K performance and memory are characterized.

**Smallest first slice:** Add a `--context-tokens` benchmark option with a dry-run guard that refuses `>4096` unless explicit thresholds are supplied.

**Files likely touched:**

- Modify: `benchmarks/bench_glm45_air_vq.py`
- Modify: `src/mlx_vq/benchmark/glm45_air.py`
- Modify: `tests/test_benchmark_metrics.py`

**Focused verification:**

- `uv run python benchmarks/bench_glm45_air_vq.py --scenario prefill --context-tokens 8192`
- Expected: FAIL with a clear message requiring `--allow-larger-context --max-mlx-peak-bytes ... --max-rss-bytes ... --max-prefill-seconds ...`.

**Minimum prerequisites before any 8K run:**

- 4K benchmark has at least two repeatable runs recorded after performance changes.
- 4K pageouts/swapouts remain `0`.
- 4K MLX peak is `<= 36,000,000,000` bytes, or a reviewed threshold is recorded.
- 4K RSS is `<= 40,000,000,000` bytes, or a reviewed threshold is recorded.
- 4K prefill is either at least 2x faster than `320.41s` or the bottleneck is proven to be outside routed VQ and accepted for expansion.

**8K pass/fail thresholds before running:**

- Finite prefill and decode logits.
- Pageouts/swapouts delta `0`.
- MLX peak `<= 42,000,000,000` bytes unless a new measured 4K slope justifies a different cap.
- RSS `<= 45,000,000,000` bytes unless a new measured 4K slope justifies a different cap.
- Prefill time `<= 2.4x` the optimized 4K prefill time.
- Decode time `<= 1.0s` for one greedy token.

**Phase gate:** 8K is either passed under predeclared thresholds or explicitly deferred with measured reasons.

**Expected risk:** Medium. Larger contexts may shift the bottleneck from MoE to attention/cache.

**Fallback if gate fails:** Keep 4K as the advertised gate and return to P2 with attention/cache-specific profiling.

## Phase Q1: Air-Only Quality Baseline

**Objective:** Define quality metrics before changing quantization.

**Smallest first slice:** Add a fixed prompt set and a quality harness that records generated text, token ids, finite-logit checks, entropy, top-k probabilities, and repeatability for the current baseline artifact.

**Files likely touched:**

- Create: `src/mlx_vq/quality/__init__.py`
- Create: `src/mlx_vq/quality/prompts.py`
- Create: `src/mlx_vq/quality/glm45_air.py`
- Create: `benchmarks/eval_glm45_air_quality.py`
- Create: `tests/test_glm45_air_quality.py`

**Prompt ladder:**

- `capital_france`: `The capital of France is`
- `short_math`: `Compute 17 + 25. Answer with only the number.`
- `code_completion`: `def fibonacci(n):`
- `instruction_following`: `Write one sentence about why caches help inference.`
- `long_recall_1k`: deterministic repeated document plus one answerable question.

**Metrics:**

- Exact prompt text and tokenizer revision.
- Generated text and token ids.
- Finite logits for prefill and decode.
- Top-1 token and top-k probabilities for fixed positions.
- Entropy and max probability sanity checks.
- Token throughput and memory metrics from P1.
- Small corpus negative log likelihood/perplexity on resident VQ only.

**What can be compared locally:**

- Before/after comparisons between Air VQ artifact variants.
- Layer/projection outputs against source tensor oracles for selected experts and prompts.
- Dense-dequant artifact oracle equality for VQ runtime correctness.

**What cannot be claimed locally yet:**

- Full dense Air top-1/top-k agreement across the whole model, because full dense Air residency is intentionally avoided.
- Absolute recovery to dense quality without a memory-safe dense/reference procedure.

**Focused verification:**

- `uv run pytest tests/test_glm45_air_quality.py -q`
- Expected: prompt registry is stable, output schema includes exact prompt IDs, quality harness can run on a tiny fixture.

**Phase gate:**

- Baseline Air VQ metrics are recorded in `artifacts/quality/glm45-air-baseline.jsonl`.
- `WORK_LOG.md` contains exact command(s), prompt IDs, generated text summary, and evidence path.

**Expected risk:** Medium. Quality metrics can become misleading if prompts or generation settings drift.

**Fallback if gate fails:** Keep only deterministic smoke prompts and layer-local probes until full quality harness stabilizes.

## Phase Q2: Source-Oracle Layer And Projection Probes

**Objective:** Use existing source shards and artifact validators to isolate quality loss without full dense Air residency.

**Smallest first slice:** Extend `scripts/validate_glm45_air_vq.py` to accept a fixed prompt-derived hidden-state capture for one sparse layer, while still reading only selected source tensors.

**Files likely touched:**

- Modify: `src/mlx_vq/validate/glm45_air_vq.py`
- Modify: `scripts/validate_glm45_air_vq.py`
- Modify: `tests/test_glm45_air_vq_validation.py`

**Focused verification:**

- `uv run pytest tests/test_glm45_air_vq_validation.py -q`
- Run existing layer probes:
  - `uv run python scripts/validate_glm45_air_vq.py --source-dir /Users/jack.mazac/.cache/huggingface/hub/models--zai-org--GLM-4.5-Air/snapshots/a24ceef6ce4f3536971efe9b778bdaa1bab18daa --artifact-dir artifacts/glm-4.5-air-vq --layer 1 --tokens 2 --min-source-weighted-cosine 0.5`
  - `uv run python scripts/validate_glm45_air_vq.py --source-dir /Users/jack.mazac/.cache/huggingface/hub/models--zai-org--GLM-4.5-Air/snapshots/a24ceef6ce4f3536971efe9b778bdaa1bab18daa --artifact-dir artifacts/glm-4.5-air-vq --layer 45 --tokens 1`

**Phase gate:**

- At least early/middle/late sparse layers have source-weighted metrics for gate/up/down and weighted routed output.
- The validator reports selected experts, source tensors read, and peak source tensor bytes.
- No full dense Air model is instantiated.

**Expected risk:** Medium. Random hidden states are useful for numeric isolation but not necessarily representative of prompt activations.

**Fallback if gate fails:** Keep random-hidden layer-local probes and delay prompt-derived activation capture until P2 profiling hooks exist.

## Phase Q3: Projection-Local Scale And RTN Policy Experiments

**Objective:** Improve data-free or lightly calibrated VQ quality by testing projection-local group size and scale estimators before heavier calibration.

**Smallest first slice:** Add a scale-estimator parameter to `quantize_weight_rtn()` with current max-abs as the default and a test proving default artifacts are unchanged.

**Files likely touched:**

- Modify: `src/mlx_vq/quant/rtn.py`
- Modify: `src/mlx_vq/convert/stream_convert.py`
- Modify: `tests/test_rtn_quantization.py`
- Modify: `tests/test_stream_convert_b2.py`

**Candidates:**

- Current max-abs scale baseline.
- MSE/RMSE-minimizing scalar per group.
- Percentile/clipped max scale per group.
- Projection-local group sizes beyond current `512` and Air `down_proj=352`, if dimensions allow.
- Per-projection policies:
  - gate/up: hidden `4096 -> 1408`, group `512`.
  - down: hidden `1408 -> 4096`, group `352`.

**Focused verification:**

- `uv run pytest tests/test_rtn_quantization.py tests/test_stream_convert_b2.py -q`
- Convert one selected layer/projection into a variant artifact directory.
- Run Q2 validator against that layer.

**Phase gate:**

- Each candidate has before/after source-oracle metrics and quality prompt metrics.
- Artifact manifest records scale estimator, group size, source revision, output bytes, and resume state.
- Resident no-dense invariant still passes on a bound variant or a tiny same-surface fixture.

**Expected risk:** Medium. Better layer-local source cosine may not improve end-to-end text.

**Fallback if gate fails:** Keep current max-abs RTN and escalate to Q4 activation-aware calibration.

## Phase Q4: Activation-Aware Calibration

**Objective:** Use a small calibration corpus to identify activation-sensitive projections/layers and choose scale or precision policies.

**Smallest first slice:** Add activation-stat collection for routed MoE inputs and selected experts without changing inference outputs.

**Files likely touched:**

- Create: `src/mlx_vq/quality/calibration.py`
- Modify: `src/mlx_vq/models/glm45_air_vq_adapter.py` only for optional benchmark/profile hooks.
- Modify: `benchmarks/eval_glm45_air_quality.py`
- Create: `tests/test_glm45_air_calibration.py`

**Calibration corpus:**

- 32 to 128 short text samples covering general, math, code, and instruction prompts.
- Optional 1K contexts only after P1/P2 benchmark harness is stable.
- Exact corpus text, tokenizer revision, and sampling settings are recorded.

**Candidate methods:**

- AWQ-style activation-aware channel salience.
- SmoothQuant-style equivalent scaling for activation outliers, only if it can be represented in source conversion and runtime without dense routed weights.
- Prompt-derived expert coverage reports so calibration does not pretend to cover unused experts.
- Top sensitive layers/projections list for Q5 selective resident-safe precision.

**Focused verification:**

- `uv run pytest tests/test_glm45_air_calibration.py -q`
- Run activation collection on the tiny fixture first.

**Phase gate:**

- Calibration output includes layer, projection, selected experts, activation stats, expert coverage, prompt IDs, and source/artifact revision.
- No dense routed expert tensors are added to resident inference.
- Before/after quality prompt metrics exist for any calibration-driven artifact.

**Expected risk:** High. Calibration can overfit a tiny corpus or fail to cover enough experts.

**Fallback if gate fails:** Use calibration only as a diagnostic ranking tool and proceed to Q5 selective precision based on repeated layer-local losses.

## Phase Q5: Resident-Safe Selective Precision And Normal-Path Review

**Objective:** Recover quality by selectively increasing precision where it matters, while preserving resident memory and no dense routed experts.

**Smallest first slice:** Add manifest support for mixed VQ code bits per layer/projection without changing the loader's default behavior.

**Files likely touched:**

- Modify: `src/mlx_vq/io/load.py`
- Modify: `src/mlx_vq/convert/stream_convert.py`
- Modify: `tests/test_quantized_vq_switch_linear.py`
- Modify: `tests/test_stream_convert_b2.py`

**Allowed resident-safe candidates:**

- VQ-1.0 baseline for all routed experts.
- VQ-2/E8P 16-bit codebooks for selected routed layers/projections if memory remains under gate.
- Projection-local scale/group-size variants.
- Existing normal-precision path review for router, shared expert, attention, embeddings, norms, and LM head.

**Disallowed in resident inference:**

- BF16 dense routed expert fallback.
- Full dense Air model residency.
- Loading raw `model.layers.*.mlp.experts.*` tensors into `model.parameters()`.

**Focused verification:**

- Loader tests prove mixed-code-bit VQ artifacts bind by metadata.
- No-dense guard tests still pass.

**Phase gate:**

- Any selective precision candidate has memory estimate, actual bind memory, quality before/after, and performance before/after.
- MLX peak/RSS do not exceed agreed Air resident caps.

**Expected risk:** High. Selective precision can recover quality but consume memory and slow prefill.

**Fallback if gate fails:** Keep baseline VQ-1.0 artifact and record candidate as rejected with memory/quality evidence.

## Phase Q6: Optional RHT, QuIP-Style, Or Heavier Calibration

**Objective:** Only after Q3-Q5 are insufficient, evaluate heavier recovery methods such as RHT/QuIP-style incoherence, adaptive rounding, or error-propagation calibration.

**Smallest first slice:** Write a design note with disk/runtime estimates before implementing any converter changes.

**Files likely touched:**

- Create: `docs/GLM45_AIR_HEAVY_QUANTIZATION_RECOVERY_NOTE.md`
- Modify later: `src/mlx_vq/quant/rht.py`
- Modify later: `src/mlx_vq/convert/stream_convert.py`
- Modify later: `src/mlx_vq/codebook/e8.py`

**Candidate methods:**

- QuIP#-style randomized Hadamard preprocessing.
- Hessian-aware or GPTQ-like selected projection calibration.
- Error-propagation/asymmetric calibration for layer outputs.
- Adaptive/codebook rounding only if storage/runtime format remains clear.

**Phase gate before implementation:**

- Resume manifest design exists.
- Disk estimate exists for full Air artifact regeneration.
- Runtime estimate exists for one layer and full conversion.
- Rollback path exists to baseline artifact.

**Expected risk:** Very high. These methods can balloon complexity and invalidate simple runtime assumptions.

**Fallback if gate fails:** Do not implement heavy recovery in this plan; preserve current artifact and documented quality limits.

## First Implementation Slice

Start here after review and explicit implementation approval.

### Slice 1: Baseline Benchmark Metrics Skeleton

**Objective:** Land metric helpers and a JSON schema test without loading GLM-4.5-Air.

**Files:**

- Create: `src/mlx_vq/benchmark/__init__.py`
- Create: `src/mlx_vq/benchmark/metrics.py`
- Create: `tests/test_benchmark_metrics.py`
- Modify: `WORK_LOG.md`
- Modify: `DISCOVERY.md` only if a durable API fact is learned.

- [ ] **Step 1: Add a failing schema test**

Expected test behavior:

```python
def test_metric_snapshot_has_required_keys():
    snapshot = collect_metric_snapshot()
    assert snapshot["mlx_active_bytes"] is not None
    assert snapshot["mlx_peak_bytes"] is not None
    assert snapshot["rss_bytes"] > 0
    assert "pageouts_delta" in snapshot
    assert "swapouts_delta" in snapshot
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
uv run pytest tests/test_benchmark_metrics.py -q
```

Expected: FAIL because `mlx_vq.benchmark.metrics` does not exist yet.

- [ ] **Step 3: Implement metric helpers**

Use:

- `mx.get_active_memory()`
- `mx.get_peak_memory()`
- `mx.get_cache_memory()`
- `mx.reset_peak_memory()`
- `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`
- macOS `vm_stat` parsing as optional best-effort for pageouts/swapouts

- [ ] **Step 4: Run focused verification**

Run:

```bash
uv run pytest tests/test_benchmark_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Run broad non-resident guard**

Run:

```bash
uv run pytest tests/test_safetensors_schema.py tests/test_stream_convert_planning.py tests/test_source_safetensors.py tests/test_glm45_air_vq_adapter.py -q
```

Expected: PASS. This verifies the metrics skeleton did not disturb resident-load or safetensors contracts.

- [ ] **Step 6: Record the loop**

Append to `WORK_LOG.md`:

- files touched
- focused command/result
- broad guard command/result
- next slice: resident Air benchmark harness

## Review Checklist

Before implementation starts:

- Air baseline evidence is visible in this plan.
- The first implementation slice does not load the full model.
- Performance and quality phases are separate but share P1 metrics.
- Larger context gates have explicit prerequisites and thresholds.
- No phase proposes resident BF16 dense routed expert fallback.
- GLM-5.2 is parked except for explicit scope language.

## Sources

- Local plan and proof ledger: `docs/PLAN.md`, `WORK_LOG.md`, `DISCOVERY.md`
- Local Air resident adapter: `src/mlx_vq/models/glm45_air_vq_adapter.py`
- Local switch path: `src/mlx_vq/nn/switch_linear.py`, `src/mlx_vq/ops/vq_switch.py`
- Local kernels: `src/mlx_vq/kernels/vq_qmv.py`, `src/mlx_vq/kernels/vq_qmm.py`, `src/mlx_vq/kernels/gather_vqmm.py`
- MLX custom kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- MLX `gather_qmm`: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html
- mlx-lm switch layers: https://github.com/ml-explore/mlx-lm/blob/564281f7/mlx_lm/models/switch_layers.py
- Apple Metal MPP guide: https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf
- GLM-4.5-Air model card: https://huggingface.co/zai-org/GLM-4.5-Air
- QuIP#: https://arxiv.org/html/2402.04396v2
- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
- SmoothQuant: https://proceedings.mlr.press/v202/xiao23c.html
- LLM.int8: https://proceedings.neurips.cc/paper_files/paper/2022/file/c3ba4962c05c49636d4c6206a97e9c8a-Paper-Conference.pdf
