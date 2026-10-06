# GLM-5.2 Codebook-VQ MLX/Metal Implementation Plan

## Summary

Build a standalone `mlx_vq` package in this repo that adds QuIP# E8-family codebook VQ inference on Apple Silicon through custom MLX modules and `mx.fast.metal_kernel` wrappers. v1 does not fork MLX: it stores VQ codes/scales in safetensors, loads them as integer tensors, and decodes only inside fused Metal kernels. The default production target is GLM-5.2 VQ-1.0 resident inference on a 128 GB M5 Max; VQ-2.0 plus SSD paging is implemented behind a flag after the resident path works.

Verified anchors:
- Local venv has `mlx.core` 0.31.2; `mlx_lm` is not installed.
- Local `mx.load` preserves `.safetensors` `uint8` and `uint16` tensors, so VQ codes can survive load without fp casting.
- Local `mx.hadamard_transform` supports 2048, 4096, 6144, and 12288 final-axis sizes.
- Local `mx.quantize(..., mode="e8p")` fails, confirming no stock VQ mode.
- GLM-5.2 config is now published: 78 layers, 3 dense + 75 sparse, 256 routed experts, top-8, 21 full DSA indexer layers + 57 shared, hidden 6144, MoE intermediate 2048, 1M context.
- GLM-4.5-Air config: `glm4_moe`, 46 layers, 1 dense + 45 sparse, 128 routed experts, top-8, hidden 4096, MoE intermediate 1408.

## Key Changes

Create this package layout:

```text
pyproject.toml
src/mlx_vq/
  codebook/e8.py
  quant/rht.py
  quant/rtn.py
  quant/policy.py
  ops/vq_linear.py
  ops/vq_switch.py
  kernels/common_e8.metal
  kernels/vq_qmv.metal
  kernels/vq_qmm.metal
  kernels/gather_vqmm.metal
  nn/linear.py
  nn/switch_linear.py
  io/schema.py
  io/load.py
  convert/inspect_hf.py
  convert/stream_convert.py
  convert/write_shards.py
  models/glm4_moe_adapter.py
  models/glm_moe_dsa_adapter.py
  paging/madvise.py
tests/
  test_e8_reference.py
  test_vq_qmv.py
  test_vq_qmm.py
  test_safetensors_schema.py
  test_switch_routing.py
  test_stream_convert_planning.py
benchmarks/
  bench_vq_qmv.py
  bench_gather_vqmm.py
```

Dependency policy:
- Add `mlx>=0.31.2`, `mlx-lm>=0.31.3`, `huggingface_hub`, `safetensors`, `numpy`, `pytest`, and `pytest-benchmark`.
- Treat `mlx-lm` GLM-5.2 support as conditional. At execution time, require an upstream version that includes IndexShare support or vendor the small adapter equivalent of mlx-lm PRs #1410/#1412.
- Do not depend on `mx.quantize(mode="e8p")`; all VQ is custom until the optional MLX C++ fork milestone.

Core interfaces:

```python
def dequantize_e8(
    codes: mx.array,              # uint8 or uint16, shape [..., out, in//8]
    scales: mx.array,             # fp16/fp32, shape [..., out, in//group_size]
    codebook: mx.array,           # uint32[256]
    *,
    in_dim: int,
    group_size: int = 512,
    code_bits: int = 8,           # 8 for VQ-1.0, 16 for VQ-2.0
    dtype: mx.Dtype = mx.float16,
) -> mx.array:                    # shape [..., out, in_dim], test-only dense materialization

def encode_e8_rtn(
    weight: mx.array,             # fp16/bf16/fp32 [out, in], in % 8 == 0
    codebook: mx.array,
    *,
    group_size: int = 512,
    code_bits: int = 8,
    rht: "RHTSpec | None" = None,
) -> tuple[mx.array, mx.array, "RHTSpec"]:
    # returns codes [out, in//8], scales [out, in//group_size], rht metadata
```

```python
def vq_qmv(x, codes, scales, codebook, *, in_dim, out_dim, group_size=512, code_bits=8) -> mx.array:
    # x shape [in] or [B, in], codes [out, in//8], output [out] or [B, out]

def vq_qmm(x, codes, scales, codebook, *, in_dim, out_dim, group_size=512, code_bits=8) -> mx.array:
    # x shape [M, in], output [M, out]

def gather_vqmm(x, codes, scales, codebook, rhs_indices, *,
                 lhs_indices=None, sorted_indices=False,
                 group_size=512, code_bits=8) -> mx.array:
    # x [T, hidden], codes [E, out, in//8], rhs_indices [T, top_k], output [T, top_k, out]
```

```python
class QuantizedVQLinear(nn.Module):
    codes: mx.array       # [out, in//8], uint8 or uint16
    scales: mx.array      # [out, in//group_size], fp16
    codebook: mx.array    # uint32[256]
    bias: mx.array | None
    input_dims: int
    output_dims: int
    code_bits: int
    group_size: int
    rht: RHTSpec | None

class QuantizedVQSwitchLinear(nn.Module):
    codes: mx.array       # [num_experts, out, in//8]
    scales: mx.array      # [num_experts, out, in//group_size]
    def __call__(self, x, indices, *, sorted_indices=False) -> mx.array
```

On-disk schema:
- `*.codes`: `uint8` for VQ-1.0 or `uint16` for VQ-2.0, shape `[out, in/8]` or `[experts, out, in/8]`, row-major, contiguous along `in/8`.
- `*.scales`: `float16`, shape `[out, in/group]` or `[experts, out, in/group]`, default `group=512`.
- `model.vq_codebook.e8`: `uint32[256]`, stored once per shard set and validated against a hash in config.
- `*.rht_su`, `*.rht_sv`: packed sign bits or deterministic seeds. Default: store seeds, with an option to materialize packed signs for reproducibility debugging.
- `quantization_config.json`:
```json
{
  "quant_method": "mlx_vq_e8",
  "version": 1,
  "default_code_bits": 8,
  "default_group_size": 512,
  "codebook": {"name": "quip_e8", "dtype": "uint32", "entries": 256, "sha256": "..."},
  "rht": {"mode": "seeded_signs", "hadamard": "mlx.core.hadamard_transform"},
  "policy": {
    "routed_experts": "vq",
    "attention": "affine_6_or_8_bit",
    "dsa_indexer": "affine_8_bit_or_fp16",
    "router": "fp16_or_affine_8_bit",
    "shared_experts": "affine_6_bit",
    "dense_ffn": "affine_6_bit",
    "embeddings_lm_head": "affine_6_bit",
    "norms": "fp16_or_fp32"
  }
}
```

## Implementation Plan

### Phase A0: Project Scaffolding And API Gates
Depends on: none
Effort: 0.5 engineer-week
Risk: low

- Add package/dependency files and a `scripts/verify_env.py` command that fails unless `mlx.core.__version__ >= 0.31.2`, `mx.fast.metal_kernel` exists, `mx.hadamard_transform` supports 4096/6144/12288, and safetensors integer dtypes round-trip.
- Add `inspect_hf.py` to fetch and validate GLM-4.5-Air and GLM-5.2 configs before any conversion. It must compute sparse-layer count from config, not constants.
- Validation gate: `pytest tests/test_safetensors_schema.py tests/test_stream_convert_planning.py`.

### Phase A1: E8 Reference Oracle
Depends on: A0
Effort: 1 engineer-week
Risk: medium

- Implement `src/mlx_vq/codebook/e8.py` with imported QuIP# codebook constants, exact codeword decoding, and RTN encode/decode reference paths.
- Use QuIP# `get_full_grid()` parity/shift behavior as the golden reference for 16-bit E8P. For 8-bit VQ-1.0, define the exact selected 256-entry E8 table and document that it is not the same storage contract as 16-bit E8P with signs.
- Validation gate: bit-exact decode against upstream QuIP# vectors; random tensor dense materialization tests require exact codes and cosine >= 0.999999 versus numpy reference.

### Phase A2: `vq_qmv` Decode-Matvec Kernel
Depends on: A1
Effort: 2 to 3 engineer-weeks
Risk: high

- Implement `common_e8.metal` decode helpers and `vq_qmv.metal`.
- Build kernels once through `mx.fast.metal_kernel(name, input_names, output_names, source, header)`, then reuse callables. No per-call JIT construction.
- Kernel design:
  - Load `uint32[256]` codebook into threadgroup memory.
  - Start with 4x codebook duplication; benchmark 1x, 4x, 8x. Do not use 32x unless qmv has no activation tile.
  - Lanes read adjacent codewords along `[out, in//8]`.
  - Decode 8 weights per codeword into registers, apply per-row-group scale, accumulate fp32, reduce with simd/quad reduction, write fp16/bf16 output.
- Validation gate: for shapes from toy through GLM expert dimensions, cosine >= 0.99999 and max-abs error within fp16 accumulation tolerance versus `dequantize_e8(...) @ x`. Perf gate: sustained bandwidth >= 50 percent of measured local memory bandwidth; if <30 percent, stop and retile.

### Phase A3: Safetensors Container And `QuantizedVQLinear`
Depends on: A2
Effort: 1.5 to 2 engineer-weeks
Risk: medium

- Implement `io/schema.py`, `io/load.py`, and `nn/linear.py`.
- Loader must not call `mx.dequantize` or materialize dense expert weights. It should instantiate `QuantizedVQLinear` directly from `codes`, `scales`, config metadata, and optional bias.
- Add resident-memory guards that estimate bytes from safetensors metadata before loading arrays.
- Validation gate: load a synthetic VQ checkpoint, assert integer dtypes stay intact, assert RSS does not jump by dense-weight size, and generate one dense-layer output through the real kernel.

### Phase A4: Small Dense Model Generation
Depends on: A3
Effort: 1 engineer-week
Risk: medium

- Use `Qwen/Qwen2.5-1.5B-Instruct` as the dense validation target because mlx-lm supports Qwen2-family models. Convert only Linear weights to VQ for this rung.
- Keep embeddings, norms, and LM head affine/fp16 first; only widen scope after the fused VQ path generates coherent tokens.
- Validation gate: prompt generation works through `QuantizedVQLinear`; logits cosine versus dense-dequant reference is measured on a fixed prompt batch.

### Phase A5: `QuantizedVQSwitchLinear` And GLM-4.5-Air MoE Fixture
Depends on: A3, A4
Effort: 3 engineer-weeks
Risk: high

- Implement `nn/switch_linear.py`, `ops/vq_switch.py`, and `models/glm4_moe_adapter.py`.
- Mirror mlx-lm GLM4 MoE routing: sigmoid router, group mask, top-k, normalized top-k probabilities, routed scaling, shared expert add.
- Store GLM-4.5-Air routed expert weights as stacked tensors:
  - `switch_mlp.gate_proj.codes [128, 1408, 4096/8]`
  - `switch_mlp.up_proj.codes [128, 1408, 4096/8]`
  - `switch_mlp.down_proj.codes [128, 4096, 1408/8]`
- Validation gate: route-token equivalence versus dense GLM4 MoE adapter on random inputs; then end-to-end GLM-4.5-Air generation with no dense expert materialization.

### Phase A6: `gather_vqmm` Sorted Expert Batching
Depends on: A5
Effort: 2 to 3 engineer-weeks
Risk: high

- Implement the VQ analog of `mx.gather_qmm` for routed experts.
- First implementation may sort `rhs_indices` in Python using mlx-lm’s `_gather_sort`/`_scatter_unsort` pattern; kernel receives sorted tokens and expert ids.
- Output contract is `[tokens, top_k, out]`; caller multiplies by router weights and sums over `top_k`.
- Validation gate: unsorted and sorted paths are numerically identical after scatter; throughput improves for repeated expert ids on GLM-4.5-Air token batches.

### Phase A7: `vq_qmm` Prefill Kernel
Depends on: A2, A6
Effort: 3 engineer-weeks
Risk: medium-high

- Implement the large-M prefill path with a two-tier strategy:
  - v1 fallback: decode KxN tiles to threadgroup/register tiles and use simdgroup matrix multiply.
  - M5 path: prototype Metal 4 tensor / MPP TensorOps path for quantized matmul, guarded by capability detection.
- Do not make Metal 4 required for M4 Max; M4 uses the simdgroup fallback.
- Validation gate: `M in {8, 32, 128, 512}` prefill outputs match dense-dequant reference with cosine >= 0.99999; dispatch selects qmv for decode and qmm for prefill.

### Phase B1: GLM-5.2 Loader Adapter And Mixed Precision Policy
Depends on: A6
Effort: 2 engineer-weeks
Risk: high

- Implement or vendor `glm_moe_dsa` IndexShare support if the installed `mlx-lm` still lacks merged PR #1410/#1412 behavior.
- Enforce GLM-5.2 config gates:
  - `model_type == "glm_moe_dsa"`
  - `num_hidden_layers == 78`
  - `mlp_layer_types` has 3 dense and 75 sparse
  - `n_routed_experts == 256`
  - `num_experts_per_tok == 8`
  - `indexer_types` has 21 `full` and 57 `shared`
- Keep DSA indexer, MLA attention, router, shared expert, dense FFNs, embeddings, LM head, and norms out of VQ; use stock affine quant or fp16/fp32.
- Validation gate: model instantiates and binds all non-VQ weights without missing/extra parameter failures.

### Phase B2: Sharded Streaming RTN Converter
Depends on: A1, A3, B1
Effort: 3 to 4 engineer-weeks
Risk: high

- Implement `convert/stream_convert.py` to stream one tensor or one expert projection at a time from the HF sharded safetensors index.
- Never collect all parameters in a dict. Plan output shards from metadata, read source tensors just-in-time, write VQ/affine shard, free, then call `mx.metal.clear_cache()`.
- For fused HF expert tensors, preserve the GLM convention: `gate_up_proj` splits or stacks into gate/up contracts exactly once, then writes VQ codes.
- Resident peak target per routed expert projection is under 1 GB; the prompt’s 75 MB per expert is plausible for a single bf16 expert projection chunk, but the converter must measure and report actual peak RSS.
- Validation gate: convert a complete GLM-4.5-Air checkpoint first; then dry-run GLM-5.2 shard planning without downloading all weights; then full GLM-5.2 conversion.

### Phase B3: GLM-5.2 Resident VQ-1.0 Inference
Depends on: B2, A7
Effort: 2 engineer-weeks
Risk: high

- Default operating point: VQ-1.0 routed experts, group size 512 scales.
- Budget from verified config:
  - Routed experts: 724.78B params.
  - VQ-1.0 codes: 90.60 GB decimal.
  - Group-512 scales: 2.83 GB decimal.
  - Routed expert VQ total: 93.43 GB decimal.
  - Non-experts at 6 to 8 bit plus norms and runtime should be budgeted around 16 to 24 GB until measured.
- KV-cache cap:
  - If storing expanded K/V, 1M context is impossible at multi-TB scale.
  - With MLA latent cache approximation `(kv_lora_rank + qk_rope_head_dim) * layers * 2 bytes`, GLM-5.2 costs about 89.86 GB per 1M tokens, so a 10 to 15 GB cache budget gives roughly 111K to 167K tokens before allocator/OS overhead.
  - Default `max_model_len` for VQ-1.0 on 128 GB: 96K tokens; allow manual override only after live RSS measurement.
- Validation gate: coherent generation on M5 Max with RSS below 120 GB, no paging, fixed prompt logits sanity, and generation at 4K, 32K, and 96K contexts.

### Phase B4: Quality Ladder And Optional Calibration
Depends on: B3
Effort: 2 to 6 engineer-weeks depending on calibration depth
Risk: high

- v1 quality path is data-free RTN only.
- Add BlockLDLQ/Hessian calibration only if RTN fails agreed quality thresholds.
- Calibration must use natural routing and report expert coverage; do not force every expert on synthetic tokens.
- Eval gates:
  - perplexity or mean-KLD on sampled corpora,
  - top-1 agreement versus bf16 or high-bit baseline,
  - coding-focused smoke evals aligned with GLM-5.2’s model card: Terminal-Bench-style tasks, SWE-style patches, and long-context tool prompts.
- Fallback policy: if VQ-1.0 fails quality, test VQ-1.5 before VQ-2.0 paging.

### Phase B5: VQ-2.0 Paging Behind A Flag
Depends on: B3
Effort: 3 engineer-weeks
Risk: high

- Implement `paging/madvise.py` using mmap plus `madvise(MADV_WILLNEED/DONTNEED)` where available.
- Keep non-experts, codebook, and hot expert windows resident; prefetch layer `L+1` selected experts while computing layer `L`.
- VQ-2.0 budget: 181.19 GB codes + 2.83 GB group-512 scales for routed experts, so it cannot be resident on 128 GB.
- Validation gate: no paging code runs unless `--expert-paging=vq2`; measure tokens/sec, page faults, cache hit rate, and latency spikes.

### Phase C: Native MLX C++ Mode, Deferred
Depends on: stable VQ format and kernels
Effort: 3 engineer-weeks
Risk: medium

- Add `QuantizationMode::e8p` in MLX C++ only after Python custom-op behavior is stable.
- Mirror MLX `quantized.cpp` qmv/qmm/gather_qmm integration.
- This is not required for v1 shipping.

## Test Plan

- Reference tests: bit-exact E8 decode, RTN encode/decode round trip, RHT seed reproducibility.
- Kernel tests: `vq_qmv`, `vq_qmm`, and `gather_vqmm` versus dense-dequant reference on random and GLM-shaped tensors.
- Loader tests: safetensors integer dtype preservation, schema validation, no dense materialization, strict missing/extra key reporting.
- Model tests: small dense generation, GLM-4.5-Air MoE routing equivalence, GLM-5.2 adapter parameter binding.
- Conversion tests: dry-run shard plan from HF index, GLM-4.5-Air full conversion, GLM-5.2 metadata-only planning, GLM-5.2 full conversion.
- Performance tests: measured memory bandwidth baseline, qmv >= 50 percent target, investigate below 30 percent, sorted `gather_vqmm` faster than unsorted on repeated experts.
- Resident tests: record RSS, page faults, KV cache growth, and context cap behavior on M5 Max and M4 Max.

## Assumptions And Open Verification

- QuIP# E8 8-bit VQ-1.0 codebook selection must be frozen before implementation; 16-bit E8P is unambiguous, but the 8-bit operating point needs an exact table contract.
- Apple GPU qmv/qmm throughput is unproven; NVIDIA QuIP# performance numbers are only targets.
- Metal 4 tensor operations should be treated as an M5 optimization, not a v1 dependency.
- GLM-5.2 HF model page reports 753B in one metadata field while config/math and official text use 744B. Converter must trust tensor inventory over either label.
- `mlx-lm` GLM-5.2 IndexShare support is in flux. Execution must pin a known-good commit or vendor the adapter until upstream release catches up.
- Full 1M context is not a 128 GB target. Default resident `max_model_len` is 96K until live KV/RSS data proves a higher cap.

## Sources Used

- Local planning prompt: [Codex Planning Prompt](/Users/jack.mazac/Developer/glm/Codex%20Planning%20Prompt%20%E2%80%94%20GLM-4_5-Air%20%E2%86%92%20GLM-5_2%20VQ%20Implementation.md)
- Local architecture doc: [Native Codebook-VQ Architecture](/Users/jack.mazac/Developer/glm/Native%20Codebook-VQ%20Sub-2-Bit%20MoE%20Inference%20on%20Apple%20Silicon_%20An%20MLX_Metal%20Implementation%20Architecture%20for%20GLM-5_2.md)
- Local Unsloth notes: [unsloth-glm-5-2.md](/Users/jack.mazac/Developer/glm/unsloth-glm-5-2.md)
- GLM-5.2 config: [zai-org/GLM-5.2 config.json](https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json)
- GLM-4.5-Air config: [zai-org/GLM-4.5-Air config.json](https://huggingface.co/zai-org/GLM-4.5-Air/raw/main/config.json)
- MLX custom kernels: [mx.fast.metal_kernel docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html)
- MLX quantization: [mx.quantize docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
- MLX MoE quantized gather: [mx.gather_qmm docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html)
- QuIP#: [paper](https://arxiv.org/html/2402.04396v2) and [quip-sharp E8P kernel](https://github.com/Cornell-RelaxML/quip-sharp/blob/main/quiptools/quiptools_e8p_gemv.cu)
- GLM-5.2 mlx-lm IndexShare status: [PR #1410](https://github.com/ml-explore/mlx-lm/pull/1410), [PR #1412](https://github.com/ml-explore/mlx-lm/pull/1412), [issue #1418](https://github.com/ml-explore/mlx-lm/issues/1418)
- Metal 4 tensor ops: [Apple inline ML operations](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4), [WWDC26 Metal guide](https://developer.apple.com/wwdc26/guides/metal/)
