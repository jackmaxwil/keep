# Native Codebook-VQ Sub-2-Bit MoE Inference on Apple Silicon (MLX/Metal): An Implementation Architecture for GLM-5.2

## TL;DR

- **Build it around the QuIP# E8P 8-dimensional lattice codebook, not AQLM**: the E8 lattice codebook is *data-free and reusable* (a fixed 1 KiB / 256-entry table shared across the entire model — QuIP#‘s E8 symmetry makes it “1000X smaller than a naive 8D codebook (1KiB vs 1MiB)”), decodes in “<4 instructions per weight,” and fits trivially in Metal’s 32 KB threadgroup memory — whereas AQLM’s per-model learned codebooks cost “about 1 day on a single A100” for 7B and “10-14 days” for 70B and are too large for L1/threadgroup cache. Crush only the routed-expert FFN weights (~97% of GLM-5.2’s 744B params) to ~1.0–1.5 effective bpw; keep MLA attention, the DSA indexer, router/gates, shared experts, embeddings and norms at 6–8 bit.
- **MLX has no native VQ/codebook quantization mode today** (it supports only `affine`, `mxfp4`, `mxfp8`, `nvfp4`; bits 2/3/4/5/6/8; group 32/64/128, packed in `uint32`). You must add a new mode. For v1, implement it as a `QuantizedVQLinear`/`gather_vqmm` custom op via `mx.fast.metal_kernel` storing raw packed arrays in safetensors and mmap-loading them with **no bf16 materialization**; for the clean “native MLX quant type,” fork the MLX C++ `quantized.cpp`/`Quantize` enum later.
- **Resident-RAM math is the binding constraint on a 128 GB Mac**: at 2 bpw (E8P), the experts alone are ~181 GB and require SSD expert-paging; at ~1.0–1.25 bpw (1-bit E8 lattice / RVQ) the experts are ~91–113 GB and the whole model fits resident at ~110–125 GB (~92% smaller than the 1,488 GB bf16 checkpoint). Choose the bpw to either fit fully resident or to combine VQ with paging.

## Key Findings

**Ground truth — GLM-5.2 architecture (confirmed).** GLM-5.2 (Zhipu/Z.ai, released June 13, 2026, MIT license) is a Mixture-of-Experts transformer with **744B total / ~40B active parameters**, **256 routed experts with top-8 routing plus 1 shared expert**, and DeepSeek Sparse Attention (DSA). The 753B figure in some secondary blogs is unconfirmed; the official `config.json` and the GLM-5 technical report (arXiv 2602.15763) give 744B. **GLM-5.2 shares the GLM-5 architecture**; where a GLM-5.2-specific config field was not directly fetchable, I substitute the confirmed GLM-5 `config.json` (model_type `glm_moe_dsa`) and flag it. Per-tensor shapes used for the bit budget:

- `hidden_size` = 6144; `num_hidden_layers` = 78 (`first_k_dense_replace` = 3 dense layers, remainder MoE; `num_nextn_predict_layers` = 1 MTP layer).
- MoE: `n_routed_experts` = 256, `n_shared_experts` = 1, `num_experts_per_tok` = 8, `moe_intermediate_size` = 2048; dense `intermediate_size` = 12288.
- Attention (MLA): `q_lora_rank` = 2048, `kv_lora_rank` = 512, `qk_nope_head_dim` = 192, `qk_rope_head_dim` = 64 (`qk_head_dim` = 256), `v_head_dim` = 256, `num_attention_heads` = 64, `head_dim` = 64.
- DSA indexer: `index_n_heads` = 32, `index_head_dim` = 128, `index_topk` = 2048.
- `vocab_size` = 154880; `rms_norm_eps` = 1e-5; `scoring_func` = sigmoid; `routed_scaling_factor` = 2.5; `tie_word_embeddings` = false; dtype bf16.

**Parameter distribution (drives the mixed-precision policy).** Each routed expert is a SwiGLU FFN with gate+up+down = 3 × 6144 × 2048 ≈ 37.75M params. With 256 experts × ~75 MoE layers ≈ 19,200 experts → ≈ **725B params in routed experts ≈ 97% of the model**. Everything else (MLA attention, DSA indexer, router/gate, shared experts, 3 dense FFNs, embeddings, norms) totals only ~19B (~3%). This is why crushing *only* the routed experts to ~1 bit yields ~90% total size reduction while protecting the quality-critical tensors.

**MLX quantization framework (confirmed capabilities & limits).** MLX (`mx.quantize`/`mx.dequantize`, `nn.QuantizedLinear`, `QuantizedSwitchLinear`, `mx.quantized_matmul`, `mx.gather_qmm`) packs weights into `uint32` with per-group fp16 `scales`/`biases`; modes are `affine`, `mxfp4`, `mxfp8`, `nvfp4`; bit-widths 2/3/4/5/6/8; group sizes 32/64/128. MoE is handled by `gather_qmm` (the quantized equivalent of `gather_mm`: `y[i,j] = x[lhs_indices[i]] @ w[rhs_indices[j]]`), with a `sorted_indices` fast path (`gather_qmm_rhs`) that batches consecutive identical expert indices.  The Metal backend has `qmv`/`qvm`/`qmm` kernels (qmv for M<~4 decode using `quad_sum`; qmm for larger M using `simdgroup_matrix` MMA, currently accumulating in fp32). **Custom kernels are exposed through `mx.fast.metal_kernel`** (JIT-compiled MSL, you pass the kernel body; MLX auto-generates the signature and supplies `{name}_shape/_strides/_ndim`). **There is no codebook/VQ mode and no Python API to register a new `mode` string** — quantization modes are C++ enums in `mlx/backend/metal/quantized.cpp`. This is the single biggest tooling gap.

**Reusable VQ methods (what to import vs. learn).**

- **QuIP# E8P — REUSABLE, recommended.** Fixed E8-lattice codebook (data-free). 16-bit codeword = **8 bits base index + 7 stored sign bits + 1 shift bit (±¼)**; the 8th sign is inferred from parity, so 2^16 effective codewords decode from a **256-entry table  of 1 KiB (256 × int32, each int32 packing 8 four-bit absolute values; entry encoding `value*2+8` maps half-integers into nibbles)**. The source table S = 227 vectors of |D̂8| with norm ≤ √10 plus 29 padding vectors of norm √12. Decodes in “<4 instructions per weight”; fits L1/threadgroup. Per-model work is only the (cheap, data-free) random Hadamard transform plus optional Hessian-based BlockLDLQ rounding. 3-bit = E8P + a 1-bit E8 codebook; 4-bit = E8P applied twice (RVQ),  with residual scales ≈1.03 (stage 1, `opt_scale=1.03` in source) and ≈3.45 (stage 2). A **1-bit E8 codebook** (256-entry, 8-dim → 1.0 bpw) is exactly the residual stage and is the natural sub-2-bit primitive. (Sources: arXiv 2402.04396 §4.2/Appendix C; `Cornell-RelaxML/quip-sharp` `lib/codebook/latticee8_padded12.py` and `quiptools/quiptools_e8p_gemv.cu`.)
- **AQLM — NOT recommended for import.** Per-model *learned* additive codebooks (>99% of quantized params live in the codebooks); per the official repo “quantizing a 7B model with default configuration takes about 1 day on a single A100 gpu… a 70B model on a single GPU would take 10-14 days” (7B → ~14h on 2 GPUs, 70B → 3 days 18h on 8×A100). Best 2-bit quality (on LLaMA-2-7B WikiText2, AQLM 2-bit ≈ 6.64 PPL @ 2.02 bits vs QuIP# 2-bit ≈ 8.22 PPL vs FP16 5.12 — AQLM is “strictly Pareto-optimal versus all prior 2-bit methods”), but its codebooks are too large for L1 cache so inference is often slower than fp16, and per-model optimization is unavoidable and infeasible for 744B on a Mac.
- **QTIP — strong future option.** Trellis-coded quantization with incoherence processing; compute-based Gaussian codes (1MAD/3INST) need *no* stored value codebook and beat QuIP#/AQLM quality, but the bitshift-trellis decode is sequential per step and a poorer first fit for a simple Metal gather kernel.
- **llama.cpp IQ1_S/IQ1_M — hardcoded grids, reusable but ggml-coupled.** IQ1_S = 1.5625 bpw (super-blocks of 8 blocks × 32 weights); IQ1_M = 1.75 bpw (16 blocks × 16 weights). `block_iq1_s` = fp16 super-block scale + `qs[QK_K/8]` + `scales[QK_K/16]` over 256-weight super-blocks;  uses a fixed `iq1s_grid` lattice. The lattice idea is importable but the storage is ggml-opinionated — as ikawrakow notes (llama.cpp Discussion #5063): “ggml … is quite opinionated when it comes to how tensor data must be organized and stored. This inevitably leads to difficulties when one wants to integrate things such as variable length/per tensor codebooks … This is one of the reasons why the i-quants here use fixed rather than per tensor codebooks.” (You explicitly rejected a GGUF port.)
- **QuaRot/SpinQuant** contribute the rotation/incoherence machinery (Hadamard/learned orthogonal) that VQ depends on; reuse `mx.hadamard_transform` (already in MLX).

**Verdict:** import the QuIP# E8 lattice codebook to *defer the learning pipeline entirely* for the codebook, and add per-model Hessian calibration only to recover quality. This is the only path that is both sub-2-bit-quality-capable and cache/kernel-friendly on Metal.

## Details

### Component architecture (text diagram)

```
                          ┌──────────────────────────────────────────────┐
   safetensors (mmap)     │  GLM-5.2-VQ checkpoint (on disk, ~110-181 GB) │
   no bf16 materialize    │  • global E8 codebook table (1 KiB, shared)  │
                          │  • per-layer RHT sign vectors SU,SV (packed) │
                          │  • routed-expert index streams (uint8/uint16)│
                          │  • per-expert/per-row scales (fp16)          │
                          │  • attention/router/shared/embed (6-8bit aff)│
                          └───────────────┬──────────────────────────────┘
                                          │ lazy load (mmap, madvise)
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                 ▼                                 ▼
  ┌───────────┐                   ┌──────────────┐                  ┌──────────────┐
  │ MLA attn  │  6-8b affine      │ Router/gate  │ fp16/8b          │ Shared expert│ 6-8b
  │ + DSA idx │  (mx.quantized_   │ top-8 of 256 │                  │ (dense FFN)  │
  └─────┬─────┘   matmul)         └──────┬───────┘                  └──────┬───────┘
        │                                │ rhs_indices (sorted)            │
        │                                ▼                                 │
        │                   ┌─────────────────────────────┐                │
        └──────────────────►│  gather_vqmm  (NEW custom   │◄───────────────┘
            hidden states   │  Metal kernel via           │  + global codebook (threadgroup)
                            │  mx.fast.metal_kernel)      │
                            │  fused: gather expert →     │
                            │  decode E8P codeword →      │
                            │  matmul, NO dense weights   │
                            └─────────────────────────────┘
```

### Part 2.1 — Quantization scheme (concrete decision)

- **Vector dimension:** 8 (E8 lattice), matching QuIP# E8P. Reason: E8 achieves optimal 8-D unit-ball packing and its symmetry compresses the 2^16 codebook to a 256-entry 1 KiB table — the only family that simultaneously gives sub-2-bit quality *and* fits Metal threadgroup memory.
- **Codebook(s):** one **global shared** table (1 KiB) reused by all 19,200 experts and all layers (E8P adds ≪0.01 bits/weight amortized). Two operating points:
  - **VQ-1.0 (fits-in-RAM):** 1-bit E8 codebook, 8-bit codeword per 8-D vector → **1.0 bpw**.
  - **VQ-2.0 (quality, needs paging):** E8P 2-bit, 16-bit codeword per 8-D vector → **2.0 bpw**.
  - **VQ-1.5 (balanced):** a 12-bit-per-8D padded lattice (4096-entry, 8 KiB table) → **1.5 bpw**.
- **Bits-per-index:** 8 (1.0 bpw), 12 (1.5 bpw), or 16 (2.0 bpw) per 8 weights.
- **Scale/outlier handling:** Random Hadamard Transform (RHT) incoherence processing (reuse `mx.hadamard_transform` + stored ±1 sign vectors SU/SV) makes weights approximately i.i.d. Gaussian so a single per-row (or per-group-of-256) fp16 scale ≈ 0.9× optimal Gaussian scale suffices; no separate outlier channels needed (RHT is what removes the outliers).
- **Mixed-precision allocation policy:**

|Component                           |~Params    |Precision              |Rationale                              |
|------------------------------------|-----------|-----------------------|---------------------------------------|
|Routed-expert gate/up/down (FFN)    |~725B (97%)|**VQ 1.0–2.0 bpw (E8)**|Bulk; tolerant to VQ after RHT         |
|MLA q/kv proj + o_proj              |~6B        |6-bit affine           |Attention coherence-critical           |
|DSA indexer (index_n_heads×head_dim)|~1B        |8-bit affine           |Sparse-attn index selection sensitivity|
|Router/gate (sigmoid scoring)       |<0.5B      |fp16 / 8-bit           |Routing errors cascade; keep high      |
|Shared expert (1/layer)             |~2.8B      |6-bit affine           |Always active, high marginal value/byte|
|Dense FFN (layers 0–2)              |~0.7B      |6-bit affine           |Few params, set output distribution    |
|Embeddings + LM head                |~1.9B      |6-bit affine           |Quality/byte favorable                 |
|LayerNorms / RMSNorm                |tiny       |fp16/fp32              |Never quantize                         |

- **Bit-budget arithmetic (VQ-1.0 target):**
  - bf16 baseline: 744e9 × 2 B = **1,488 GB**.
  - Experts at 1.0 bpw: 725e9 × 1/8 B = **90.6 GB** (+ scales: one fp16 per 8-D group at group 256 ⇒ 725e9/256×2 B ≈ 5.7 GB; tighten with group 512 to ~2.8 GB).
  - Non-experts at ~6.5-bit avg: 19e9 × 6.5/8 B ≈ **15.4 GB**.
  - **Total resident ≈ 90.6 + ~4 + 15.4 ≈ 110 GB** → fits a 128 GB Mac with ~15 GB headroom for KV cache + activations. **Effective model-wide ≈ 1.18 bpw; ≈ 92.6% reduction vs bf16.**
  - VQ-2.0 variant: experts 725e9 × 2/8 = **181 GB** ⇒ exceeds 128 GB ⇒ **must page cold experts from SSD** (see MoE section). Effective ≈ 2.1 bpw, ~87% reduction.
- **Tradeoff stated:** E8P (uniform lattice) vs AQLM (learned) — we trade ~the last fraction of a bit of accuracy and the option of input-adaptive codebooks **for** zero codebook-learning cost, a 1 KiB cache-resident table, and a <4-instruction decode. We trade VQ-1.0’s lower quality **for** full residency; VQ-2.0’s higher quality **for** SSD paging latency.

### Part 2.2 — Storage format (on-disk and in-memory)

Use a **safetensors-compatible** container so MLX’s lazy mmap loader works unmodified; add a `quantization_config.json` describing the VQ mode. Per quantized expert weight `W ∈ R^{out×in}` (e.g. 2048×6144 for down_proj):

```
Tensor: model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.codes
  dtype : uint16 (VQ-2.0) | uint8 (VQ-1.0)
  shape : [out, in/8]            # one codeword per 8 input dims
  layout: row-major; codewords for consecutive input dims are CONTIGUOUS
          -> coalesced reads across the in-dimension
Tensor: ...{proj}.scales
  dtype : float16
  shape : [out, in/group]        # group=256 or 512; per-row-group scale
Global (stored once per file, referenced by config):
  e8_codebook : uint32[256]      # 1 KiB packed-abs table (8 nibbles/entry)
Per-layer (shared by all experts in the layer):
  attn/.../rht_SU : packed int8 sign vector, length=in  (or store a uint32 seed)
  attn/.../rht_SV : packed int8 sign vector, length=out
```

Byte layout of one E8P codeword (16-bit, little-endian):

```
 bit:  15 .......... 8 | 7 ........ 1 | 0
       [ base index   ] [ 7 signs    ] [shift ±1/4]
       8 bits -> S[256]   7 stored      8th sign = parity(7 signs)⊕parity(S entry)
```

**Avoiding RAM inflation at load — the hard constraint.**

1. **mmap, never densify.** The codes/scales tensors are mmap’d from safetensors (MLX already loads via a lazy mmap loader). Resident set = packed codes + scales + 1 KiB codebook. No `dequantize()` is ever called at load (that path — which `dequantize_model()` uses to reconstruct `weight = dequantize(...)` — is explicitly bypassed for VQ layers).
1. **Decode happens inside the matmul kernel**, per-tile, into registers/threadgroup, and is discarded — a dense bf16 expert is *never* allocated. This is the same principle as MLX’s fused `affine_qmm_t` (fused dequant+matmul) and QuIP#’s fused decode-into-`mma.sync`.
1. **MLX machinery changes needed.** MLX’s loader will read `codes` as a plain `uint16`/`uint8` array (fine). The gap: `nn.QuantizedLinear`/`QuantizedSwitchLinear` assume the affine triplet `(weight:uint32, scales, biases)`. You introduce `QuantizedVQLinear`/`QuantizedVQSwitchLinear` modules whose params are `(codes, scales, codebook_ref)` and whose `__call__` dispatches to the custom `vq_matmul`/`gather_vqmm`. Critically, quantized tensors must **skip dtype casting** on load (casting `uint16` codes would corrupt them) — reuse the existing “quantized weights skip the dtype cast” rule.
1. **MLX cannot today** express this as a first-class `mode="e8p"` from Python. **Workaround:** ship the custom modules + `mx.fast.metal_kernel` op in a small Python package (the proven pattern used by mlx-vlm’s TurboQuant and the community VQ-KV work); later upstream a C++ `QuantizeMode::e8p` into `quantized.cpp` for the truly “native” type.

### Part 2.3 — The Metal kernel (technical core)

**Goal:** a fused **gather → decode → matmul** that, for each output element, reads packed codewords, reconstructs 8 fp16 weights from the 1 KiB codebook in registers, multiplies by the 8 activations, and accumulates — never writing a dense weight tile to device memory.

**Codebook residency.** The 256-entry E8 table is 1 KiB. Apple GPUs expose **32 KB threadgroup memory**. Load the table once per threadgroup into `threadgroup uint32 cb[256]` (1 KiB). Optional bank-conflict mitigation by duplication: QuIP#’s CUDA path duplicates 32× (= 32 KB) to kill conflicts,  but on Metal that exhausts the entire 32 KB budget and blocks any activation/accumulator tiling — recommend **4–8× duplication (4–8 KB)** as the balance, leaving ≥24 KB for activation tiles. State the tradeoff: more duplication → fewer LUT bank conflicts but less room for the activation tile / lower occupancy.

**Two kernels (mirror MLX’s qmv vs qmm split):**

*(a) Decode-matvec (`vq_qmv`), the M=1 decode hot path.* Each simdgroup owns a set of output rows; the 32 lanes stride over the `in/8` codeword axis. Pattern follows MLX’s fast qmv (quad_sum over quadgroups, each thread processing many values to saturate bandwidth — qmv is memory-bound, so the design target is coalesced codeword reads, not ALU).

```metal
// threadgroup uint32 cb[256] preloaded from global e8_codebook
// x_tg[in] : activations in threadgroup memory (broadcast)
// codes : device const uint16* , row-major [out, in/8]
// per lane: accumulate partial dot for assigned output row(s)
float acc = 0.0f;
uint base = row * (IN/8);
for (uint c = lane; c < IN/8; c += 32) {           // coalesced: lanes read adjacent codewords
    uint16_t cw  = codes[base + c];
    uint     idx = cw >> 8;                         // 8-bit table index
    uint     sgn = (cw >> 1) & 0x7F;                // 7 stored signs
    uint     shft = cw & 0x1;                       // ±1/4
    uint     packed = cb[idx];                      // 1 lookup -> 8 nibbles
    // parity of stored signs ⊕ parity(packed) -> 8th sign  (E8 trick)
    uint p = popcount(sgn) ^ (parity_of_nibbles(packed) & 1);
    uint signs8 = sgn | (p << 7);
    #pragma unroll
    for (int d = 0; d < 8; ++d) {                   // unpack 8 weights
        float wabs = ((float)((packed >> (4*d)) & 0xF) - 8.0f) * 0.5f;
        float w    = ((signs8 >> d) & 1) ? -wabs : wabs;
        w += shft ? 0.25f : -0.25f;
        acc += w * x_tg[c*8 + d];                   // ~4 instr/weight incl. multiply
    }
}
acc = quad_sum(acc);                                // simd reduction across lanes
if (lane_is_reducer) out[row] = scale[row_group] * acc;
```

*Inner-loop / gather-latency note:* the decode is pure ALU + one **threadgroup** LUT read (not device gather), so the per-weight cost is ~4 instructions and the LUT access never leaves on-chip memory — this is precisely why QuIP# sustains >50% of peak memory bandwidth and why E8P beats AQLM (whose >1 MiB codebook spills L1, causing the cache misses that make AQLM slower than fp16). The kernel stays **memory-bandwidth-bound on the codeword stream**, which is the desired regime for decode. QuIP#’s real CUDA kernel computes the parity by an XOR-shift cascade (`s ^= s>>4; s ^= s>>8; s ^= s>>16`) and applies signs via LOP3 bit-ops fused directly into `mma.sync.aligned.m16n8k16` — the Metal analog uses `simd`-level bit ops feeding `simdgroup_multiply_accumulate`.

*(b) Decode-matmul (`vq_qmm`), prompt/prefill (M large).* Decode a K×N weight tile from codewords into a `threadgroup half` tile, then run `simdgroup_half8x8`/`simdgroup_float8x8` MMA (`simdgroup_multiply_accumulate`) exactly like MLX’s `qmm` (which today accumulates in fp32).  On M5-class GPUs use the Metal 4 / NAX tensor path (WWDC26 `matmul2d` cooperative tensors) which can ingest quantized data directly; on older Apple GPUs keep the steel simdgroup-MMA fallback. This mirrors QuIP#’s strategy of fusing decode into `mma.sync.m16n8k16`.

**Thread/threadgroup tiling.** vq_qmv: threadgroup = (32 × R) where R = rows per threadgroup (start R=8); grid covers `out_rows × experts`. vq_qmm: 32×32 or 64×64 output tiles, K-loop decodes 8-wide chunks; split-K for large `in` (6144) as MLX’s qvm split-K does.

**Memory coalescing.** Because codewords for consecutive input dims are contiguous `uint16`, the `codes[base + c]` reads by adjacent lanes are fully coalesced; activations are broadcast from threadgroup memory; scales are read once per row-group.

**Dispatch through MLX.** Register the body with `mx.fast.metal_kernel(name="vq_qmv", input_names=["codes","x","cb","scale"], output_names=["out"], source=...)`, build it once and reuse (avoid per-call JIT). For autodiff-free inference this is sufficient; `mx.fast.metal_kernel` also supports atomic outputs and custom VJP if training is later needed. 

**Validation vs reference.** For every shape, compare `vq_qmv/qmm` output to a pure-MLX reference `dequantize_e8p(codes, cb, scale) @ x` (which *does* materialize the dense weight, only in the test harness) and require cosine ≥ 0.99999 / max-abs-error within VQ noise — the same correctness bar the MLX community used for the fused TurboQuant KV kernel (“cosine 1.0 vs dequantize path”). Use QuIP#’s reference `get_full_grid()` (builds the 2^16×8 table) to cross-check the parity/shift decode against the upstream implementation bit-for-bit.

### Part 2.4 — MoE integration

- **Interface to router/gate.** GLM-5.2 routes top-8 of 256 via sigmoid scoring with `routed_scaling_factor` 2.5 and `norm_topk_prob`. The router stays fp16/8-bit. Its top-k indices drive a `gather_vqmm` that is the VQ analog of MLX’s `gather_qmm`: `y[t] = Σ_{e∈topk(t)} g_{t,e} · vq_matmul(x[t], codes[e], scale[e], CB)`.
- **Sorted-indices batching.** Reuse MLX’s `gather_qmm_rhs` sorted-indices optimization: argsort tokens by expert id so consecutive tokens hitting the same expert share the loaded codeword tile and the (already threadgroup-resident) codebook — amortizing the codeword stream across a token batch.
- **Per-expert vs shared codebook.** Use **one global codebook** for all experts (E8 is data-free, so per-expert codebooks would buy almost nothing while destroying the 1 KiB-fits-cache property and blowing up storage). Only the **scales** are per-expert/per-row. Tradeoff: a shared lattice slightly underfits any single expert’s post-RHT distribution, but RHT makes every expert ≈ the same Gaussian, so the loss is minimal and the cache/throughput win is large.
- **Expert offloading / paging (only for VQ-2.0).** At 1.0–1.25 bpw the model is fully resident — *no paging needed*, which is the cleanest design. At 2.0 bpw (181 GB > 128 GB) compound VQ with paging: keep non-expert weights + the global codebook + hot experts resident, mmap the cold expert index streams, and use a **sliding-window prefetch** (materialize layer L+1’s selected experts while computing layer L) with `madvise(MADV_WILLNEED/DONTNEED)` — the approach proven by Flash-MoE (which streams only active experts via `pread` and lets the macOS page cache manage residency, reporting a ~71% cache hit rate and ~4–7 tok/s on a 209 GB model on an M3 Max) and JANGPress (mmap + per-token `MADV_DONTNEED` over cold routed-expert pages to serve >RAM MoE bundles on a single Mac, cited running 167 GB Kimi-K2.6 on a 128 GB Mac). Because VQ already shrinks each expert ~8–16×, the bytes paged per token drop proportionally, so VQ + paging compounds favorably: at 1.0 bpw each expert is ~4.7 MB, so even a fully cold top-8 set is ~38 MB/layer.

### Part 2.5 — The quantizer pipeline

**Import-existing-codebook path (recommended v1, defers learning):**

1. **RHT incoherence (data-free).** For each quantized matrix, apply `W ← SU · Had · W · Had · SV` with random ±1 sign vectors (store SU/SV or a seed). Uses `mx.hadamard_transform`. No data.
1. **Rounding.** v1: **RTN** onto the E8 lattice (nearest codeword), fully data-free — gets you an end-to-end-correct model immediately. v2: **BlockLDLQ** adaptive rounding using a per-layer Hessian `H ≈ E[xx^T]` estimated from calibration activations. Per QuIP# §F.7: “we can quantize Llama 2 70B without fine tuning in under 10 GPU-hours and with fine tuning in around 100 GPU-hours. Both numbers do not include Hessian generation.”
1. **Calibration data.** 256–4096 sequences (per AQLM ablations, “Gains saturate around 2,000 calibration sequences (useful range: 512–4,096)”; QuIP# fine-tunes on 256 RedPajama sequences). **MoE caveat:** with 19,200 experts and top-8 routing, each expert sees only a sliver of tokens; you need enough calibration tokens (and *natural* top-k routing, not forced all-expert activation) so every expert’s Hessian/scale is well-estimated — the same lesson the GLM-5 NVFP4 quantizer reported (“calibration on a much larger number of samples to ensure broad expert coverage through natural routing”). Budget tens of millions of calibration tokens.
1. **Optional fine-tuning** (QuIP#-style intra-block) to recover the last bit of quality; the E8 codebook itself is *not* trained. Per QuIP# (PMC12395268): “Our fine tuning method runs on a small development set and can be performed in around 50 GPU-hours for a 70B parameter model.”

**Compute cost estimate.** The codebook is free (imported). The cost is (a) RHT — seconds/layer, and (b) Hessian estimation — forward passes over calibration data. On a single high-RAM Mac (M3 Ultra/M4 Max class) or a rented A100/H100, expect the dominant cost to be running ~10⁷–10⁸ calibration tokens through a 744B MoE to cover all experts: order **single-digit to low-tens of GPU-days** for full BlockLDLQ+calibration at 744B (extrapolating QuIP#’s <10 GPU-h/70B-no-FT and ~50 GPU-h/70B fine-tuning, and noting MoE expert-coverage overhead). The **data-free RTN v1 path is hours**, enabling an immediate working model before any calibration.

**Why not AQLM here:** learned codebooks alone cost “about 1 day on a single A100” for 7B and “10-14 days” for 70B; they do not amortize a shared table — multiply by the 744B scale and it is impractical, and the >1 MiB codebooks would not fit Metal threadgroup memory anyway.

### Part 2.6 — Build sequence, risk, effort

**Dependency-ordered milestones (≈ 1 engineer; weeks are calendar effort):**

|#|Milestone                                                                                                             |Depends on|Effort|Risk                                             |
|-|----------------------------------------------------------------------------------------------------------------------|----------|------|-------------------------------------------------|
|1|Pure-MLX/numpy E8P encode+decode reference (`dequantize_e8p`); correctness oracle                                     |—         |1 wk  |Low                                              |
|2|`mx.fast.metal_kernel` **vq_qmv** for one Linear; validate cosine ≥ 0.99999 vs #1                                     |1         |2–3 wk|**High** (kernel correctness, parity trick, perf)|
|3|safetensors VQ container + `quantization_config.json` + mmap lazy load with **no densify**; `QuantizedVQLinear` module|2         |2 wk  |Med (MLX loader assumes affine triplet)          |
|4|End-to-end on a small dense model (quantize a 1–3B to E8P, generate)                                                  |2,3       |1 wk  |Low                                              |
|5|**gather_vqmm** + sorted-index batching; integrate router; test on small MoE (GLM-4.5-Air / Qwen3-MoE)                |4         |3 wk  |**High** (MoE routing × VQ interplay)            |
|6|Scale to GLM-5.2: sharded conversion, mixed-precision policy, expert paging (VQ-2.0)                                  |5         |3 wk  |**High** (181 GB paging perf, SSD bandwidth)     |
|7|Quantizer: RHT + RTN (data-free) v1, then BlockLDLQ + calibration v2                                                  |3         |4 wk  |Med-High (MoE expert coverage, quality)          |
|8|`vq_qmm` MMA prefill path + perf tuning (duplication factor, tiling, NAX)                                             |5         |3 wk  |Med (bandwidth vs ALU balance)                   |
|9|(Optional) Upstream C++ `QuantizeMode::e8p` into MLX `quantized.cpp` for native type                                  |8         |3 wk  |Med                                              |

**Correctness-first ordering:** #1–#4 get a *small model* running end-to-end on the real kernel before any MoE or scale complexity; #5 adds MoE on a small model; only #6 touches 744B. Quality work (#7) proceeds in parallel because the data-free RTN path already produces a runnable (if lossy) model.

**Highest-risk components:** (1) the **vq_qmv/qmm Metal kernel** — parity decode bugs are silent and the perf must stay bandwidth-bound; (2) **sub-2-bit quality on MoE experts** — 1.0 bpw may degrade GLM-5.2’s agentic/coding quality more than acceptable, in which case fall back to VQ-1.5/2.0; (3) **MLX’s lack of a native VQ mode** — the custom-op route works but is not “native” until the C++ fork (#9); (4) **VQ-2.0 paging throughput** on consumer SSDs.

## Recommendations

1. **Start now with the data-free path to de-risk the kernel (weeks 1–4).** Implement the E8P reference + `vq_qmv` + safetensors/mmap loader, and prove a small dense model generates correctly on the fused kernel before writing any MoE code. **Go/no-go gate:** cosine ≥ 0.99999 vs the dequantize reference and `vq_qmv` sustaining ≥ 50% of measured memory bandwidth (QuIP#’s achieved bar). If perf is below ~30% bandwidth, revisit codebook-duplication factor and tiling before scaling.
1. **Default the experts to VQ-1.0 (1.0 bpw, 8-bit codeword) for a 128 GB Mac** so the whole model is resident (~110 GB) and you avoid paging entirely. Keep attention/router/shared/embeddings at 6–8-bit affine via stock `mx.quantized_matmul`. **Threshold to switch to VQ-1.5/2.0:** if perplexity/agentic-eval degradation at 1.0 bpw exceeds your task tolerance (validate on SWE-bench-style coding tasks since GLM-5.2 is coding-first), step to 1.5 bpw (still ~125 GB resident) and only adopt 2.0 bpw + SSD paging if quality still misses.
1. **Import the QuIP# E8 codebook; do not build an AQLM-style learner.** Ship RTN-on-lattice first; add BlockLDLQ + Hessian calibration only if RTN quality is insufficient. **Threshold:** if data-free RTN already lands within your eval tolerance, skip calibration entirely and save the GPU-days.
1. **Use one global codebook and per-expert scales**; route MoE through a `gather_vqmm` with sorted-index batching. Reserve expert paging strictly for the VQ-2.0 configuration.
1. **Defer the MLX C++ fork (native `mode="e8p"`) to the end.** Build on `mx.fast.metal_kernel` + custom modules first (proven by the MLX TurboQuant/VQ-KV community work); upstream the native type once the format and kernel are stable.
1. **Plan calibration around MoE expert coverage**, not token count alone: drive natural top-k routing over tens of millions of tokens so all 19,200 experts get well-estimated scales/Hessians.

## Caveats

- **GLM-5.2 vs GLM-5 specs:** GLM-5.2’s per-field `config.json` was not directly fetchable in this research; the per-tensor shapes above are the confirmed **GLM-5** `config.json` (`glm_moe_dsa`, 744B), which the GLM-5.2 NVFP4 model card and GLM-5 technical report indicate GLM-5.2 shares. Secondary blogs variously cite **744B vs 753B total**; treat 744B (official config / report) as authoritative and re-verify the exact MoE-layer count and any GLM-5.2 routing changes against the published GLM-5.2 `config.json` before finalizing the bit budget. The exact number of MoE layers (used as ~75 here; config `num_hidden_layers`=78 with 3 dense + 1 MTP) should be confirmed per the released checkpoint.
- **Performance is projected, not measured.** The >50% bandwidth and <4-instruction/weight figures are QuIP#’s on NVIDIA GPUs; Apple-GPU throughput for this kernel is unproven and community reports note IQ1-style lattice quants have historically been *slow* on Apple Silicon. Treat all throughput as a target to validate, not a guarantee.
- **Sub-2-bit quality at MoE scale is not established for GLM-5.2.** QuIP# 2-bit quality numbers are for dense Llama-1/2 (e.g., LLaMA-2-7B 2-bit ≈ 8.22 WikiText2 PPL vs FP16 5.12); 1.0 bpw on 256-expert MoE is more aggressive and may need RVQ/fine-tuning to be usable.
- **MLX is a moving target.** Quantization modes, `gather_qmm` internals, and the custom-kernel API have changed across releases; verify against the MLX version you build on. A native VQ mode requires a C++ change that upstream may or may not accept.
- **DSA/MLA attention quantization** is left at 6–8 bit here on the assumption it is coherence-critical; the DSA indexer in particular is unstudied under quantization and may need fp16.
- **macOS unified-memory pressure:** even at ~110 GB resident you must leave headroom for KV cache (GLM-5.2’s 1M-token context is enormous — full-context KV will not fit; cap `max_model_len` accordingly) and the OS; budget conservatively.