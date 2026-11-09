# DeepSeek-V4.1-Flash-REAP-75: local test plan (M4 Max)

Goal: run the published [jackmaxwil/DeepSeek-V4.1-Flash-REAP-75](https://huggingface.co/jackmaxwil/DeepSeek-V4.1-Flash-REAP-75)
on this Mac. First prove that the exported checkpoint computes what the Runpod eval measured. Then
produce text from it. Speed is a later question.

REAP-75 is the testable one. REAP-50's routed experts alone are 148 GiB, more than the machine's
128 GiB, so they cannot stay in memory.

## What is in the checkpoint

| Part | Size | How a local run touches it |
|---|---:|---|
| Engram n-gram tables (layers 1 and 14) | 189 GiB | sparse row gathers from the shard files on disk (memory-mapped), never loaded whole |
| Routed experts, 96 per layer, FP4 | 81 GiB | streamed one layer at a time |
| Vision tower | 2 GiB | not used |
| Attention, shared experts, embeddings, head | 1 GiB | per layer |

Disk is no longer a constraint: 948 GiB was free on 2026-11-08, after the cleanup.

## Why the torch reference path comes first, not MLX

V4.1 is not V4-Flash plus Engram. On top of Engram it adds:
- cross-layer KV sharing (`kv_source_layers`);
- a two-level indexer (`index_source_layers`, candidate blocks);
- per-layer compression ratios;
- DSpark heads.

The MLX runtime in `src/keep/quality/dsv4_*` and `src/keep/convert/dsv4_resident.py` covers none
of this.

The single-GPU driver already runs DeepSeek's own `inference/model.py` unchanged, one layer at a
time:
- `scripts/runpod/v41_reap/stream.py` already gathers Engram rows from the shards on the host.
- `scripts/runpod/v41_reap/torch_kernel.py` already ports `sparse_attn` and `hc_split_sinkhorn` to
  plain torch.

The only pieces missing on a Mac are the quantized GEMMs and the activation quantizers. Those are
the tilelang kernels, which are GPU-only.

So the first local run reuses the code that produced the published numbers. An MLX port is worth
it only once that run passes, and only if the speed matters.

## Steps

1. **Environment.**
   - What: a separate venv with `torch` (CPU/MPS wheel), `safetensors`, `tokenizers`, `sympy`,
     `numpy` and `huggingface_hub`. The repo `.venv` has no torch, and should stay that way.
   - Check: `python -c "import torch; torch.float8_e8m0fnu"` works.

2. **Download REAP-75.**
   - What: `hf download jackmaxwil/DeepSeek-V4.1-Flash-REAP-75 --local-dir ~/Models/v41-reap75`, then
     `shasum -a 256 -c SHA256SUMS` in that directory. That is 273 GiB; time depends on the line.
   - Check: every checksum passes.

3. **Pull the eval pack.**
   - What: copy only `pack/v41-pack.jsonl` and `pack/manifest.json` from the stopped pod
     `neckuzwv11dy2i`, probably tens of MB. Start the pod with zero GPUs for this and stop it right
     after.
   - Why: the stats dataset has teacher logits but no input token ids, and the parity check needs
     the exact eval inputs.
   - After: terminate the pod, with Jack's OK.
   - Check: `pack_sha256` in the manifest matches `run-config.json` (`6a354705…`).

4. **CPU stand-ins for the quantized kernels**, in `torch_kernel.py`:
   - `act_quant`: an exact port of the per-32-block FP8 activation quantization (`fp8_block_size = 32` in this release), including the
     e8m0 scale rounding. It quantizes and dequantizes, so the numerics match the pod.
   - `fp8_gemm`: dequantize the e4m3 weight with its 32x32 e8m0 block scales, then do a bf16/fp32
     matmul.
   - `fp4_gemm`: decode the e2m1 nibbles (low nibble first, as in `keep.convert.fp4_expert`) with
     per-32 e8m0 scales, then matmul. Dequantize one expert at a time, so a layer never holds more
     than one expert in bf16.
   - Check: extend `tests/test_runpod_v41_stream.py` with a toy model in fp8 and fp4. It must match
     the same model computed in plain bf16 from the dequantized weights to within float tolerance.

5. **Memory gate.**
   - What: run one real layer on one short eval record, under `/usr/bin/time -l`.
   - Expected peak: a few GiB of weights plus activations.
   - Check: the projected peak for the full run is written down. If it is over 24 GB, stop and ask
     Jack.

6. **Parity: the real test.**
   - What: run the exported REAP-75 layer-streamed over 2 or 3 of the shortest eval sessions. Score
     it against the teacher top-K in the stats dataset (`eval/sessions/*.npz`).
   - Compare with the pod's masked-router numbers for the same positions (`student-96/`).
   - Expectation: KL, NLL and top-1 agree per position up to the small difference between the
     tilelang kernels and the dequantize-then-matmul stand-ins.
   - Pass: |ΔKL| and |ΔNLL| averaged per session are small next to the REAP-75 numbers themselves
     (KL 1.35, NLL change −0.13 on all), and top-1 agreement is within a point.
   - If it passes, the "not run end to end" caveat comes off both model cards.
   - If it fails, the export (`prune.py`: router slicing, expert renumbering) is wrong. Then fix the
     export and republish both models.

7. **Text smoke test.**
   - What: greedy decode of about 20 tokens from a short coding prompt, layer-streamed.
   - Cost: each token re-reads about 84 GiB of non-Engram weights from the SSD, so expect tens of
     seconds per token. That is fine for a smoke test.
   - Option: keeping the experts in memory (about 85 GiB) would be faster, but it is over the 24 GB
     budget and needs Jack's OK.

## Speed path after parity: Rapid-MLX

[raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) (read at `7d2b20c`, 2026-11-08) already
runs DeepSeek-V4.1-Flash-1017 in MLX, from the same source revision (`1bae0e10`). What matters for
REAP-75:

- **Engram:** tables are re-quantized to MLX affine 4-bit, group 32 (about 61 GB per layer). Rows
  are read from mmap with MADV_RANDOM, through a 16,384-row LRU and a one-layer-ahead prefetch
  (`rapid_mlx/models/deepseek_v41_native/engram.py`).
- **Experts:** read from `config.json` (`n_routed_experts`, optional per-layer counts), so a
  pruned checkpoint fits the model code.
- **Conversion:** their loader reads only converted MLX builds. `convert.py` dequantizes our
  FP8/FP4 release format and requantizes it. It asserts expert ids are contiguous from 0, which
  `prune.py` already guarantees. But it buffers a whole quantized Engram table (about 61 GB) before
  saving, over the 24 GB budget, so it would need a streaming write first.
- **Fit (estimated from their byte counts):** with their 2/3-bit GPTQ recipe, REAP-75 is about
  58 GB resident plus 8 GB of DSpark drafter, with Engram on the SSD. That is over the 24 GB budget
  and needs Jack's OK.
- **Kernel gap at 96 experts:** their fused decode-MoE kernel requires `dim < 16 * n_experts`
  (`moe_decode.py:557`). At 96 experts that fails (5120 >= 1536), so it silently falls back to the
  stock path.
- **Wired memory:** `load.py:181-188` always raises the MLX wired limit, which conflicts with this
  repo's wired-memory policy. Their product lane also refuses machines under 224 GiB
  (`artifacts.py:57`).
- **Speed:** 34-49 tok/s greedy with DSpark speculative decoding, on an M3 Ultra 256 GB.
- **Correctness evidence:** teacher-forced KL 0.26 and top-1 85% against the BF16 source.

Never run their scripts or package from the checkout. `scripts/calibrate_deepseek_v41_reap.py`
executes `runtime/runtime.py` from a model directory.

## Out of scope for now

- An MLX port of V4.1 (Engram, KV sharing, indexer, compression).
- The vision path and DSpark/MTP speculative heads.
- REAP-50, which has to be tested on a GPU pod or with the experts streamed.
- Speed work.

## Decisions for Jack

- Go for the 273 GiB download (step 2).
- Starting the stopped pod for a few minutes with zero GPUs to pull the pack, then terminating it
  (step 3).
