# V4.1-Flash REAP on one GPU: design

Date: 2026-11-06. This replaces the TP8 plan in
`2026-11-05-v41-runpod-reap-runbook.md`, which needed an 8×H200 pod that
Runpod could not place. Every figure below comes from the reference config and
code at revision `1bae0e10`. I fetched `inference/{config,model,kernel,convert,engram}.py`;
`model.py` has sha256 `e5decd76…8b8d2d`, which matches the pin in `collect.py`.

## Why 8 GPUs was never about compute

| Quantity | Value |
|---|---|
| Active parameters per token | about 15.6B: 374M per layer × 40 layers, plus the head |
| Forward cost | about 31 GFLOP per token |
| Total work | 3.2M calibration tokens, plus 0.45M eval tokens × (1 teacher + 4 students) ≈ 5.5M token passes ≈ 1.7e17 FLOP |
| Work on one H100 | 15 minutes at 200 TFLOPS effective; 1 hour at 50 TFLOPS |
| Weights | 510 GB on disk. Routed experts are 6.8 GB per layer in FP4, Engram tables are about 196 GB. |

The 8×H200 pod existed only to hold all 510 GB in GPU memory at once. The work
itself fits on one GPU in under an hour. So the fix is to **stream weights
through one GPU, one layer at a time, over large token batches**. Holding every
weight resident is what forced the 8-GPU pod.

## Design

1. **Load layer weights straight from the HF shards.** There is no `convert.py`
   step and no TP8 copy. With one GPU, `convert.py` only renames tensors,
   dequantizes `wo_a` to bf16 and views the experts as `float4_e2m1fn_x2`. That
   is about 20 lines, applied per tensor as it loads. This saves the hour of
   conversion, the 510 GB TP8 copy, and the need for more than 550 GB of RAM.
2. **Run the reference code unchanged, one layer at a time.** Build the
   reference `Block` modules with `world_size=1`. For each chunk of records
   (whole records, so attention sees full context):
   - keep the chunk's hyper-connection residual in GPU memory, as
     `[tokens, 4, 5120]` bf16 (40 KB per token);
   - for each layer, load its weights, prefetching the next layer's;
   - run attention one micro-batch of records at a time;
   - run the MoE over the whole chunk at once, so each expert sees about 15K
     tokens instead of about 100, which makes the expert GEMMs efficient.

   The REAP, imatrix and route hooks in `make_moe_forward` stay as they are.
3. **Save and restore the cross-layer attention state.** Layers pass
   compressed KV, indexer keys, top-k indices and candidate blocks to later
   layers through the module-level `shared_attn` object. KV comes from source
   layers 2, 8, 14 and 20; the indexer from 2, 8, 14, 20, 24, 28, 32 and 36;
   candidates from layer 20 into every layer after it. Done one layer at a
   time, that state has to be saved per micro-batch and restored before the
   next layer. The candidate mask is the large part, up to about 16 KB per
   token at 16K context. It gets bit-packed, or kept in host RAM.
4. **Gather the Engram rows on the host, at run time.** The rows Engram
   fetches depend only on the token ids (`NgramHashState` → `embed(hash_ids)`);
   the gate needs the hidden state, but the lookup does not. A stand-in for
   `ParallelEngramEmbedding` memory-maps the two 196 GB tables straight from the
   shard files, gathers the FP8 rows and scales a chunk needs (24 × 256 values
   per token per Engram layer) and dequantizes them exactly as the module does.
   The tables never reach the GPU. I first planned to precompute the rows in
   the pack stage (about 48 GB on disk); the run-time gather does the same
   thing with no extra stage and no extra disk.
5. **Do the whole job in two sweeps over the weights.**
   - **Sweep A, calibration** (4 chunks on an 80 GB GPU, 2 on an H200): writes
     the REAP stats, imatrix and routes, then `finalize-reap` and
     `reap-select` produce the survivors for 256/192/128/96.
   - **Sweep B, eval:** the teacher and the 4 masked-router students run side
     by side through each layer, so each layer is read once for all 5 streams.
     This writes the teacher's top-2048 logits and the student
     KL / NLL change / top-1 agreement, in the same npz schema as now. That
     replaces the separate student `torchrun` pass.

   Every chunk is one pass over the backbone weights, about 320 GB without the
   Engram tables. A chunk holds `--chunk-tokens` resident tokens (default 512K,
   counting each stream), so one 80 GB GPU makes about 7 calibration passes and
   5 eval passes. That is about 4 TB of reads. From NVMe this is 20-30 minutes,
   overlapped with compute by a one-layer prefetch, and less when the pod's
   page cache holds the shards.
6. **Use as many GPUs as are free, data-parallel by chunk.** Chunks are
   independent, so any pod with 1 to 8 GPUs of the Hopper or Blackwell
   class (80 GB or more) works. The stats are sums, so the shards add together
   exactly as the per-session npz files do today.
7. **Skip the MTP drafter.** It isn't needed for backbone REAP, and the pruner
   already leaves `mtp.*` alone.

## REAP-specific streamlining

- **Calibration budget:** the per-session npz files already let me compute a
  split-half survivor Jaccard offline, at no extra cost. If two halves of the
  calibration set pick nearly the same survivors (Jaccard ≥ 0.98 at N=96), then
  3.2M tokens is more than needed and the next run can use fewer.
- **Candidates:** all four are cheap now. Each extra student costs memory
  during Sweep B, not another pass over the weights.
- **E8P imatrix from pruned routing:** the teacher imatrix counts traffic
  under the full 384-expert routing. After pruning, the surviving experts take
  over the pruned experts' tokens. Sweep B can also write a pruned-routing
  imatrix for 192 and 96 at no extra cost. The catch is that it comes from the
  eval split, so it would leak a little into the KL measurement if it were
  used to fit E8P. I'll write it, but not use it by default.

## What changes in the kit

| File | Change |
|---|---|
| `stream.py` (new) | `HFCheckpoint` (per-layer loads with `convert.py`'s fixups), `EngramRows` (host gather), `LayerStreamer` (the reference `Block`s one layer at a time, with `shared_attn` saved and restored per record). |
| `torch_kernel.py` (new) | Plain-torch stand-ins for `kernel.py`, for the CPU tests. |
| `collect.py` | The `torchrun` driver is replaced by `--sweep calibration|eval`, `--shard I/N` and `--chunk-tokens`. `SessionStats`, the MoE hook, the head capture and the npz schemas are unchanged. |
| `build_pack.py` | No change. |
| `pod.sh` | `convert` and the TP8 directory are gone. `gpu` runs one `collect.py` per GPU for sweep A, then select, then sweep B. |
| `prune.py`, `score.py`, `finalize-reap`, `reap-select` | No change. |

## Validation before any spend

1. **Run the real `model.py` on the CPU, at tiny size.** Replace
   `kernel.py`'s tilelang functions with plain-torch versions (`act_quant`,
   `fp4_act_quant`, `fp8_gemm`, `fp4_gemm`, `sparse_attn`,
   `hc_split_sinkhorn`; the repo already has a torch Sinkhorn). Use a tiny
   random `ModelArgs` that still has hyper-connections, KV and indexer sources,
   a candidate source and Engram. The test then checks that the
   one-layer-at-a-time driver matches `Transformer.forward` logits and REAP
   stats with:
   - one record per micro-batch;
   - several records per micro-batch;
   - records split across chunks.

   This is the test that catches mistakes in saving and restoring
   `shared_attn`.

   **Status (2026-11-06): done.** `tests/test_runpod_v41_stream.py` runs the
   pinned `model.py` and `engram.py` (copied to `tests/fixtures/v41_reference/`)
   on a 6-layer toy with every feature above. The streamed hidden state equals
   `Transformer.forward` bit for bit, with one MoE batch per chunk and with one
   per record, and a record's output does not depend on its chunk-mates. Breaking
   any one `shared_attn` field, the clone of block buffers, or the Engram lookup
   fails the test. Both sweeps are checked too:
   - calibration REAP, imatrix and route stats match the reference per session
     (route counts exactly, sums to 1e-6);
   - eval teacher logits match;
   - a keep-all student scores KL 0;
   - a keep-4-of-8 student's NLL matches the reference with that router mask.
2. **First 10 minutes on the pod:** run the same equivalence test with the real
   tilelang kernels at tiny size, then one real 16K record through all 40
   layers. Check the teacher NLL is sensible, and measure GPU memory and tok/s.
   Stop if anything is off.

## Cost

These prices are from the Runpod catalog read of 2026-11-06. 1×H100 SXM costs
$2.69/h, and 1×H200 costs $3.59/h on community cloud or $5.29/h on secure.

| Stage | Time on 1×H100 |
|---|---|
| fetch and pack, in parallel | 1-3 h |
| sweeps A and B | about 1-2 h |
| prune 192 and 96 | about 1 h |

The total is about 3-6 h, roughly **$10-25**, against $125-340 for the TP8
plan. The disk needed is about 1.2 TB:

| Item | Size |
|---|---|
| HF source | 510 GB |
| out | about 30 GB |
| REAP-192 | 366 GB |
| REAP-96 | 293 GB |

## Risks

- **GPU kernels:** the tilelang kernels are written for Hopper and Blackwell.
  H100 and H200 are the safe target. The RTX PRO 6000 (SM120) is not verified.
- **`shared_attn` state:** saving and restoring it is the subtle part, and the
  CPU equivalence test covers it.
- **Engram host gather:** this relies on `embed(hash_ids)` being a pure row
  gather. I checked the code: `ParallelEngramEmbedding.forward` gathers FP8
  rows and their scales, then dequantizes, and the CPU test runs the real
  module's math through the stand-in.
- **Indexer memory at 16K context:** the indexer's `bshd,btd->bsht` einsum is
  about 16 GB for one 16K record. It should fit on 80 GB, but the first pod
  minutes must check `max_gpu_gb` in the progress log before the full run.
