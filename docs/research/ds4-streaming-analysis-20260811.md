# DwarfStar (ds4) streaming analysis — mining a C engine for the Wave 3 local teacher run

**Date:** 2026-08-11 · **Author:** Opus 5 (research pass) · **Status:** Analysis, no code changes

**Subject:** antirez's DwarfStar, cloned at `.repos/ds4` (HEAD `84cc882 rocm: enable
DSpark speculative decoding`). A single-binary C99 + Metal/CUDA/ROCm inference engine
purpose-built for DeepSeek-V4-Flash / V4-Pro / GLM-5.2, with SSD expert streaming,
hot-expert preload, multi-machine pipeline and tensor parallelism, and DSpark
speculative decoding.

**Question asked:** can ds4 itself produce the Wave 3 teacher outputs (top-2048 logits
per supervised position over 257 sessions / 11.7 M tokens / 81,920-token window; final
hidden states + MTP logits for 120 sessions; activation statistics for 40), or do we
mine its design for our own MLX streaming teacher runner?

All line citations are against the clone at the HEAD above.

---

## 0. Verdict, up front

**ds4 as a teacher engine: infeasible today, feasible after a bounded C patch — but the
patch is not the binding constraint. Residency is.**

Three findings dominate everything else:

1. **ds4 never materializes logits for more than one position per prefill chunk.**
   `uint32_t output_row = (uint32_t)n_tokens - 1u;` (`ds4.c:35047`, and identically at
   `ds4.c:34529`) — the LM head runs on exactly one row of the batch, and non-final
   chunks get `NULL` logits outright (`ds4.c:35257`). The only per-position path that
   exists, `--perplexity-file`, advances **one decode step per scored token**
   (`ds4_cli.c:1151`). At 11.7 M supervised tokens that is 11.7 M decode steps.

2. **There is a real, already-working multi-row logits path — capped at 16 rows.**
   `metal_graph_encode_output_head_batch` (`ds4.c:25653`) runs HC collapse + output norm
   + vocab projection over every row of a batch into `g->spec_logits`, and the caller
   reads back `n_tokens * DS4_N_VOCAB` floats (`ds4.c:35561-35565`). The buffer is
   allocated as `16 * DS4_N_VOCAB * sizeof(float)` (`ds4.c:17315-17316`). Widening that
   allocation and exposing it is the smallest change that gets prefill-speed teacher
   logits. This is the single most valuable thing in the repo for us.

3. **SSD-streamed prefill on a Mac is catastrophically slower than resident prefill —
   by 1–2 orders of magnitude, not the "still fast" the README implies.** The only
   quantitative streaming-vs-resident prefill comparison in the whole repo is
   `README.md:695`: **~3–5 tok/s streaming on one Mac** vs **~94 tok/s** resident across
   two, for GLM-5.2 IQ2_XXS at a 4,096-token context. Meanwhile resident Flash q2 on a
   single M5 Max does **398–790 tok/s** (`speed-bench/m5_max.csv:2-33`).

The consequence for Wave 3 is a strategy inversion:

> **The second Mac is not a fallback. It is the enabling requirement.** A single
> 128 GB M5 Max cannot hold a quantization-faithful V4-Flash artifact (~156 GB), so it
> must stream, and streaming puts the corpus prefill in the weeks range. Two Macs
> holding the model resident across a pipeline split put it in the **~6-hour** range.

Note also that the two-Mac mode we need is **pipeline parallelism over plain TCP**, not the
RDMA tensor-parallel mode the brief anticipated: TP replicates the KV cache, refuses
`--ssd-streaming` outright (`ds4_tp.c:496-499`), is CLI-only, and has no fault tolerance.
PP is the one with measured long-prompt prefill gains (**1.85x at 63,819 tokens**,
`README.md:438`) and real route-rebuild recovery.

**Recommendation in one line:** keep Wave 3's logit/hidden-state/MTP generation on the AWS
block — it needs no new engine work — but pull the **calibration/imatrix statistics** and
the **DSpark divergence probe** onto the local Mac, where ds4 does both *today with no code
changes*, and use `--expert-profile` to regenerate an expert hotlist against our own corpus
for the MLX runner. Details in §9.

Detail, evidence and the ranked transferable-ideas list follow.

---

## 1. SSD streaming architecture

### 1.1 What is (and is not) in the files named in the brief

`ds4_ssd.c` / `ds4_ssd.h` are **not** the streaming engine. They are 211 lines of budget
arithmetic and argument parsing:

- `ds4_parse_streaming_cache_experts_arg` (`ds4_ssd.c:46`) — accepts either `N` (exact
  dynamic expert slot count) or `NGB` (a routed-memory byte budget).
- `ds4_ssd_auto_cache_plan` (`ds4_ssd.c:108`) — takes the backend's recommended working
  set, multiplies by a percentage, subtracts non-routed weights, and converts the
  remainder to an expert count.
- `ds4_ssd_auto_cache_percent` (`ds4_ssd.c:80`) — **80%** by default, overridable via
  `DS4_SSD_AUTO_CACHE_PCT` in the range 50..95. The comment at `ds4_ssd.c:95-104` is
  worth reading: it records that the expert cache is the single biggest decode lever but
  that pushing the split higher trips the OOM killer on a unified-memory machine.
- `ds4_ssd_memory_lock_acquire` (`ds4_ssd.c:137`) — a diagnostic that mmaps + touches +
  `mlock`s N GiB in 256 MiB chunks to simulate a smaller machine (`--simulate-used-memory`).

`ds4_layer_pack.c` / `.h` are likewise **not** SSD-related. They implement a
monotonic-contiguous *multi-GPU* placement packer (`ds4_compute_layer_placement`,
`ds4_layer_pack.c:14`): walk entries in forward order (entry 0 = embedding, 1..n_layers =
transformer layers, n_layers+1 = output head), assign each to the first device with room,
and once anything spills to CPU everything after it does too. Relevant to the two-machine
split, not to disk streaming.

The actual streaming machinery lives in `ds4.c` (~lines 17,500–18,900 and 34,400–35,100)
and `ds4_metal.m` (~lines 11,900–14,200).

### 1.2 Two coexisting weight-access modes

ds4 deliberately runs **two different I/O paths against the same GGUF**, and picks per
phase.

**(a) mmap + zero-copy Metal views — the prefill path.**
The model is opened once with `MAP_SHARED` on the Metal path and `MAP_PRIVATE` on the CPU
path (`ds4.c:2476-2477`); the comment at `ds4.c:2463-2475` explains the split is defensive
against a Darwin VM panic in map-count accounting when the CPU backend streams a very large
GGUF through a shared mapping. Metal then wraps *slices of that file mapping* as buffers
with `newBufferWithBytesNoCopy` (`ds4_metal.m:1881`, `ds4_metal.m:11507`), installed via
`ds4_gpu_set_model_map_spans` (`ds4_metal.m`, entry point used from
`metal_graph_install_model_spans`, `ds4.c:17520-17537`). Per layer, prefill calls
`metal_graph_stream_map_layer` (`ds4.c` — builds spans for layer `il` and installs them).

This is the deepest trick in the engine: on unified memory, GPU reads of weights become
**page faults on a file-backed mapping**, with no explicit copy and no staging buffer. GGUF
tensor data is aligned to `general.alignment`, default 32 bytes (`ds4.c:2366`,
`ds4.c:2429`), and the no-copy buffer creation page-aligns the base itself
(`ds4_metal.m:11507` uses `base + page_offset`).

**(b) Explicit `pread` into slab-allocated Metal buffers — the expert-cache path.**
`ds4_gpu_stream_expert_pread_into` (`ds4_metal.m`) is a plain retry loop around `pread(2)`
into the CPU-visible pointer of a shared `MTLBuffer`. Notably:

- **Buffered I/O.** No `O_DIRECT`, no `F_NOCACHE` anywhere in the tree. The macOS unified
  buffer cache is therefore an implicit **second-level expert cache** sitting under ds4's
  own. This is a deliberate-looking choice and it matters: on a 128 GB machine with a
  59 GB explicit cache, the remaining free RAM still absorbs expert re-reads.
- **No alignment requirement.** Offsets are raw GGUF tensor offsets; no sector alignment,
  no read-modify-write.

### 1.3 Prefetch and hinting

Three independent mechanisms, all opt-in-able / disable-able by environment variable:

| Mechanism | Implementation | Enable |
| --- | --- | --- |
| macOS `F_RDADVISE` readahead | `metal_graph_stream_readahead_range_impl`, `ds4.c:17588-17620` (chunked to `INT_MAX` per `struct radvisory`) | `DS4_METAL_ENABLE_STREAMING_READAHEAD` |
| `posix_madvise(POSIX_MADV_WILLNEED)` | `metal_graph_stream_madvise_willneed_range_impl`, `ds4.c:17626-17674` (page-aligns down, rounds length up, clamps to file size) | `DS4_METAL_ENABLE_STREAMING_MADVISE_WILLNEED` |
| Threaded explicit page-in / `pread` | `metal_graph_stream_pagein_job`, `ds4.c:17720+`, dispatched at `ds4.c:18820` | on by default for streaming prefill |

Eviction has a matching hint: `ds4_gpu_stream_expert_evict_dontneed_range`
(`ds4_metal.m:12780`) issues `MADV_DONTNEED` over the evicted slot so the page cache does
not retain a copy that is no longer wanted. The accounting counter is
`g_stream_expert_cache_evict_advise_bytes` (`ds4_metal.m:391`).

### 1.4 The persistent pread worker pool

`ds4_metal.m:12004-12200` implements a **persistent pthread pool, up to 18 threads**
(`g_stream_expert_pread_pool_threads[18]`, `ds4_metal.m:12008`; clamped at
`ds4_metal.m:12080`), coordinated with a generation counter + two condvars, with workers
pulling from a shared task index (`ds4_metal.m:12045-12060`). Disable with
`DS4_METAL_STREAMING_EXPERT_PREAD_POOL=0` (`ds4_metal.m:12018`).

The design intent is explicit in the structure: threads are created **once** and reused per
dispatch (`ds4_gpu_stream_expert_pread_pool_dispatch`, `ds4_metal.m:12180`), because the
per-layer prefill loop dispatches a fresh task set 43 times per chunk and pthread creation
at that rate would dominate. This is the single most directly portable idea in the file —
see §8.

### 1.5 The expert cache: structure and eviction

**Structure** (`ds4_metal.m:640-700`): a *dense, directly-indexed* two-dimensional table

```c
static ds4_gpu_stream_expert_cache_entry
    g_stream_expert_cache[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER]
                         [DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
```

O(1) lookup by `(layer, expert)`, no hashing, no chaining. Each entry carries
`gate/up/down` buffer handles + inner offsets, `last_used`, `use_count`, `inflight_seq`,
`slab_slot`, `valid`, `slab_backed` (`ds4_metal.m:640-652`).

**Backing store**: a single-size-class **slab allocator**. `g_stream_expert_cache_slabs[]`
with per-slab start-slot / slot-count / slots-used arrays and an explicit free-slot stack
(`ds4_metal.m:684-691`), all slots exactly `g_stream_expert_cache_slab_slot_bytes` wide
(`ds4_metal.m:699`). Slots can be `mlock`ed individually
(`g_stream_expert_cache_slab_slot_locked`, `ds4_metal.m:691`) with a budget cap and a
"relief" fallback when locking fails.

The single size class has a real consequence, documented at `ds4.c:4630-4636`: on a
**mixed-precision GGUF** (e.g. Q4_K experts on a few layers among IQ2 layers) only layers
whose per-expert byte size matches the dominant slab class can be served from the cache at
all; the others fall back to mapped-model views. The slab class is chosen as the *most
common* per-expert size, not the first (`ds4_streaming_routed_expert_bytes`, `ds4.c` —
"Choosing that first layer as the slab class makes every ordinary layer bypass the cache").

**Eviction** (`ds4_metal.m:14005-14070`) is **LFU-with-decay, LRU tiebreak, with a
protected set**:

```c
const uint32_t hotness = g_stream_expert_cache_route_hotness[layer][expert];
if (hotness < lowest_hotness ||
    (hotness == lowest_hotness && e->last_used < oldest)) { ... victim = ... }
```

- Candidates must be *reusable* (matching byte sizes), not *in-flight*
  (`ds4_gpu_stream_expert_cache_entry_inflight`), and not *protected* — the experts
  selected by the current routing step are exempt
  (`ds4_gpu_stream_expert_cache_entry_protected`, `ds4_metal.m:14031`).
- Hotness is incremented on every routing selection
  (`ds4_gpu_stream_expert_cache_note_route_hotness`, `ds4_metal.m:12896`) and **halved
  periodically** (`hotness >>= 1`, `ds4_metal.m:12867`) on a token-count trigger
  (`ds4_gpu_stream_expert_cache_maybe_decay_route_hotness`, `ds4_metal.m:12880`). This is
  a classic aging counter — it prevents early-corpus experts from pinning forever.
- The victim scan is a **full linear sweep of the layer×expert table** every time a slot is
  needed, instrumented as `reuse_scan_calls` / `reuse_scan_entries_avg`
  (`ds4_metal.m:14050`). For Flash that is 43 × 256 = 11,008 entries per eviction. This is
  the one clearly suboptimal piece of the design.

### 1.6 Hot-expert preload and where the hotlist comes from

**The hotlist is measured routing telemetry, baked into the binary as a header.**

- Collection: `--expert-profile FILE` (`ds4_help.c:193`, `ds4_help.c:276`) activates
  `ds4_expert_profile` (`ds4.c:1232-1259`), which accumulates a per-`(layer, expert)`
  hit histogram *and* a weight histogram, plus adjacency statistics (`adjacent_pairs`,
  `adjacent_overlap_sum`, `adjacent_jaccard_sum`) and a multi-capacity cache-hit
  simulation (`cache_hits[cap][layer]`).
- Emission: `ds4_expert_profile_write_hotlist_file` (`ds4.c:1540`) sorts by
  `(count, weight)` descending (`ds4_expert_hotlist_sort_cmp`, `ds4.c:1281`) and writes a
  text file with header `# ds4 expert hotlist v1` and columns `layer expert hits weight`
  (`ds4.c:1567-1577`).
- Baked defaults: `ds4_streaming_hotlist.inc` (13,334 lines) holds
  `ds4_default_streaming_hotlist_pro[][2]` (`:3`) and
  `ds4_default_streaming_hotlist_flash[][2]` (`:6894`) — **6,884 entries for Pro and 6,436
  for Flash**, each a bare `{layer, expert}` pair in priority order. Flash has
  43 × 256 = 11,008 routed experts, so the baked hotlist covers **58%** of them.
  `ds4_streaming_hotlist_glm52.inc:4` holds ~6,500 for GLM-5.2. Header comment:
  *"Generated from ds4 expert hotlist profiles; sorted by hits/weight."*
- Loading: `metal_graph_streaming_expert_hotlist_load_default` (`ds4.c:20701`) selects the
  table by model variant; `metal_graph_streaming_expert_hotlist_load_file` (`ds4.c:20614`)
  parses a user-supplied file pointed at by `DS4_METAL_STREAMING_EXPERT_HOTLIST`. Entries
  with `hits == 0` are skipped (`ds4.c:20659`).

> **Provenance gap.** *Which corpus* produced the baked hotlists is documented nowhere.
> `grep -rn hotlist README.md QA_BEFORE_RELEASES.md AGENT.md gguf-tools/README.md` returns
> nothing, the clone has a single squashed commit so git history is no help, and there is
> no generation script in `Makefile`, `misc/`, `tests/`, or `speed-bench/`. Treat the
> Flash hotlist as *"antirez's personal usage distribution"* of unknown composition. For a
> teacher run over our own corpus this is a reason to **regenerate rather than reuse** —
> and ds4 gives us the tool to do so (`--expert-profile`).

**Preload sizing** (`metal_graph_streaming_expert_preload_count`, `ds4.c:20746`): an
explicit `--ssd-streaming-preload-experts N` wins; otherwise auto mode preloads the whole
cache budget but **capped at 4,096 experts** (`ds4.c:20770`), overridable with
`DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP`. The comment is instructive: auto mode is
"a hot seed, not a request to synchronously fill the whole cache. Large Flash caches can
otherwise spend startup doing thousands of preads into shared Metal buffers and trip the
system watchdog before decode begins."

There is also a clever **free re-seed**: after each prefill layer, the hotlist experts for
that layer are blitted model→cache *from the already-mapped views* rather than re-read
(`metal_graph_seed_streaming_expert_cache_layer_from_mapped_hotlist`, `ds4.c:20790+`), and
after the whole prefill both
`metal_graph_seed_streaming_expert_cache_from_hotlist` and
`..._from_prefill` run (`ds4.c:35041-35045`) so decode starts warm on what prefill actually
routed.

`--ssd-streaming-cold` (`ds4_help.c:171`) disables all of this; it exists for measurement,
and the README says so (`README.md:326`).

### 1.7 The two-layer reservation and overlapped prefill

```c
enum { DS4_STREAMING_PREFILL_HEADROOM_LAYERS = 2 };   // ds4.c:4567
```

`ds4_streaming_prefill_headroom_bytes` (`ds4.c:4600`) computes
`per_expert_bytes × DS4_N_EXPERT × min(cacheable_layers, 2)` and this is subtracted from
the byte budget **before** the dynamic cache is sized (`ds4.c:55488-55505`); if the budget
is smaller than the headroom, startup fails with an explicit error. The startup log prints
the decomposition (`ds4.c:55639-55647`):
`total = prefill headroom + full layers + dynamic cache`.

Why two: layer-major prefill needs layer `il` resident while it prefetches layer `il+1`.
The prefetch is genuinely overlapped and the lookahead is **deeper than two**:

```c
enum { DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD = 4 };    // ds4.c:18662
```

with the runtime depth from `DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_AHEAD`
(`ds4.c:18664-18675`), defaulting to 1 when layer-prepare is off. The loop at
`ds4.c:34751-34776` starts `il+1 .. il+ahead` prepares before encoding layer `il`, and
falls back to a plain readahead of `il+1` when the prepare path is disabled
(`ds4.c:34779-34787`). When there is no next layer it prefetches the output head instead
(`metal_graph_stream_readahead_output`, `ds4.c:34781`).

### 1.8 The selected-expert page-in optimisation

The most sophisticated piece, at `ds4.c:18680-18830`. Before a layer's FFN runs, ds4 reads
back the **router's already-computed selected-expert ids** for the whole chunk
(`ds4_gpu_tensor_read(metal_graph_batch_router_selected(g), ...)`, `ds4.c:18713`), builds a
`seen[]` bitmap over the 256 experts, then **coalesces runs of selected experts into
contiguous byte ranges** — including a tunable *gap* parameter that bridges short unselected
stretches to trade a little wasted I/O for far fewer, larger reads
(`ds4.c:18744-18762`). Three ranges are emitted per run (gate, up, down;
`ds4.c:18789-18800`), then dispatched across the thread pool.

This is exactly the right shape: *the routing decision is known before the weights are
needed*, so the I/O for a layer can be issued as a small number of large sequential reads
of precisely the bytes that will be used.

### 1.9 What actually gets read, quantitatively

For Flash (`DS4_SHAPE_FLASH`, `ds4.c:540-575`): 43 layers, `n_embd` 4096, `n_ff_exp` 2048,
`n_expert` 256, `n_expert_used` 6.

Per routed expert at MXFP4 (17 bytes / 32 values): gate + up + down =
3 × (4096 × 2048 / 32 × 17) = **12.75 MiB**. Per layer: 256 × 12.75 MiB = **3.19 GiB**.
Whole routed set: **≈137 GiB**. The two-layer prefill reservation is therefore
**≈6.4 GiB** — a meaningful bite out of a 128 GB machine before the dynamic cache gets
anything.

Now the part that governs a long-context teacher run. The default prefill chunk for Flash
is **4,096 tokens** (`ds4_prefill_cap_for_prompt`, `ds4.c`; `ds4_help.c:176`). At 4,096
tokens × 6 selections = 24,576 draws over 256 experts per layer, **essentially every
expert in every layer is touched in every chunk**. So an 81,920-token prompt is
81,920 / 4,096 = **20 full sweeps of the 137 GiB routed set ≈ 2.7 TiB of reads per
session**, less whatever the expert cache and the page cache absorb.

The counter-pressure is the indexer staging buffer, which is the only term that grows with
chunk size (`metal_graph_context_bytes_for_kv_policy`, `ds4.c:16143-16147`):

```c
bytes = kv_cache_bytes + 2ull * comp_cap * prefill_cap * sizeof(float);
```

with `comp_cap = ctx / 4 + 2 = 20,482` at our 81,920 window. That gives:

| `--prefill-chunk` | sweeps per 81,920-token session | routed bytes read | indexer staging |
| ---: | ---: | ---: | ---: |
| 4,096 (default) | 20 | ~2.7 TiB | 0.63 GiB |
| 8,192 | 10 | ~1.34 TiB | 1.25 GiB |
| 16,384 | 5 | ~0.67 TiB | 2.5 GiB |
| 32,768 | 3 | ~0.40 TiB | 5.0 GiB |

**Raising `--prefill-chunk` is the single highest-leverage streaming knob for a
long-context teacher pass**, and it is nearly free: streaming cost falls linearly with the
number of sweeps while the staging cost rises linearly from a small base. This is not
stated anywhere in the ds4 docs and did not show up in any committed benchmark.

---

## 2. Prefill throughput evidence

Everything below is **resident** (model fully in unified memory) unless flagged. The
committed sweeps carry no streaming flag, and `ds4-bench` requires an explicit
`--ssd-streaming` opt-in (`ds4_bench.c:286`), so no committed CSV is a streaming run.

### 2.1 Methodology (so the numbers can be read correctly)

`ds4-bench` (`ds4_bench.c:683-698`) times `ds4_session_sync()` over the **incremental
prefix only** — `prefill_tokens = frontier - previous`, 2,048 by default. Each CSV row is
therefore the *instantaneous* prefill rate for that 2,048-token interval **at that context
depth**, not a cumulative average. It snapshots and restores KV around the generation probe
so decode cannot contaminate the next interval (`ds4_bench.c:705-795`). Corpus is
`speed-bench/promessi_sposi.txt` (1.3 MB). Invocation is documented at
`speed-bench/README.md:1-28`.

Note also `README.md:792-793`: `--power N` throttles by sleeping *between layers during
prefill*, so any prefill number is only valid at `--power 100`.

### 2.2 M5 Max 128 GB, Metal, Flash q2, resident — the machine we actually have

From `speed-bench/m5_max.csv`:

| ctx | prefill tok/s | decode tok/s | line |
| ---: | ---: | ---: | --- |
| 2,048 | **790.18** | 39.35 | `:2` |
| 8,192 | 683.85 | 37.03 | `:5` |
| 16,384 | 572.53 | 36.14 | `:9` |
| 32,768 | 557.04 | 34.36 | `:17` |
| 49,152 | 448.53 | 29.07 | `:25` |
| 65,536 | **398.50** | 27.64 | `:33` |

Summarised in `README.md:261-264`. There is a curious plateau/rebound between 18k and 30k
(565–570 tok/s) before the decline resumes.

**No Mac CSV in the repo goes past 65,536 tokens.** Our target window is 81,920. The only
evidence beyond 64k is `speed-bench/m3_max_ts.svg`, a plot with no accompanying CSV, no
README reference and no recorded model/quant/residency — an M3 Max sweep to ~133k that
degrades to roughly half its 2k rate by 128k. Treat as indicative only.

### 2.3 Other hardware, for calibration

| Machine | Model | 2,048 | 16,384 | 32,768 | 65,536 | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| M5 Max 128 GB | Flash q2 | 790.18 | 572.53 | 557.04 | 398.50 | `m5_max.csv` |
| M4 Max 128 GB | Flash q2 | 343.76 | 277.04 | 247.91 | 204.96 | `m4_max.csv` |
| M2 Ultra | Flash q2 | 410.62 | 362.43 | 325.77 | 269.75 | `m2_ultra.csv` |
| M3 Ultra 512 GB | **Pro** q2 | 183.06 | 146.74 | 138.82 | — | `pro_model_m3_ultra.csv` |
| DGX Spark GB10 | Flash q2 | 825.76 | 872.44 | 855.94 | 822.98 | `gb10.csv` |

The **M4 Max is ~2.3x slower than the M5 Max at prefill** and holds that ratio across the
sweep — directly relevant to sizing a two-machine split (§5). `m4_max.csv` and
`m2_ultra.csv` are tracked but referenced nowhere in the docs.

Note also the shape contrast: every Mac degrades with context (M5 Max is the worst, at
**50% of its 2k rate by 64k**), while the CUDA GB10 is essentially flat. On Apple silicon,
long-context prefill throughput is not a constant.

### 2.4 Streaming prefill — the one number that exists

`README.md:695`, two M5 Max 128 GB, GLM-5.2 IQ2_XXS (188 GiB):

> prefill (4,096 tokens): **~94 tok/s** across two Macs, fully memory-resident, vs
> **~3–5 tok/s** on one Mac with SSD streaming.

That is a **20–30x prefill penalty**, and against the same machine's *resident Flash*
numbers it is 100–200x. Sanity-checking it against §1.9: if ~88 GiB had to come off disk
per 4,096-token chunk and the chunk took ~1,000 s, effective read bandwidth was
**~86 MB/s** — roughly 1/60th of sequential NVMe. Streamed prefill is not achieving
sequential bandwidth; it is dominated by random access and cache thrash.

This directly contradicts the README's own qualitative claim at `README.md:296-301`
("Long prefills can still be fast"). Where a measured number and a prose claim disagree,
we should trust the number.

QA (`QA_BEFORE_RELEASES.md:181-182`, `:341`, `:371-387`) requires resident-vs-streaming
runs only for **correctness**, never for speed — no streaming prefill throughput is
recorded anywhere in the release process.

### 2.5 The number that matters for Wave 3

11.7 M supervised tokens, at the rates above:

| Regime | Rate | Wall clock for 11.7 M tokens | Basis |
| --- | ---: | ---: | --- |
| Two M5 Max resident, pipelined | 654.79 tok/s | **~5.0 h** | measured, `README.md:438` @63,819 |
| M5 Max + M4 Max resident, pipelined | ~550 tok/s est. | **~5.9 h** | §4.8, M4 is ~2.3x slower |
| Single M5 Max resident, ~64k ctx | 398.50 tok/s | **~8.1 h** | measured, `m5_max.csv:33` |
| Single M5 Max SSD-streamed | 3–50 tok/s | **2.7 – 45 days** | extrapolated from `README.md:695` |

The last row is the whole argument. Note the resident rows are *hypothetical for a faithful
artifact* on a single machine — 156 GB does not fit in 128 GB, which is precisely why the
single-machine row cannot actually be realised (§3.6, §4.8).

---

## 3. Can ds4 produce the teacher outputs?

### 3.1 The four deliverables against the code

| Deliverable | Status | Evidence |
| --- | --- | --- |
| **(a)** top-2048 logits per supervised position | **Missing** — one position per chunk | `ds4.c:35047`, `:35257`, `:35096` |
| **(b)** final hidden states per position | **Half present** — batched HC state already read back; post-norm collapse is last-row-only | `ds4.c:59698`, `:58950` |
| **(c)** MTP-head logits | **Computed but unreachable** — buffer exists, no accessor | `ds4.c:26071`, `:26102`, `:60904` |
| **(d)** activation statistics | **Present and usable today** (MoE second moments only) | `ds4.c:33869-33884`, `:34878` |

### 3.2 (a) Per-position logits — the blocker, and the way through

The output head runs on one row:

```c
/* ds4.c:35047 — metal_graph_prefill_layer_major */
uint32_t output_row = (uint32_t)n_tokens - 1u;
```

`DS4_METAL_GRAPH_OUTPUT_ROW` (`ds4.c:35049-35056`) can *move* that row but not multiply it.
Non-final chunks are given `NULL` logits entirely (`ds4.c:35257`), the readback is one
`DS4_N_VOCAB` row wide (`ds4.c:35096`), and the session's logits buffer is a single
`DS4_N_VOCAB` float array (`ds4.c:49294`, allocated `ds4.c:58513`).

The existing per-position path is decode-serial. `--perplexity-file` (`ds4_help.c:277`,
impl `ds4_cli.c:1086-1167`) seeds a 32-token prefix then loops
`ds4_session_token_logprob` → `ds4_session_eval` **one token at a time**
(`ds4_cli.c:1138`, `:1151`). Correct, and completely unusable at 11.7 M tokens.

**The way through** is `metal_graph_encode_output_head_batch` (`ds4.c:25653`), which
already does HC collapse + output norm + vocab projection over *all* rows of a batch into
`g->spec_logits`, and whose caller already reads back the full
`n_tokens × DS4_N_VOCAB` block:

```c
/* ds4.c:35561-35565 */
if (ok && row_logits) {
    ok = ds4_gpu_tensor_read(g->spec_logits, 0, row_logits,
                             (uint64_t)n_tokens * DS4_N_VOCAB * sizeof(row_logits[0])) != 0;
}
```

It is capped at 16 rows only because `g->spec_logits` is allocated as
`16 * DS4_N_VOCAB * sizeof(float)` (`ds4.c:17315-17316`, bounds-checked at
`ds4.c:25686-25689`), and it is `static` with no public entry point — it exists to serve
the DSpark/MTP verifier.

Two further things must be fixed to make it usable at our scale:

- **Top-K selection is the wrong algorithm.** `ds4_session_top_logprobs` (`ds4.c:60750`) is
  an insertion sort into a k-array *inside* the vocab scan (`ds4.c:60759-60771`) —
  O(k · vocab) = 2048 × 129,280 ≈ **265 M comparisons per position** on the CPU. At
  11.7 M positions that alone is the dominant cost. It needs a partial selection
  (threshold pass + heap), and ideally on-GPU: `ds4_gpu_indexer_topk_tensor`
  (`ds4.c:22997` et al.) already performs top-K over rows and is the natural kernel to
  reuse.
- **Readback bandwidth.** 129,280 × 4 B = **517 KB of raw logits per position**. Over
  11.7 M positions that is 6 TB moved if the reduction happens on the host. The top-K must
  happen before the readback.

Vocab is **129,280** for both Flash and Pro (`ds4.c:547`, `:585`); GLM-5.2 is 154,880
(`ds4.c:623`).

### 3.3 (b) Hidden states — mostly already there

`ds4_session_eval_layer_slice` (`ds4.h:429-440`), the distributed-inference entry point,
reads back per-position hidden states for a whole batch:

```c
/* ds4.c:59698 */
if (ok && output_hc) {
    ok = ds4_gpu_tensor_read(metal_graph_batch_cur_hc(g), 0, output_hc, hc_bytes) != 0;
}
/* hc_bytes = n_tokens * hc_dim * sizeof(float), ds4.c:59499 */
```

`hc_dim = DS4_N_HC × DS4_N_EMBD` = 4 × 4096 = **16,384 floats per position**. Setting
`layer_end = n_layer - 1` with `output_logits = false` yields the final-layer residual for
every position — already host-side, already wire-serialisable
(`ds4_distributed.c:2506`, `:2567`).

Caveat: that is the **pre-collapse manifold-hyper-connection state**, not the
post-`output_norm` vector. The collapse+norm exists as
`ds4_session_eval_output_head_from_hc` (`ds4.h:441`) but hardcodes the last row:

```c
/* ds4.c:58950 */
const float *last_hc = hidden_hc + (uint64_t)(n_tokens - 1u) * hidden_dim;
```

Lifting that to a loop (or reusing the batched norm at `ds4.c:25683-25714`) is small.

Note the size: 16,384 floats/position at fp32 is **64 KB/position**. For 120 sessions ×
81,920 tokens that is ~630 GB at fp32, ~315 GB at fp16. Storage planning matters more than
the extraction patch.

### 3.4 (c) MTP logits — computed, gated, unexported

`metal_graph_encode_output_head_mtp` (`ds4.c:26071`) runs the MTP HC head and projects
through the **base model's** output matrix to full vocab (`ds4.c:26102`). The result lands
in `s->mtp_logits`, a `DS4_N_VOCAB` buffer (`ds4.c:58756`) — but **only when
`DS4_MTP_FULL_LOGITS` is set** (`ds4.c:60904`, `ds4.c:66100-66101`); otherwise the GPU
returns only the argmax. There is no accessor: `ds4_session_top_logprobs` and
`ds4_session_copy_logits` read `s->logits`, never `s->mtp_logits`. The public MTP surface
is just `ds4_engine_has_mtp` (`ds4.h:422`) and `ds4_engine_mtp_draft_tokens` (`:423`).

Also relevant to the campaign's Wave 3 step 3: DSpark captures **mean-over-HC-rows from
selected target layers** and deliberately keeps them on-GPU to avoid readbacks
(`ds4.c:15093-15095`, buffers at `ds4.c:16014-16019`). The DSpark support GGUF is a
separate ~6 GB file (`download_model.sh:14`, `:71-72`) built by
`deepseek4-quantize --dspark-support` from the `mtp.*` tensors, with target layers
defaulting to `40,41,42` (`gguf-tools/deepseek4-quantize.c:2647`) — consistent with the
campaign's Wave 1 finding.

### 3.5 (d) Activation statistics — works today, under streaming

This one is a genuine win. `--imatrix-dataset` / `--imatrix-out` / `--imatrix-max-prompts`
/ `--imatrix-max-tokens` (`ds4_help.c:278-281`) drive
`ds4_engine_collect_imatrix` (`ds4.c:52280`), which:

- hooks straight into the **batched prefill loop** (`ds4.c:34878`, `:34937`) — no decode
  serialisation, full prefill speed;
- observes the exact release graph without changing inference math (comment,
  `ds4.c:33858-33864`);
- accumulates per-`(layer, expert)` **sum of squares** of both MoE matmul inputs —
  `gate_up_sum2[layer][expert][4096]` from `batch_ffn_norm` and
  `down_sum2[layer][expert][2048]` from `batch_routed_mid` (`ds4.c:33869-33884`,
  accumulation at `ds4.c:33947-33965`);
- writes llama.cpp legacy imatrix `.dat` (`imatrix_collector_save`, `ds4.c:34012`);
- **propagates the streaming flag**: `g.ssd_streaming = e->ssd_streaming`
  (`ds4.c:52322-52324`) — so it runs on the SSD-streaming path.

Constraints: Metal-only (`ds4.c:52297-52300`), and it is *corpus-aggregate* — no
per-position, no per-session, no mean/variance/max, nothing for attention or the residual
stream. If Wave 3's "calibration statistics / activation ranges" means an imatrix for
quantization calibration, **ds4 delivers it locally with no code change**. If it means
per-layer activation ranges for outlier analysis, it does not.

`--expert-profile` (§1.6) additionally gives per-layer routing histograms, adjacent-token
Jaccard overlap and a cache-hit simulation — telemetry, not distillation signal, but
directly useful for VQ work.

### 3.6 Which GGUF is quality-equivalent to the source checkpoint

This matters: a teacher pass against a re-quantized artifact teaches the student the
quantizer's errors.

**The answer is `ds4f-mxfp4`, and the fidelity claim is verified in code, not just
asserted.** `repack_fp4_weight_mxfp4` (`gguf-tools/deepseek4-quantize.c:824`) requires
packed `I8` weights with `F8_E8M0` scales (`:825-826`), repacks them into GGUF's 17-byte
MXFP4 block (`:845-846`), and then **verifies every byte and dies on mismatch**:

```c
if (dst[0] != scale->data[block_index]) die("MXFP4 scale repack mismatch");   /* :870 */
if (src_code != dst_code)               die("MXFP4 code repack mismatch");    /* :876 */
```

The usage text is explicit: *"MXFP4 is supported only for lossless repacking of packed
routed experts"* (`:2664`). `README.md:120-123` says the same in prose: it "preserves
DeepSeek's released MXFP4 routed-expert weights rather than requantizing them."

| Variant | `download_model.sh` | Size | Routed experts | Verdict |
| --- | --- | ---: | --- | --- |
| `ds4f-mxfp4` | `:9`, `:65-66` | **~156 GB** | **bit-exact native FP4** | **Only faithful option** |
| `ds4f-q4` | `:8`, `:62-63` | ~153 GB | Q4_K re-quantized | Re-quantized — not a teacher |
| `ds4f-q2-q4` | `:10`, `:56-58` | ~98 GB | mixed IQ2_XXS/Q4_K | Re-quantized |
| `ds4f-q2` | `:7`, `:52-53` | ~81 GB | IQ2_XXS + Q2_K, imatrix | Re-quantized |

The residents are the remaining wrinkle. The mxfp4 filename
(`DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-…`,
`download_model.sh:9`) decomposes as:

- **HC / compressor / indexer at F16** — the source is `F8_E4M3`, and e4m3 (4-bit exponent,
  3-bit mantissa) is a strict subset of fp16, so this upcast is **exactly lossless**.
- **Attention projections, shared expert, output head at Q8_0** — an 8-bit *integer*
  block quantization of 8-bit *float* source. Tight, but **not** bit-exact.

Since `deepseek4-quantize` exposes `--attention-proj`, `--shared`, `--output`,
`--embedding` and `--dense` as independent type flags (`:2653-2659`), a **fully faithful
artifact is buildable with the shipped tool**: MXFP4 experts + F16 for every resident.
Cost is roughly **+8 GB over the 156 GB release, so ~164 GB**.

Either way: **156–164 GB against a 128 GB machine.** The faithful artifact does not fit.
That is the structural fact behind the whole verdict.

### 3.7 Verdict

**Infeasible as shipped. Feasible after a bounded patch, but only in a resident (two-Mac)
configuration.**

What is missing, in dependency order:

1. **Batched output head over all rows.** Widen `spec_logits` past 16 (`ds4.c:17315`),
   hoist the single-row head at `ds4.c:35047` / `:35257` into an all-rows mode, expose an
   entry point in `ds4.h`. *Medium C work in a 2.9 MB file with no test harness for this
   path.*
2. **On-GPU top-2048 + binary writer.** Replace `ds4.c:60750`'s O(k·vocab) scan; reuse
   `ds4_gpu_indexer_topk_tensor`. JSON is not viable at 2048 × 11.7 M.
3. **Per-position post-norm hidden states.** Loop `ds4.c:58950`.
4. **MTP logits accessor.** `ds4_session_copy_mtp_logits()` + CLI flag; buffer and env gate
   already exist.
5. **Activation statistics.** Already works (§3.5).

None of this is unreasonable. But it is a real engineering project inside an unfamiliar
codebase (`ds4.c` alone is 66,714 lines / 2.9 MB; `ds4_metal.m` another 43,259), and it
buys us a *runtime* we still have to feed a resident
model to. Weigh it against patching our own MLX path, where we control the code.

---

## 4. Distributed pipeline parallelism — the two-Mac option

### 4.1 There are two different "two MacBooks" features, and the README conflates them

This is the biggest readability trap in the repo. `README.md:400-449` describes
**pipeline parallelism** across "two 128 GB MacBooks"; `README.md:617-706` describes
**tensor parallelism** across "two 128 GB MacBooks". They are different subsystems,
different files, different transports, and mutually exclusive at the CLI
(`ds4_tp.c:503-506` rejects `--role` + TP; PP requires `--layers`, TP forbids it at
`ds4_tp.c:434-437`).

| | Pipeline parallelism | Tensor parallelism |
| --- | --- | --- |
| File | `ds4_distributed.c` (324 KB) | `ds4_tp.c` (86 KB) |
| Nodes | N, contiguous layer slices via `--layers A:B` | exactly 2, always 50/50 |
| Transport | **plain TCP only** | **RDMA over Thunderbolt**, TCP fallback |
| What is split | layers (weights + KV both sharded) | routed experts only; dense/attn/embed/head **replicated**, KV **replicated** |
| Helps prefill | **yes, ~1.85x measured** | yes (~94 vs 3–5 tok/s vs streaming) |
| Helps decode | **no — 19.4% worse** (`README.md:444-449`) | yes (16.8 vs 4.8 tok/s) |
| Works with `--ssd-streaming` | yes | **no** — rejected at `ds4_tp.c:496-499` |
| Exposed by | `ds4`, `ds4-server`, `ds4-bench` | **`ds4` CLI only** |
| Fault tolerance | real: route rebuild + transcript replay | **none** — worker `break`s and exits |

**For a teacher prefill run, pipeline parallelism is the right mode**, and RDMA is
irrelevant: PP has no RDMA path at all.

### 4.2 PP transport and framing

Plain blocking TCP. `dist_set_socket_low_latency` (`ds4_distributed.c:1027`) sets
`TCP_NODELAY`, `SO_KEEPALIVE`, a 60 s send timeout (`:1030`) and 128 MB socket buffers
(`:710-722`). **No receive timeout by default** (`:1043-1060`) — a hung-but-connected peer
blocks indefinitely unless `DS4_DIST_SOCKET_RECV_TIMEOUT_SEC` is set. Listener at
`:1193-1221` (`AF_UNSPEC`, `listen(fd, 64)`).

Framing is hand-rolled fixed binary: 12-byte header `{magic, type, bytes}`
(`ds4_distributed.c:71-75`), magic `"DS4D"` (`:44`), message types
`HELLO / ERROR / WORK / RESULT / SNAPSHOT_*` (`:45-53`). No discovery, no registry —
workers dial a hardcoded `--coordinator HOST PORT` in a 1 s retry loop (`:7986-7992`).

`README.md:612-615` is blunt about the security posture: no encryption, no
authentication, not release-stable, same-commit builds required, trusted networks only.
For a lab run over a direct cable this is acceptable; note it and do not expose the port.

### 4.3 What crosses the wire

**Forward (activations), per token per boundary:** `n_hc × n_embd` floats. For Flash that
is 4 × 4096 = **16,384 floats = 64 KiB/token at fp32**
(`ds4_engine_hidden_f32_values`, `ds4.c:58122`; payload sizing at
`ds4_distributed.c:2685-2686`). `--dist-activation-bits 16|8` halves/quarters it
(`ds4_distributed.c:818`, `:878`; default 32 at `:63`), though `README.md:566-573` says
the reduction "didn't provide a significant improvement".

**Return (logits), per generated token: fp32, hard-pinned.**

```c
/* ds4_distributed.c:5862-5864 */
} else if (status == 0 && result_kind == DS4_DIST_RESULT_LOGITS) {
    payload_bits = 32u;
```

129,280 × 4 = **505 KiB per token**, ~8x the forward payload. This is what makes
high-latency links collapse (`README.md:527-531`: 582.99 → 114.88 tok/s prefill and
25.09 → 3.63 tok/s decode from Thunderbolt 5 to VPN).

**Prefill amortizes well**: non-final chunks return only an ACK
(`DS4_DIST_RESULT_ACK`, `ds4_distributed.c:7415-7417`), so a 4,096-token chunk moves
256 MiB forward and a few bytes back. Over Thunderbolt 5 that is sub-second. **For a
prefill-only teacher pass the network is not the bottleneck.**

### 4.4 Where the output head runs — and why that is good news

Default topology (`--layers 20:output`): the **last node** owns the head, computes logits
locally, and ships them back:

```c
/* ds4_distributed.c:7409-7421 */
const bool local_output_logits = output_logits && !has_next && !final_ack_only;
const uint32_t result_kind = final_ack_only ? DS4_DIST_RESULT_ACK
    : (local_output_logits ? DS4_DIST_RESULT_LOGITS : DS4_DIST_RESULT_HIDDEN_STATE);
```

The coordinator `memcpy`s them into its sampling buffer (`ds4_distributed.c:2625-2628`).
The alternative topology (`--layers 20:42`, head on the coordinator) makes the last worker
return the **hidden state** and the coordinator run
`ds4_session_eval_output_head_from_hc` (`ds4_distributed.c:2640-2646`) —
`README.md:511-516` recommends it only for slow links.

**This is the useful part:** logit extraction in PP mode carries the *same* one-row
limitation as single-machine (it is the same `ds4_session_eval_layer_slice` code path,
`ds4.c:59698`), so the batched-head patch from §3.2 is still required. But it also means:

> **The teacher-extraction patch can live entirely on the node that owns the output head,
> writing top-K to that node's local disk. No wire-protocol change is needed** — teacher
> outputs need to reach storage, not the coordinator.

That materially shrinks the patch surface: no new message type, no framing change, no
version bump, no coordinator work.

Per-position **hidden states are already crossing the wire** for the whole chunk
(`hidden_bytes = n_tokens * hc_values * 4`, `ds4_distributed.c:2685-2686`), so deliverable
(b) is even closer in PP mode than single-machine.

### 4.5 KV in PP mode

Sharded, as we want: each node opens its engine with `load_slice` / `load_layer_start` /
`load_layer_end` (`ds4_distributed.c:8375-8382`) and sizes context memory with the
slice-aware estimator (`ds4.c:39192-39199`). Consistency is maintained by mirroring the
**token timeline**, not KV bytes: every WORK frame carries a rolling 64-bit prefix hash
and the expected post-hash (`ds4_distributed.c:92-98`), and a worker refuses work whose
prefix hash does not match its own KV state (`:7492-7499`). A restarted worker cannot
silently accept work for the wrong position. Snapshots are topology-neutral — gathered on
save, re-split on load (`ds4_distributed.h:103-118`).

### 4.6 Operational reality

Launch is two commands with hardcoded addresses (`README.md:495-508`):

```sh
# coordinator, owns tokenization, sampling, the prompt, and layers 0..30
./ds4 -m <model.gguf> --role coordinator --layers 0:30  --listen 169.254.43.68 1234
# worker
./ds4 -m <model.gguf> --role worker --layers 31:output --coordinator 169.254.43.68 1234
```

Validation is strict and fails loudly (`dist_validate_options`, `ds4_distributed.c:8303`;
`dist_validate_layers_for_model`, `:8388`): coordinator must start at layer 0
(`:8404-8407`), the route must be contiguous and reach the last layer (`:6187-6190`), HELLO
carries `model_id` / `quant_bits` / layer range / ctx (`:1762-1788`), and a worker with a
smaller ctx is rejected (`:4233-4243`). Because `load_slice` maps only the requested range,
**both machines can hold the same full GGUF on disk and map different halves** — no
pre-split artifact needed (the `pro-q4-layers*` split files exist because 430 GB does not
fit comfortably on disk, not because the protocol needs it).

Recovery is genuinely engineered: socket death removes the worker and rebuilds the route
(`:4095-4142`), and `dist_coordinator_rebuild_from_transcript` (`:2948`) distinguishes
"remote error → replay on the same route" from "transport error → drop route and wait"
(`forget_route = rc != DS4_DIST_RECV_REMOTE_ERROR`). For a multi-hour unattended teacher
run this matters.

### 4.7 Measured two-Mac prefill

`README.md:434-438`, two M5 Max 128 GB over Thunderbolt 5, Q4 Flash split, default 4,096
chunk. **Caveat stated in the README itself (`:428-433`): the single-process reference
column is Q2 on one machine, so the baseline is slightly favoured.**

| Prompt | Single-process ref (Q2, 1 Mac) | Two Macs (Q4, PP) | Speedup |
| ---: | ---: | ---: | ---: |
| 9,421 | 421.70 tok/s | 582.22 tok/s | 1.38x |
| 28,684 | 405.30 tok/s | 674.16 tok/s | 1.66x |
| 63,819 | 353.62 tok/s | **654.79 tok/s** | **1.85x** |

**The speedup grows with context** — exactly the regime our 81,920-token windows live in.
Link sensitivity (`README.md:527-531`, 8,192-token prompt): Thunderbolt 5 (0.45 ms)
582.99 tok/s; WiFi (77.20 ms) 250.70; VPN (152.10 ms) 114.88.

### 4.8 Assessment for our M5 Max + M4 Max pair

**Verdict: viable, and it is the only configuration that makes a local teacher run
sensible. But it is a *pipeline-parallel, Thunderbolt-cabled, resident* deployment — not
the RDMA tensor-parallel one the brief anticipated, and not a streaming one.**

*Fit.* The faithful `ds4f-mxfp4` artifact is ~156 GB (~164 GB fully faithful, §3.6). Split
across two 128 GB machines that is **~78–82 GB of weights per node**, plus a KV slice
(~0.65 GiB, half of §5's 1.3 GiB), indexer staging (≤5 GiB at chunk 32,768) and graph
scratch. Comfortable, with room to raise the prefill chunk. Requires the wired-memory bump
(`sudo sysctl iogpu.wired_limit_mb=120000`, `README.md:636-638`).

*Throughput.* The measured pair is 2 × M5 Max; ours is M5 Max + M4 Max, and the M4 is
**~2.3x slower at prefill across the whole sweep** (§2.3). PP throughput on a balanced
split is approximately additive — consistent with the measured 1.85x on a matched pair. A
layer split weighted ~63/37 toward the M5 should land near **353 + 207 ≈ 560 tok/s at
~64k context**, i.e. ~1.6x the M5 alone. `--layers` is exactly the knob for this; tune it
by watching the per-token local/remote layer telemetry the coordinator prints
(`README.md:478-484` shows the balanced-split example).

At ~550 tok/s, **11.7 M tokens ≈ 5.9 hours** of prefill. That is the number that makes a
local run thinkable.

*Costs and risks.*

- **Decode is 19.4% *worse*** (`README.md:444-449`). Irrelevant for a prefill-only teacher
  pass; disqualifying if the run needs generation.
- **PP is TCP-only.** The brief's "RDMA fallback" framing applies to TP, which we cannot
  use (it rejects `--ssd-streaming`, replicates KV, needs an ownership-aware IQ2/Q2 routed
  layout for the GLM path, and has no fault tolerance). Use a Thunderbolt 5 cable and plain
  TCP over the link-local address.
- **Unverified for MXFP4.** Every committed two-Mac number is Q4 Flash (PP) or IQ2_XXS GLM
  (TP). Nothing exercises MXFP4 across a pipeline. **Smoke-test one session before
  planning around it.**
- **Two machines, same commit, no auth, manual per-boot network setup.** Acceptable in a
  lab; budget an afternoon of yak-shaving.
- **`README.md:704-705`** notes the split graph's changed floating-point reduction order is
  not byte-identical to single-machine execution. That is stated for TP; for a *teacher*
  artifact any reduction-order change is worth a parity check against a single-machine run
  on a short prompt before generating 11.7 M tokens of supervision.

---

## 5. KV compression at our 81,920-token window

**Mechanism.** V4-Flash uses layer-dependent time-axis KV compression on top of a raw
sliding window (`MODEL_CARD.md:25-50`), and ds4 hard-codes the shape:

| 0-based layers | ratio | extra state |
| --- | ---: | --- |
| 0, 1 | none | raw 128-token sliding window only |
| even ≥ 2 | 4 | compressed KV **+ indexer KV**, top-512 selection |
| odd ≥ 3 | 128 | compressed KV only |

Constants from `DS4_SHAPE_FLASH` (`ds4.c:540-575`): `n_layer` 43, `n_head_dim` 512,
`n_swa` 128, `n_indexer_head` 64, `n_indexer_head_dim` 128, `n_indexer_top_k` 512.

**The formula** (`metal_graph_kv_cache_bytes_for_context`, `ds4.c:16109-16126`):

```c
bytes = n_layer * raw_cap * n_head_dim * sizeof(float);
for each layer with ratio != 0:
    comp_cap = ctx / ratio + 2;
    bytes += comp_cap * n_head_dim * (F16 ? 2 : 4);
    if (ratio == 4) bytes += comp_cap * n_indexer_head_dim * sizeof(float);
```

`raw_cap` is `align_up(n_swa + prefill_cap, 256)` clamped to **8,192**
(`engine_planner_raw_cap`, `ds4.c:48702-48718`) — it does *not* grow with context. The
compressed cache is F16 on Metal (`DS4_GPU_ATTN_COMP_CACHE_F16 1`, `ds4.c:14953`).

**At ctx = 81,920:**

| Component | Count | Bytes |
| --- | ---: | ---: |
| Raw window (all 43 layers, `raw_cap` 8,192) | 43 | 688 MiB |
| Ratio-4 compressed KV + indexer (`comp_cap` 20,482) | 21 layers | 630 MiB |
| Ratio-128 compressed KV (`comp_cap` 642) | 20 layers | 13 MiB |
| **Total KV state** | | **≈1.3 GiB** |

**Does it matter at 81,920? Enormously — but as a gift, not a problem.** An 80k context on
a conventional GQA model would cost tens of GB. Here it is **~1.3 GiB**, and the raw-window
term (the largest of the three) is *context-independent*. Practical consequences:

- **Context is not a memory constraint for us at all.** The entire 128 GB budget can go to
  weights and the expert cache. Long windows are cheap; do not shorten sessions to save
  memory.
- **The one term that does scale is the indexer staging buffer**, `2 × comp_cap ×
  prefill_cap × 4 B` (`ds4.c:16143-16144`) — see the table in §1.9. This, not the KV cache,
  is what bounds the prefill chunk.
- **Compressed attention is a distillation-fidelity question, not just a speed one.** The
  ratio-4 layers *select* up to 512 compressed rows via the indexer. Teacher logits
  therefore depend on the sparse-selection path. The campaign already flags this as Wave 3
  step 5 ("DSpark divergence probe … teacher logits sparse vs full attention"); this
  analysis confirms the concern is real and structural, and that ds4 implements the sparse
  path as the default.

---

## 6. `dir-steering/`

Runtime **activation steering**, not disk streaming: a 43 × 4096 f32 file of one normalized
direction per layer, applied as `y = y - scale * d[layer] * dot(d[layer], y)` after FFN
and/or attention outputs (`dir-steering/README.md:1-22`; flags `--dir-steering-file`,
`--dir-steering-ffn`, `--dir-steering-attn`, `ds4_help.c:221-223`). Irrelevant to the
teacher run **with one exception worth noting**: its extractor
(`dir-steering/tools/build_direction.py:88-99`) works by setting
`DS4_METAL_GRAPH_DUMP_PREFIX` / `_NAME` / `_POS`, which is how I found ds4's generic
graph-tensor dump hook (§7).

---

## 7. Bonus finding: the graph dump hook

`metal_graph_debug_dump_tensor` (`ds4.c:16228`) writes any named graph tensor to
`<prefix>_<name>-<layer>_pos<pos>.bin` as f32, filtered by
`DS4_METAL_GRAPH_DUMP_PREFIX` / `_NAME` / `_LAYER` / `_POS`
(`metal_graph_debug_get_config`, `ds4.c:16189-16213`). ~40 tensors are instrumented,
including `attn_out`, `ffn_out`, `ffn_moe_logits`, `ffn_moe_probs`, `result_norm` and
`result_output` (full vocab) — see the call sites at `ds4.c:22163-25231`.

Crucially, the **batched** call sites dump all rows: `ds4.c:30455` dumps
`n_tokens × DS4_N_EMBD` of `batch_ffn_out`. So per-position intermediate activations are
already extractable *without any patch* — for research-scale batches.

Two caveats that rule it out for production teacher extraction: each dump forces
`ds4_gpu_synchronize()` and restarts the Metal command batch (`ds4.c:16244-16259`), and
setting `DS4_METAL_GRAPH_DUMP_PREFIX` **disables the streaming layer-batch optimisation**
(`ds4.c:17584-17585`) and several other fast paths (`ds4.c:61690`, `:62212`, `:63140`,
`:63291`, `:63439`, `:65413`). It is a debugging instrument. But for the **DSpark
divergence probe** (Wave 3 step 5) on a 10-session subset, it is exactly the right tool and
costs zero engineering.

---

## 8. Transferable ideas, ranked by expected impact on our teacher run

Ranked by expected value **for the Wave 3 local run specifically**, with portability to an
MLX/Python streaming runner called out.

### Rank 1 — Router-first selected-expert coalesced page-in

*What:* read back the router's selected-expert ids for the whole chunk **before** the FFN
needs the weights, build the set of touched experts, coalesce them into contiguous byte
ranges (with a gap parameter that bridges short unselected runs), and issue a small number
of large reads. `ds4.c:18680-18830`.

*Why it wins:* it converts "256 random 12.75 MiB scattered reads per layer" into a handful
of long sequential ones. Given the ~86 MB/s effective bandwidth implied by
`README.md:695`, random-access collapse is the entire streaming problem.

*Portable?* **Fully.** MLX gives us the router output as an array; the coalescing and the
`pread` scheduling are plain Python/NumPy + `os.preadv`. **Do this first.**

### Rank 2 — Maximize the prefill chunk

*What:* streaming cost scales with the *number of chunk sweeps* over the expert set, while
the only counter-scaling term (indexer staging, `2 × comp_cap × prefill_cap × 4 B`,
`ds4.c:16143`) is small in absolute terms. §1.9 table: chunk 4,096 → 32,768 cuts routed
reads per 81,920-token session from ~2.7 TiB to ~0.40 TiB for +4.4 GiB of staging.

*Portable?* **Fully, and it is a one-line policy choice in our runner.** Highest
value-per-effort item in this document.

### Rank 3 — Persistent I/O worker pool with a generation counter

*What:* up to 18 threads created once, coordinated by a generation counter + two condvars,
pulling from a shared task index. `ds4_metal.m:12004-12200`.

*Why it wins:* the per-layer loop dispatches a fresh task set 43 times per chunk; thread
creation at that rate would dominate. Deep queue depth is also what gets an NVMe SSD near
its rated bandwidth.

*Portable?* **Yes, with care.** Python threads are fine here because `os.pread` releases
the GIL. A `ThreadPoolExecutor` created once at runner startup, ~8–16 workers, is the
direct analogue. Do **not** use a process pool — the point is shared destination buffers.

### Rank 4 — Aging-counter (LFU-with-decay) eviction with a protected set

*What:* victim = lowest route-hotness, LRU tiebreak; hotness incremented per routing
selection and **halved periodically**; currently-routed experts exempt.
`ds4_metal.m:14005-14070`, decay at `:12867`, protection at `:14031`.

*Why it wins:* pure LRU thrashes badly under MoE routing, where a small hot set coexists
with a long uniform tail. The halving is what stops early-corpus experts pinning forever.

*Portable?* **Fully** — it is a dict + a counter array. **Improve on it while porting:**
ds4's victim search is a full linear sweep of 43 × 256 entries per eviction
(`ds4_metal.m:14013-14047`). Use a bucket queue or a small heap.

### Rank 5 — Reserve N full routed layers as prefill headroom, then size the dynamic cache

*What:* subtract `per_expert_bytes × n_expert × 2` from the budget *before* computing the
dynamic cache, and fail loudly if the budget cannot cover it.
`ds4.c:4567`, `:4600`, `:55488-55505`.

*Why it wins:* it guarantees the overlapped-prefill invariant (layer `il` resident while
`il+1` loads) instead of hoping the cache happens to hold it. The loud failure is good
design — silent under-provisioning would show up only as a mysterious slowdown.

*Portable?* **Fully** — it is budget arithmetic. Consider raising the reservation to match
a deeper lookahead: ds4 allows prefetch depth up to 4 (`ds4.c:18662`) but only reserves 2,
which looks like a latent mismatch.

### Rank 6 — Regenerate the hot-expert list against *our* corpus

*What:* `--expert-profile FILE` (`ds4_help.c:193`) produces a per-`(layer, expert)` hit +
weight histogram, adjacency statistics and a multi-capacity cache-hit simulation
(`ds4.c:1232-1259`), emitted sorted by hits (`ds4.c:1540-1577`).

*Why it wins:* the baked hotlists have **unknown corpus provenance** (§1.6). Our
distillation corpus is code-agent-heavy and will not match. A single profiling pass over a
sample of the 257 sessions gives a hotlist matched to the actual workload — and the
cache-hit simulation tells us the right cache size *before* committing to a long run.

*Portable?* The **collection** can be done by ds4 today and the output consumed by our MLX
runner — a rare case where we can use the C engine as a measurement instrument without
depending on it at runtime.

### Rank 7 — Buffered I/O as a deliberate second-level cache

*What:* no `O_DIRECT` / `F_NOCACHE` anywhere; instead `MADV_DONTNEED` on eviction
(`ds4_metal.m:12780`) to stop the page cache retaining dead slots.

*Why it wins:* on a 128 GB machine, free RAM outside the explicit cache still absorbs
re-reads. Fighting the page cache would be a mistake.

*Portable?* **Fully** — it is a decision not to do something, plus `posix_fadvise(DONTNEED)`
on eviction.

### Not portable — noted for completeness

- **Zero-copy `newBufferWithBytesNoCopy` over the file mmap** (`ds4_metal.m:1881`,
  `:11507`). This is the deepest idea in ds4 and has **no MLX equivalent**: MLX arrays own
  buffers allocated from its own Metal heap, and there is no public API to wrap an existing
  file mapping as an `mx.array` without a copy. Our runner must copy into MLX-owned buffers.
  This is the single biggest structural advantage the C engine retains, and it is the
  honest argument for using ds4 rather than porting.
- **`F_RDADVISE`** (`ds4.c:17588`) — Darwin-specific `fcntl`. Reachable from Python via
  `fcntl.fcntl` with a hand-packed `struct radvisory`, but `posix_fadvise`/`madvise`
  equivalents cover most of the benefit.
- **Single-size-class slab allocator over `MTLBuffer`s** (`ds4_metal.m:684-699`) — the
  mixed-precision caveat at `ds4.c:4630-4636` is a warning, not a pattern to copy. Our VQ
  artifacts are heterogeneous by design; a size-class-per-layer pool is the right shape.

---

## 9. Recommendations for the Wave 3 local-run design

### 9.1 The strategic point

Wave 3 as planned rents one 24 h p5.48xlarge (~$997) and does everything on an 8×H100 node
with the model resident in HBM. The local alternative was framed as "SSD-stream the teacher
on one M5 Max". **This analysis says that framing is wrong in a specific, useful way:**

- **Single-Mac SSD streaming is not a viable teacher path.** Not because streaming is badly
  built — ds4's streaming design is genuinely good (§1) — but because the measured penalty
  is 20–200x (§2.4), which puts 11.7 M tokens in the *weeks* range against ~8 h resident.
- **Two Macs holding the model resident across a pipeline split is a viable path**, at
  ~5.9 h of prefill (§4.8), which is the same order as the AWS block's 6 h teacher-cache
  budget.
- **But ds4 cannot emit the teacher outputs without a patch** (§3.7), and that patch lands
  inside the most actively-changing part of an unfamiliar 2.9 MB C file with no upstream
  relationship.

So the real decision is not "AWS vs local streaming". It is **"AWS block" vs "two-Mac
resident PP + a ds4 fork"**, and the AWS block still wins on schedule risk — it needs no
new engine work at all.

### 9.2 Recommended plan

**Keep Wave 3 on AWS for the logit/hidden-state/MTP deliverables.** The $997 block buys a
resident FP8 model, a known-good extraction path, and no C forking. Do not re-plan it
around ds4.

**Pull three things forward to the local machine, where ds4 is a net win today:**

1. **Calibration statistics / imatrix — run locally now, free.** `--imatrix-dataset` /
   `--imatrix-out` works on the batched prefill path, propagates the streaming flag
   (`ds4.c:52322-52324`), and needs zero code changes (§3.5). This removes Wave 3 step 4
   (2 h of block time) from the rental. *Caveat: it produces per-expert MoE-input second
   moments only — confirm that is what "calibration statistics" means before deleting the
   AWS step.*

2. **The DSpark divergence probe (Wave 3 step 5) — run locally now, free.** The graph dump
   hook (§7) gives per-position `ffn_moe_logits`, `ffn_moe_probs`, `attn_out`, `ffn_out`
   and full-vocab `result_output` on a 10-session subset with three environment variables
   and no patch. The probe is explicitly a research question, not a production artifact, so
   the dump path's fast-path disabling does not matter. This removes another 1 h from the
   block.

3. **Regenerate the expert hotlist against our corpus — do this regardless.**
   `--expert-profile` (§1.6, ranked #6 in §8) gives per-`(layer, expert)` hit and weight
   histograms *plus a multi-capacity cache-hit simulation*, over a sample of our own 257
   sessions. The baked hotlists have unknown provenance and our corpus is
   code-agent-heavy. This output feeds our own MLX runner directly — ds4 as a measurement
   instrument, with no runtime dependency.

**Build our MLX streaming runner around the four portable ideas, in rank order (§8):**
router-first coalesced page-in, maximal prefill chunk, a persistent `os.pread` thread pool,
and aging-counter eviction with a protected set. Ranks 1 and 2 are where nearly all the
value is, and rank 2 is a one-line policy choice.

**Set the prefill chunk aggressively.** Streaming cost scales with the number of chunk
sweeps over the expert set; the only counter-scaling term is
`2 × comp_cap × prefill_cap × 4 B` (`ds4.c:16143`), which is ~5 GiB even at chunk 32,768
against an 81,920 window. Default 4,096 → 32,768 cuts routed reads per session from
~2.7 TiB to ~0.40 TiB (§1.9). Nothing in ds4's own benchmarks explores this.

**Do not shorten the 81,920-token windows to save memory.** Total KV state at that window
is **~1.3 GiB** (§5), and the largest component (the raw sliding window) does not grow with
context at all. Context is essentially free on this architecture.

### 9.3 If the two-Mac local run is pursued anyway

Order of operations, cheapest disqualifying test first:

1. **Download `ds4f-mxfp4` and verify it loads.** It is the only artifact whose fidelity to
   the source checkpoint is verified in code (§3.6). If a fully faithful variant is wanted,
   rebuild with `deepseek4-quantize --experts mxfp4 --attention-proj f16 --shared f16
   --output f16` (+~8 GB).
2. **Smoke-test PP with MXFP4 across the two Macs on a short prompt.** Nothing in the repo
   exercises MXFP4 across a pipeline (§4.8). Compare logits against a single-machine run;
   `--dist-replay-check` (`ds4_help.c:239`) exists for exactly this.
3. **Measure one real 81,920-token session** before planning 257. No committed Mac
   benchmark goes past 65,536 tokens and the M5 Max curve is still falling there (§2.2).
4. **Tune `--layers` for the M5/M4 asymmetry** (~63/37 toward the M5), using the
   coordinator's per-token local/remote layer telemetry.
5. **Only then** write the patch: batched output head (widen `spec_logits` past 16 at
   `ds4.c:17315`, hoist the single-row head at `ds4.c:35047`/`:35257`), on-GPU top-2048
   reusing `ds4_gpu_indexer_topk_tensor`, and a binary writer — **on the node that owns the
   output head, writing to local disk, with no wire-protocol change** (§4.4).

Steps 1–4 cost days and can kill the plan cheaply. Step 5 is the only expensive one and it
comes last.

### 9.4 Storage note

Per-position hidden states are 16,384 floats = **64 KiB/position** (§3.3, `ds4.c:58122`).
120 sessions × 81,920 tokens ≈ **630 GB at fp32, ~315 GB at fp16**. Top-2048 logits at
fp16 + int32 ids are ~12 KB/position → 257 × 81,920 × 12 KB ≈ **250 GB**. Local disk free
is 1.4 TB: workable at fp16, tight if anything is kept at fp32. Plan the dtype before the
run, not after.

---

## 10. Open questions and risks

1. **No Mac prefill data exists beyond 65,536 tokens.** Our window is 81,920. The M5 Max
   curve is still falling at 64k (398 tok/s, 50% of its 2k rate) and the only >64k evidence
   is an unlabelled SVG. **Measure a single 81,920-token session before committing to a
   257-session plan.**
2. **Hotlist provenance is undocumented** (§1.6). Do not assume the baked Flash hotlist
   transfers to a code-agent distillation corpus.
3. **The README's "long prefills can still be fast" claim is contradicted by its own
   measurement** (`README.md:296-301` vs `:695`). Trust the number.
4. **Sparse vs full attention in the teacher signal.** Ratio-4 layers select ≤512
   compressed rows via the indexer. Teacher logits are conditioned on that selection.
   Wave 3 step 5 already plans the probe; §7 shows the graph dump hook makes it nearly
   free.
5. **Hidden-state volume.** 16,384 floats/position = 64 KB/position (§3.3). 120 sessions ×
   81,920 tokens ≈ 630 GB at fp32. Storage and transfer planning is a bigger problem than
   the extraction patch. Local disk free is 1.4 TB — workable at fp16, tight at fp32.
6. **`ds4_eval.c` is not a perplexity tool.** It is a multiple-choice / security-CWE
   benchmark harness (its only logit contact is ranking `</think>` for a soft-close
   heuristic, `ds4_eval.c:2540-2546`). Do not plan around it.
7. **Patching ds4 means owning a fork.** Single squashed commit, no upstream relationship,
   2.9 MB `ds4.c`, and the paths we need to change (`spec_logits` sizing, the output-head
   row loop) sit inside the speculative-decoding machinery, which is the most actively
   changing part of the tree (HEAD commit is a DSpark change).
