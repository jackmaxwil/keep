# What KEEP has learned, and which of it could go beyond the prior art

Written 2026-11-09 for the paper track (*Offload the Residual, Not the Expert*). Jack asked what the
whole project has measured that may not be in the literature, and whether any win is a bigger lever
than it looked. Three sources: this campaign's own results
([recovery log](2026-11-08-v41-reap-recovery-log.md)), a survey of the older KEEP documents, and the
prior-art search in the [plan](2026-11-09-routed-precision-plan.md). The scheduled experiments (X1–X7)
are in section 7 of the plan; this file holds the evidence and the reasoning.

Nothing here has been checked against the literature beyond the six papers in the plan. Each lever
gets its own search before it is claimed.

## 1. Findings from this campaign

Differences are domain-mean KL on the Mac eval set (22 sessions, first 1,024 tokens), with a paired
block bootstrap over positions (`routed_report.py --compare`). The interval covers position noise
within the 22 sessions, not which sessions were drawn.

| Finding | Number | 95% interval |
|---|---:|---|
| Choosing 96 experts by total routed mass on the pruned model's routing, against REAP-75's set, both at 2 bits | −0.177 KL | −0.200 to −0.148 |
| A second prune pass (192 → 144 → 96) over one pass | −0.024 | −0.034 to −0.012 |
| Static allocator at 288 units against 144 experts at 2 bits | −0.003 | −0.009 to +0.005 |

- **Selection on the pruned model's routing is the lever; precision allocation was not.** Each of the
  two changes alone (the ranking rule, or measuring on the pruned model) gave about 0.025. Together
  0.177. A second pass is statistically real but small, most of it on tool-use.
- **Quantization statistics survive pruning; routing does not.** Refitting the 2-bit codes with
  importance measured on the pruned model changed nothing (0.753 against 0.747). The importance
  cosine was 0.976; the selection overlap with REAP-75 was 58%.
- **Losses are not additive.** KEEP 2-bit costs +0.17 KL on REAP-50 and +0.02 on a 96-expert model.
- **Two regimes by depth.** The top 96 experts hold 72–83% of routed mass in layers 0–19 and 94–98%
  in layers 20–39. Yet moving budget toward the diffuse layers lost to a uniform budget (1.046
  against 0.955 at 128 experts): concentration is not sensitivity.
- **A static 1-bit tier is nearly worthless** (0.025 KL at 192 units, nothing at 288). The 1-bit
  code's error is 0.31 per projection.
- **Nested lattice codes cost about 12%.** A 1-bit E8 residual on a 1-bit E8 base scores 0.101
  (gate/up) against 0.090 for a flat 2-bit E8P fit.
- **The pruned model beats the full model on tool-use**, by 1.22 nats per token of NLL, on the pod's
  own student as well as locally. Not explained.
- **Sidecars (running):** a rank-256 linear correction removes 36–76% of the layer gap where the
  gap is small (layers 0–3) and 9–15% where it is large (layers 4–6).

## 2. Findings from the older KEEP documents

From a survey of every document under `docs/` and the module docstrings. Rows marked ✓ were
re-read at the cited line for this note; the rest are as the survey reported them.

### Memory and speed come from different weights

- ✓ On V4-Flash, decoding one token reads 15.27 GB: attention 67%, routed experts 11%
  (`2026-08-21-roadmap.md:116-121`). Residents are dequantized to bf16 at bind.
- ✓ On V4.1, each expert kept in every layer costs 0.359 GB of RAM at 2.031 bits per weight "but adds
  no per-token" reads, and fitting the Mac "forces a ~50% prune"
  (`2026-11-05-v41-plan-and-phase0-status.md:15-21`).
- GLM-5.3 inverts this: residents are about 85% of decode traffic (`glm-5.3-flash/2026-10-06-plan.md`).

So on the DeepSeek family, expert precision buys memory and almost no speed. MatMoE argues dynamic
precision from latency; that argument does not carry to this architecture, and the memory argument
is the one this paper makes.

### The Mac thrashes at a 91 GB resident set

- ✓ At 90.99 GB of MLX-active memory the bind alone logged 535,792 swapouts (8.4 GB of anonymous
  buffers), and byte-identical work spanned 16.5× in time across rows that all passed the quiet
  gate (`2026-08-21-roadmap.md:84-89, 208-232`). The document's own conclusion is that the timings
  were "memory weather, not decode".
- Policy since then: leave the wired limit at its default (`docs/troubleshooting.md`).

Anonymous memory at this footprint gets compressed and swapped, which writes to the SSD and stalls
decode. Clean file-backed pages cannot be swapped; they are dropped and read again. That is the
case for lever X4.

### The quantizer

- ✓ A 1-bit code gave a 0.56 block cosine, "not a usable model"; the SwiGLU amplifies projection error
  about 2.8× at block level; E8P beats affine quantization below about 3.1 bits per weight and
  loses above; 2 bits is "at its rate-distortion limit" (`research/wave5-pilot-report.md:12-14, 122,
  215, 305, 339`).
- ✓ Incoherence processing (the random rotation QuIP# is built on) made the importance-aware fit 15%
  worse on the down projection, 9.18e-2 against 7.97e-2, because the rotation flattens a per-column
  importance that spans 180× (`research/wave5-pilot-report.md:144-149`).
- The codebook step returns 0.71 of fractional error per bit per weight and group-size steps
  0.17–0.24, so group size is the rate dial. Uniform group size beats non-uniform at matched rate.
  The belief that the down projection needs extra protection was not supported.
- Only 1 of 11,008 (layer, expert) pairs had no calibration routes.
- An fp32 reduction flips near-tie argmins in about 30% of scale last bits, so exact fits keep
  float64 sums on the CPU.

### Checks that lie

- ✓ A gate/up-swapped artifact loads, gives finite logits and scores cosine 0.9993 against the correct
  one; only a discrimination ratio (4.3e10) caught it (`research/wave5-materializer-report.md:12,
  101`).
- A per-call-sync benchmark harness produced false kills: the E8P kernel read 52.8 GB/s and was
  marked KILL, then 122–126 GB/s chained and 201.9–205.5 GB/s tuned (`2026-11-05-phase0-kill-tests.md`).
- Two "slower than source" verdicts were bugs in the measuring path: a Python-version mismatch that
  made about 50 kernels unreachable (8.36× once fixed), and a loader that re-globbed the build tree
  on every call (4.53× once fixed) (`handoff.md`, `2026-08-21-compounded-speed-verdict.md`).

### Recovery levers already tried on GLM-4.5-Air

- ✓ Router distillation was rejected: best global top-1 change "slightly negative"
  (`glm45-air/rc-pipeline.md:176`).
- ✓ A rank-4 sidecar raised report top-1 from 0.795 to 0.852 and cut KL to 0.314, but ran 1.718× the
  control and failed the speed gate (`glm45-air/rc-pipeline.md:167, 895, 930`). A proxy win that
  failed end to end.
- Raising non-expert precision changed nothing once a loader bug was fixed.

### Streaming and I/O

- A layer-outer, chunk-inner teacher loop is bit-identical to chunk-major and avoided 58.8 h of I/O.
- Prefill I/O favoured VQ: 6 `preadv` calls per layer against 1,536; the gap to the source model was
  compute, not I/O.
- Engram lookups depend only on token ids; the tables are memory-mapped and gathered on the host.
  The V4.1 plan already keeps Engram's 203 GB on the SSD.
- ✓ A reference point from outside: the Rapid-MLX V4.1 build reports teacher-forced KL 0.26 and top-1
  85% against the BF16 source, at 34–49 tok/s on a 256 GB M3 Ultra
  (`2026-11-08-v41-reap75-local-test-plan.md:125-126`).

## 3. What looks like it could go beyond the prior art

In the order I would bet on them. X numbers refer to the plan.

1. **Prune twice, at the first cut (X1).** The one clearly significant win of this campaign has only
   been applied to the second halving. REAP-50's own KL (0.578) is the floor under every number
   here, and it was chosen the old way. If the effect repeats at 384 → 192 it lifts the whole
   frontier, and "measure routing on the pruned model, rank by total mass" is a two-line change to
   REAP that anyone can use. Needs the full model: a pod run.
2. **Keep every expert and page residuals instead of pruning half (X3).** The V4.1 plan prunes 50%
   because each kept expert costs 0.359 GB and the model has to fit. All 384 one-bit bases cost the
   same memory as 192 experts at 2 bits. Whether 384 bases plus a residual cache beats REAP-50 at
   2 bits at equal memory is the title's claim at full scale, and no paper found so far compares
   pruning against residual offload. The local sweep (192 bases against 96 or 144 at 2 bits) is its
   proxy; the weak 1-bit base is the risk.
3. **The OS page cache as the precision cache (X4).** Wire the bases; map the residuals as clean
   file pages; `mincore` before use, `madvise` to fetch. No cache code, and under memory pressure the
   model loses precision instead of swapping. KEEP has already measured the failure this avoids
   (535,792 swapouts, 16.5× timing spread at 91 GB). This is specific to how unified memory and the
   macOS VM behave, and it turns "SSD churn" from writes into reads.
4. **Pruning as specialization (X2).** If the tool-use result holds up against the resident full
   model and on a downstream benchmark, a calibration-targeted prune is not only smaller but better
   on its target domains. Large if true; the prior is that it is partly a teacher artifact.
5. **One artifact for every budget.** Because quantization statistics survive pruning and one mass
   ranking yields nested expert sets (pm-144 ⊂ pm-160; 97% overlap between passes), a single set of
   files can serve any memory budget: nested in expert count and in bits. MatMoE nests bits only.
   This is a packaging result that follows from 1–3 rather than an experiment of its own.
6. **Finer rungs and a stronger base (X5).** The 1-bit base is the weak link by three separate
   measurements. Projection-level residuals give 1.33- and 1.67-bit rungs from the existing codes,
   and a 2-bit base with a 1-bit residual moves the whole ladder to where the base is usable.
7. **Sensitivity-based budgets (X6)** and **a shared base across experts (X7)** are cheaper to test
   than to argue about.

Smaller results that are probably worth a paragraph or a short note each: incoherence processing
hurting importance-aware lattice fits; cosine similarity passing a gate/up-swapped model; and the
measurement lessons (per-call sync, one-sided paging gates).

## 4. A correction to my own earlier claim

I first wrote that a nested cache "holds twice as many hot experts" as a separate-copy cache at
equal memory. That is true only if the separate-copy cache keeps the low-precision copy of every hot
expert. One that drops it and reads it back on demotion uses the same memory as the nested design;
what it pays is traffic, three units per promote/demote cycle against one, and a slightly more
accurate high tier in return (0.090 against 0.101). The comparison is therefore run both ways, and
at equal SSD rate as well as equal memory. The title's claim may turn out to be about traffic and
simplicity more than about memory.
