# DSV4-Flash compounded speed verdict

Date: 2026-08-21. Basis: committed Task 2 and Task 4 evidence on this machine
(M5 Max laptop, 128 GB), `main` at `433ec34d` plus the Task 5 fix round.

> ## SUPERSEDED IN PART — read this first
>
> The 0.0934906005x component ratio below was **not measuring E8P kernel
> performance**. It was measuring a Python import bug: `load_native()` in
> `src/mlx_vq/kernels/nax.py` never registered `sys.modules["_vqnax"]`, so its
> `importlib.import_module` fast path was dead and every call re-globbed the
> 78-file build tree (including a redundant recursive `build/**/_vqnax*.so` that
> re-matched the same path) and re-executed the 3.5 MB extension. The M=1 E8P path
> hits that three times per projection.
>
> After a 4-line fix (`sys.modules` registration, `@cache` on `load_native`,
> `is_available`, `_repo_root`, `_kernel_dir`, and deleting the redundant glob),
> the same committed rail on 9 fresh processes, 9/9 clean, reports:
>
> | | Before | After | Change |
> |---|---|---|---|
> | median source/VQ ratio | 0.0934906005x | **0.4239291863x** | **4.53x better** |
> | VQ vs source | 10.70x slower | **2.36x slower** | |
> | MTP needed for parity | 10.70x | **2.36x** | |
>
> Recomposed with the same Task 4 MTP ratios:
>
> | MTP ratio | Composed vs native source | Verdict |
> |---|---|---|
> | worst clean pair, 1.4479585875x | 0.6138x | 1.63x slower |
> | **median, 1.7700938194x** | **0.7504x** | **1.33x slower** |
> | best clean pair, 3.2972944650x | **1.3978x** | **1.40x FASTER than source** |
>
> Quote **ratios only**. Absolute times drifted on the untouched source arm too
> (2.988 ms to 1.628 ms median) across sessions, which is thermal/machine state,
> not code. The paired per-row ratio is the only portable figure.
>
> Post-fix evidence: `~/keep-artifacts/dsv4-m1-postloaderfix-20260821/`. The gate
> still reads STOP, correctly — 2.36x slower is still slower. But parity is now a
> bounded dispatch problem, not a 10x wall, and the best observed MTP pair already
> composes to faster than source.
>
> Everything below is the pre-fix analysis, retained because the *method* stands
> and the Task 7 findings are unaffected.

## The number nobody wrote down

Tasks 1-5 produced two independent ratios that were each reported in isolation.
Composed, they are the campaign's actual speed position:

| Coordinate | Measured | Source |
|---|---|---|
| VQ E8P expert projections vs native source MXFP4, M=1, 9/9 clean rows | **0.0934906005x** (10.70x slower) | [m1-summary.json](../../artifacts/benchmarks/dsv4-task2-20260819/m1-summary.json) |
| MTP speculative vs autoregressive, both on the compressed model, 3/5 clean pairs | **1.7700938194x** (median) | [headline-quiet60-r3/summary.json](../../artifacts/benchmarks/dsv4-task4-fix1-20260819/headline-quiet60-r3/summary.json) |

The MTP win is measured *inside* the compressed model. It does not compete with
the source; it competes with compressed autoregressive decode. Composing the two:

| MTP ratio used | Composed vs native source | Verdict |
|---|---|---|
| worst clean pair, 1.4479585875x | 0.1354x | 7.39x slower than source |
| **median, 1.7700938194x** | **0.1655x** | **6.04x slower than source** |
| best clean pair, 3.2972944650x | 0.3083x | 3.24x slower than source |

**MTP would need a sustained 10.70x to reach parity with native MXFP4 source.**
The best single clean pair ever observed was 3.30x. The gap is 3.24x, and the
median case is 6.04x short.

Full-model prefill agrees in sign: 0.4451716951x of source throughput
([prefill-control-comparison.json](../../artifacts/benchmarks/dsv4-task2-20260819/prefill-control-comparison.json)),
though every row in that comparison was pageout-dirty and it is completeness
evidence only.

## Limits on this composition

- The 0.0935x coordinate is **layer-0 expert-projection component evidence**, not
  full-model decode. Residents did not exist when it was measured. Composing it
  with a full-model MTP ratio is an estimate, not a measurement.
- The 1.77x coordinate carries its own instability: baseline elapsed swung
  5.34 s to 42.17 s across rows that both passed the quiet gate, and 2 of 5 pairs
  were excluded dirty. Verify dispatch was 12 NAX vs 504 Metal calls per
  speculative row, so most of the verify window is not on the measured kernel.
- Neither coordinate has been superseded by a single same-machine full-model
  decode comparison of native source vs VQ+MTP. That measurement does not exist.
  Until it does, the composed figure is the best available estimate and it is
  negative.

## What this changes

1. **DSV4-Flash VQ is not a speed play at current kernel performance.** It is a
   memory-footprint and quality play: 84.687 GB resident+routed against a 163 GB
   source, fully resident, with Task 4 quality parity intact. That is the claim
   the campaign can defend.
2. **Do not publish 1.77x as a speedup** without the compressed-vs-source
   qualifier. Standalone it reads as "MTP made DSV4 fast"; composed it is 6x
   slower than just running the source.
3. **Task 7 outranks Task 6.** Converting the drafter's dense BF16 experts
   (~39 GB) to native MXFP4 (~10 GB) is the only queued item that moves real
   compute onto the fast kernel and removes ~29 GB of the memory pressure that
   contaminated every timing row in Tasks 2 and 4. Task 6's quiet-window harness
   only makes timing claims *cleaner*; it is worth building once there is a claim
   worth defending.

   Scope limit: Task 7 touches the **drafter** only. The 43 backbone VQ E8P
   layers keep the 10.70x kernel deficit. Task 7 improves the drafter half of
   the speculative loop and the memory environment — it does not by itself
   close the composed gap.
4. **The 10.70x is a kernel problem, not an architecture problem, and it is
   unqueued.** VQ E8P projections run 32.71 ms median against source MXFP4 at
   2.99 ms median for identical work (`nax_e8p_fp16_sorted_steel_m32n64`). No
   task in the current handoff targets that kernel. It is the single
   highest-leverage open item in the repo and should be scoped after Task 7.

## Task 7 as written is a no-op — and the real lever is its inverse

Measured against the pinned revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`,
the drafter has **no dense BF16 experts to convert**. Full `mtp.*` dtype census
from the checkpoint index and safetensors headers:

| dtype | tensors | bytes |
|---|---|---|
| `I8` (FP4 packed two-per-byte) | 2,304 | 9.664 GB |
| `F8_E8M0` (group-32 scales) | 2,329 | 0.604 GB |
| `F8_E4M3` | 25 | 0.447 GB |
| `F32` | 27 | 0.010 GB |
| `BF16` | 20 | 0.139 GB |
| **total** | **4,705** | **10.863 GB** |

All 2,304 `I8` tensors are routed expert weights (768 per block x 3 blocks,
256 experts x 3 projections). Every one of the 20 BF16 tensors is a norm, a
router gate, or a markov/confidence head — **zero BF16 expert tensors**. The
handoff's "~39 GB dense bf16" is 4x the 9.664 GB FP4 payload, i.e. the
hypothetical dequantized size, not a measured on-disk state. The "~10 GB native
mxfp4" target is what the checkpoint already ships.

The pipeline did the **opposite** of Task 7. The VQ sweep re-compressed the
already-native FP4 drafter into E8P:

| Drafter routed experts | Bytes | Kernel |
|---|---|---|
| source, native MXFP4 | 9.664 GB | fast native path, 2.99 ms median at M=1 |
| `dsv4-vq-e8p-g512` `mtp-0000{0,1,2}-*`, E8P | 4.907 GB | E8P path, 32.71 ms median at M=1 |

Drafter expert tensors are byte-identical in shape and dtype to the backbone
layer-0 experts that Task 2 measured (`[2048, 2048]` w1/w3, `[4096, 1024]` w2,
`I8` + `F8_E8M0`, 256 experts, group size 512, 16-bit codes). The 0.0934906005x
ratio therefore transfers directly — no new benchmark is needed to establish it,
and running one would only reproduce a number already committed.

So the campaign spent 4.757 GB of memory savings to move the drafter onto a
kernel measured 10.70x slower — on the component that executes on **every**
speculative step, at a 57.9% acceptance rate.

**Proposed replacement for Task 7:** bind `mtp.{0,1,2}` routed experts from the
source checkpoint's native FP4 payload instead of the E8P artifact, then re-run
the Task 4 headline series.

- Cost: **+5.3603 GB** resident, not +4.757 GB. Corrected 2026-08-21: the earlier
  figure subtracted 9.664 - 4.907 and dropped the 0.6040 GB of `F8_E8M0` group
  scales, which must be resident for `mx.gather_qmm(mode="mxfp4")`. Full source
  MTP routed payload is 10.2677 GB (9.6637 GB codes + 0.6040 GB scales) against
  the E8P artifact's 4.9073 GB. Steady-state parameter storage goes to ~96.35 GB.
- Benefit: the drafter's expert math moves to the native FP4 kernel.
- Risk, and it is now the dominant one: the committed Task 4 bind proof already
  recorded `swapouts_delta: 90,444` and `pageouts_delta: 443` *during the bind
  itself* at 90.99 GB active. Adding 5.36 GB puts steady state near 96 GB, and
  with the observed 3.56 GB MLX cache that is ~99.9 GB of a 128 GB machine. Every
  Task 2 and Task 4 timing row was already pageout-contaminated at the lower
  figure. This change may well return *fewer* clean pairs rather than a faster
  median. Transient peak during expert stacking adds ~3.42 GB per stage if done
  one projection at a time, or ~20 GB if the intermediate `mx.eval` is omitted.
- Silent-wrong hazard: the E8P artifact renamed `w1`/`w2`/`w3` to
  `gate`/`down`/`up` at materialization, so the rebind reintroduces the
  `w1 -> gate`, `w2 -> down`, `w3 -> up` mapping into the production path.
  Swapping `w1` and `w3` yields matching shapes, finite logits, and a wrong
  model. Only a decode-and-compare test against the raw source tensors catches it.
- Also touches the composite loader, the MTP binder, and the
  payload-authenticated receipt that Task 4 fix rounds 1-4 hardened. Requires a
  fresh bind proof.

This is a different change from the one the handoff specifies, with a real memory
cost, so it is recorded here rather than executed.
