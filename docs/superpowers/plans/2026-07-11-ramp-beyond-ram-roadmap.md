# RAMP "Beyond RAM" — Colibri-Inspired Runtime Program Roadmap

> **This is a program roadmap, not a task plan.** It sequences five sibling
> implementation plans and states the constraints every one of them inherits.
> Each linked plan is self-contained and produces working, testable software on
> its own. Execute plans in the recommended order; within a plan, use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans`.

**Origin:** Ideas lifted from [`JustVugg/colibri`](https://github.com/JustVugg/colibri),
a pure-C runtime that runs GLM-5.2 (744B MoE) on ~25 GB RAM by streaming routed
experts from disk. KEEP/RAMP solves the same "huge MoE on a small Mac" problem
from the opposite direction (compress experts to ~2 bpw, keep everything
resident). The program below fuses the two: **KEEP-compressed experts that are
also streamed from disk**, plus the surrounding runtime intelligence.

---

## Why five plans, not one

The seven ideas decompose into five plan-sized subsystems along a hard
dependency spine. F2, F4, and F6 cannot produce working software until F1's
tiered cache exists, so they are grouped into one "cache-intelligence" plan that
lands after F1. F3, F5, and F7 are independent.

```
                    ┌─────────────────────────────────────────┐
                    │  Plan A — Expert streaming (F1)           │  FOUNDATION
                    │  disk-backed tiered expert store + LRU     │
                    └───────────────┬───────────────────────────┘
                                    │ provides ExpertResidencyCache / ExpertStore
                    ┌───────────────▼───────────────────────────┐
                    │  Plan B — Cache intelligence               │
                    │  F6 RAM auto-size · F2 usage promotion ·    │
                    │  F4 predictive prefetch                     │
                    └────────────────────────────────────────────┘

   Plan C — KV-cache persistence (F3)   ── independent ──┐
   Plan D — MTP speculative decode (F5) ── independent ──┤  any order
   Plan E — Kernel parity harness (F7)  ── independent ──┘  (E first is safest)
```

| Plan | File | Features | Depends on |
| --- | --- | --- | --- |
| **A** | [`2026-07-11-ramp-expert-streaming.md`](2026-07-11-ramp-expert-streaming.md) | F1 disk-streaming | — |
| **B** | [`2026-07-11-ramp-cache-intelligence.md`](2026-07-11-ramp-cache-intelligence.md) | F6 RAM auto-size, F2 usage promotion, F4 prefetch | Plan A |
| **C** | [`2026-07-11-ramp-kv-cache-persistence.md`](2026-07-11-ramp-kv-cache-persistence.md) | F3 warm KV-cache | — |
| **D** | [`2026-07-11-ramp-mtp-speculative-decode.md`](2026-07-11-ramp-mtp-speculative-decode.md) | F5 MTP speculative decode | — |
| **E** | [`2026-07-11-ramp-kernel-parity-harness.md`](2026-07-11-ramp-kernel-parity-harness.md) | F7 bit-exact kernel parity | — |

## Recommended execution order

1. **Plan E (kernel parity)** — smallest, fully standalone, lowest collision risk
   with the in-flight codex campaign, and it is the numerical safety net you want
   *before* moving compressed experts on and off disk. Locks a golden reference.
2. **Plan A (streaming)** — the foundational bet; unblocks Plan B.
3. **Plan B (cache intelligence)** — makes the streamed cache smart (auto-size,
   pin hot experts, prefetch).
4. **Plan C (KV persistence)** and **Plan D (MTP)** — independent; schedule
   whenever. Plan C is the cheap, high-visibility "warm reopen" UX win.

---

## Global Constraints (inherited by every plan)

Copy these verbatim into each plan's own Global Constraints section. They come
from `AGENTS.md`, `README.md`, `docs/GLM45_AIR_RC_PIPELINE.md`, and the GLM-5.2
recovery handoff, and they are non-negotiable.

- **Runtime target:** Apple silicon Mac, macOS 14+, MLX. The active model is
  `GLM-5.2-REAP-KEEP-504B` (78 hidden layers, sparse MoE, `n_routed_experts`
  per profile); GLM-4.5-Air is the smaller secondary target.
- **The NAX speed wall (governs every runtime change):** the NAX fast prefill
  kernel serves **`code_bits=8` only**. A 16-bit/E8P routed projection falls off
  NAX to the Metal fallback (~40 ms vs ~5 ms per projection) and fails Lane S.
  **No runtime feature in this program may move a routed expert off the 8-bit
  NAX path.** Streaming, prefetch, and promotion must preserve the existing
  `gather_vqmm` → `nax_e8` dispatch byte-for-byte.
- **Memory-clean acceptance:** any benchmark row with non-zero `pageouts` or
  `swapouts` delta is INVALID for acceptance evidence. `collect_metric_snapshot`
  (`src/mlx_vq/benchmark/metrics.py:178`) already tracks these — every new
  benchmark must report them. Streaming *intentionally* pages from disk; its
  benchmarks measure tok/s and cache-hit rate, and must be labeled as streaming
  runs, never compared against resident Lane S rows.
- **Do not set wired limit by default.** Leave `GLM_MLX_WIRED_LIMIT_GB` and
  `GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB` unset. Plan B's RAM auto-sizing sizes the
  *software* expert-cache budget only; it may call `mx.set_wired_limit()` **only**
  behind an explicit, off-by-default opt-in flag.
- **Heavy jobs:** any command that loads a large model or runs eval/benchmark
  work must acquire the advisory `.keep-heavy-job.lock` flock at the repo root.
  `runs/` and the two handoff `.md` files at repo root are protected paths — do
  not write elsewhere into them.
- **Artifacts are gitignored.** `artifacts/` is not in git; evidence lives on
  disk referenced by hash. Tests must never depend on a resident model artifact
  being present — use synthetic fixtures (small fabricated VQ shards) so the
  whole suite runs on a machine with no model downloaded.
- **Naming:** public surfaces say **KEEP** (method) and **RAMP** (runtime) first;
  "VQ" only as the technical term. New modules live under `src/mlx_vq/` (where
  the machinery is); add `src/ramp/` re-exports as the final task of each plan.
- **Tests:** `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest`.
  Files are `tests/test_*.py`, pytest style, `from __future__ import annotations`
  on line 1, `tmp_path` for scratch, `np.testing.assert_allclose(..., rtol, atol)`
  for numerics. `pyproject.toml` sets `pythonpath=["src"]`, `testpaths=["tests"]`.
- **Git hygiene:** frequent commits, one per task. Do NOT push, merge, publish,
  or upload without explicit approval. A codex agent is active on branch
  `keep-glm52-pipeline-and-p1-lock`; coordinate file ownership — these plans
  create new files under `src/mlx_vq/runtime/`, `src/mlx_vq/kernels/parity.py`,
  and `tests/test_*` that the campaign does not touch.

## Shared architecture facts (verified 2026-07-11)

The plans reference these repeatedly; they were confirmed by reading the code.

- **Expert load is fully eager.** `load_quantized_vq_switch_linear(path, prefix)`
  (`src/mlx_vq/io/load.py:235`) calls `mx.load(str(path), format="safetensors")`
  and returns a `QuantizedVQSwitchLinear` (`src/mlx_vq/nn/switch_linear.py:92`)
  holding resident `codes[num_experts, out, in//8]` + `scales[..., in//group]`.
- **Per-layer shard layout.** `load_glm52_vq_switch_glu(artifact_dir, layer)`
  (`src/mlx_vq/models/glm52_vq_adapter.py:581`) reads three shards per sparse
  layer: `layer-{layer:05d}-{gate_proj,up_proj,down_proj}.safetensors`, prefix
  `model.layers.{layer}.mlp.switch_mlp.{proj}`.
- **Bind loop.** `bind_glm52_vq_experts(model, artifact_dir, *, layers, profile,
  strict, artifact_paths)` (`glm52_vq_adapter.py:698`) iterates sparse layers and
  calls `layer.mlp.bind_switch_mlp(switch_mlp)` (a `Glm52VQMoE`, L427).
- **Forward seam.** `Glm52VQMoE.__call__` (L451) computes `inds, scores =
  self.gate(x)` then `y = self.switch_mlp(x, inds)`. The `switch_mlp` is a
  `QuantizedVQSwitchGLU` (`src/mlx_vq/models/glm4_moe_adapter.py:109`).
- **Kernel dispatch.** `QuantizedVQSwitchLinear.__call__` (L310) →
  `gather_vqmm` (`src/mlx_vq/ops/vq_switch.py:526`) →
  `gather_vqmm_sorted_routes` (L252) → `nax_e8` path (L316,
  `nax.nax_e8_fp16_sorted_steel_matmul`). Kernels compiled via
  `mx.fast.metal_kernel`; native ext loaded by `nax.load_native()`
  (`src/mlx_vq/kernels/nax.py:53`).
- **Generation loop.** `_stream_mlx_tokens(runtime, messages)`
  (`src/mlx_vq/build/chat.py:232`) uses `mlx_lm.generate.generate_step(...,
  prompt_cache=runtime.model.make_cache())`. `GLM52VQModel.make_cache()`
  (`glm52_vq_adapter.py:567`) returns `list[CacheList]` (each wrapping one/two
  `mlx_lm.models.cache.KVCache`). GLM45-Air returns `list[KVCache]`.
- **MTP is present in source but excluded at load.** `_is_mtp_tensor(name, args)`
  (`glm52_vq_adapter.py:287`) flags layers `>= num_hidden_layers`; binding skips
  them (L840-845); `NON_VQ_EXPECTED_MTP_TENSORS = 2039`
  (`src/mlx_vq/convert/glm52_non_vq.py:58`); the validator *forbids* layer-78
  artifacts (`src/mlx_vq/validate/glm52_artifact.py:554`). F5 must deliberately
  re-admit the MTP head.
- **Kernels expose numpy references.** Every kernel has a `*_reference_np`
  twin: `vq_qmv`/`vq_qmv_reference_np` (`src/mlx_vq/kernels/vq_qmv.py:163/126`),
  `vq_qmm`/`vq_qmm_reference_np` (`vq_qmm.py:93/69`), and `gather_vqmm_kernel`
  (`gather_vqmm.py:305`). Golden E8 vectors live in `tests/test_e8_reference.py`
  via `mlx_vq.codebook.e8` (`decode_e8p`, `e8p_full_grid`, pinned SHAs).
- **Memory telemetry.** `src/mlx_vq/benchmark/metrics.py`: `current_rss_bytes()`
  (L39), `collect_vm_stat_counts()` (L57), `collect_metric_snapshot(*,
  previous_vm_stat_counts)` (L178), `_mlx_memory_value(name)` (L19, wraps
  `mx.get_active_memory/get_peak_memory/get_cache_memory`).
- **CLI.** `keep = keep.cli:main` → `mlx_vq.build.cli.main`. `build_parser()`
  (`src/mlx_vq/build/cli.py:418`) uses argparse subparsers; each subcommand does
  `sub.set_defaults(func=_cmd_x)` and handlers are `_cmd_x(args) -> int`.

## Program-level risks

- **Streaming latency is real.** Colibri's cold decode is ~0.05–0.1 tok/s; ~1
  tok/s on fast NVMe. On a Mac streaming compressed (~1–2 bpw) experts the I/O is
  far smaller, but a budget below the per-token working set still means a disk
  read per layer per token. Plan A ships behind an off-by-default flag and never
  regresses the resident path.
- **Do not compare streaming runs to Lane S.** Lane S is a resident,
  memory-clean benchmark. Streaming runs page from disk by design; they get their
  own labeled benchmark surface (Plan A, Task: benchmark).
- **MTP re-admission fights an existing invariant.** F5 must relax the layer-78
  forbid-guard *only* on an explicit opt-in path, leaving the default acceptance
  pipeline's guard intact.
