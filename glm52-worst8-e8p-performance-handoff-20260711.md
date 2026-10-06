# GLM-5.2 worst8 E8P performance handoff — 2026-07-11

## Verdict

**STOPPED BY USER: the current worst8 E8P materialization path is too slow to
continue. Do not relaunch the same command unchanged.**

The run was healthy but made no publishable group-level progress after
3h 15m 09.58s. Because the implementation processes 24 groups serially and did
not finish the first group, the observed upper bound is less than 1/24 of the
job (<4.17%). The exact in-group percentage is not instrumented. A straight-line
projection is therefore at least 78 hours, before audit and reevaluation.

## Clean stop state

- Producer PID `48401` received `SIGTERM` at `2026-07-11T21:39:40Z`.
- Supervisor PID `48383` observed the producer exit and then exited normally.
- Terminal classification: `finished`, exit code `-15`,
  `physical_complete=false`, `manifest_sha256=null`.
- `keep recovery campaign status --json` reports `active_processes=[]`, no
  contradictions, and `.keep-heavy-job.lock` free.
- The scheduled polling process was also stopped.
- Expected output
  `artifacts/quality/glm52-recovery-worst8-e8p-artifact-20260711` is absent.
  There is no partial group to resume.
- Both producer stdout and stderr logs are zero bytes.

## Authenticated run identity

| Field | Value |
|---|---|
| Campaign | `glm52-recovery-v1-20260711` |
| Experiment | `worst8-e8p` |
| Transition | `worst8-e8p-rematerialize` |
| Started | `2026-07-11T18:24:30.739322Z` |
| Finished | `2026-07-11T21:39:40.315326Z` |
| Command SHA-256 | `4722fe30777676ee0f5792bfde7045ccaa3809c80ba80b8a3eca27df42c449c7` |
| Launch token | `ce1ee633c06f987a59177a289ea3fad590b776a752b28bd1df95a0b9ca9bf5a8` |
| Terminal event SHA-256 | `cd00445a9030fd1048dd9ac9e05a6b7450c41c82840b7776edb4d75e95442775` |
| Terminal file SHA-256 | `5e3120e6899056af3b96233e751c9792c0cc02c08c3d6f570b652fbd568737b3` |
| Ledger file SHA-256 after stop | `f720ddccfc213748207c59d066c73b11e458e6d3154ee715fe3df8a230e7759f` |
| Campaign config semantic identity | `ff98d4b78a425963f067f78f1c7721dcf7799de2856b40978d19aec1f2a8a794` |

Evidence paths:

- `artifacts/quality/glm52-recovery-v1-20260711-ledger.jsonl`
- `artifacts/quality/glm52-recovery-v1-20260711-launches/2026-07-11T18-24-30.739322Z-worst8-e8p-rematerialize-4722fe307776.terminal.json`
- `artifacts/quality/glm52-recovery-v1-20260711-launches/2026-07-11T18-24-30.739322Z-worst8-e8p-rematerialize-4722fe307776.stdout.log`
- `artifacts/quality/glm52-recovery-v1-20260711-launches/2026-07-11T18-24-30.739322Z-worst8-e8p-rematerialize-4722fe307776.stderr.log`

The ledger is append-only and now ends at sequence 11 with the authenticated
`transition_finished` event. Preserve it; a future launch must append a new
launch identity rather than rewriting this outcome.

## Measured performance

At the last pre-stop sample (3h 03m elapsed), the producer was healthy at about
99% CPU with 9,891,184 KiB RSS and an advancing CPU clock. No stderr, crash,
lock conflict, or campaign contradiction was observed. This was compute-bound,
not stalled.

The seed contract for every routed group has 168 experts and approximately
48 MiB of decoded weights. The E8P experiment selects eight layers and all
three routed projections, for 24 serial groups.

## Root cause in the current implementation

The hot path is a brute-force, CPU/NumPy E8P nearest-code search:

1. `quantize_weight_importance_aware` builds the full 65,536-entry E8P grid for
   16-bit codes (`src/mlx_vq/convert/glm52_recovery_materialize.py:671`).
2. For every group and codeword, it calls the diagonal-Hessian nearest-code
   routine (`glm52_recovery_materialize.py:680-704`).
3. It runs three alternating scale/code iterations and then a fourth final code
   assignment (`glm52_recovery_materialize.py:706-726`).
4. `nearest_codebook_indices_diagonal_hessian` scans the whole codebook in
   8,192-entry chunks and materializes distance matrices on CPU
   (`src/mlx_vq/quant/rtn.py:114-135`).
5. The outer materializer loads and quantizes all 168 experts for one projection
   before transactionally publishing that group, then advances serially to the
   next group (`glm52_recovery_materialize.py:1783-1862`).

Consequences:

- The 16-bit path expands code search from 256 to 65,536 candidates.
- Four exhaustive assignments multiply that work again.
- There is no expert-level checkpoint, progress counter, or durable resume
  boundary before a complete layer/projection group is published.
- Current monitoring can only report `0/24` until the first multi-hour group
  finishes, so it cannot provide a truthful percentage or ETA.

## Required next wave before any rerun

Treat performance as an implementation task, not an operations wait:

1. Add a deterministic microbenchmark for one representative expert and all
   three projection shapes. Record wall time, CPU time, peak RSS, and byte-exact
   output identity.
2. Add progress/resume at expert granularity using authenticated, transactional
   checkpoints. A stopped run must retain completed experts without publishing
   a false complete group.
3. Replace or accelerate exhaustive E8P search. Evaluate, in this order:
   - the mathematical E8P lattice decoder already implied by the codebook;
   - batched/vectorized expert search with bounded memory;
   - Metal/MLX acceleration on the host, while leaving custom wired-memory
     limits unset;
   - bounded process parallelism only after memory and determinism are proven.
4. Preserve exact diagonal-Hessian semantics, deterministic tie-breaking,
   uint16 code identity, source authority, and transactional publication.
5. Prove byte identity against the current exhaustive implementation on small
   fixtures and representative real expert slices before accepting an
   optimization.
6. Establish an acceptance gate before the 24-group rerun: projected end-to-end
   materialization should be measured in hours, not days, and monitoring must
   expose completed experts/groups plus a data-derived ETA.
7. Only after that gate passes, append a new campaign transition and rerun the
   worst8 artifact. Then perform first-class artifact audit, the frozen 66-row
   reevaluation, selection-only re-attribution, and the decision packet.

## Existing completed campaign state to preserve

- Branch: `keep-glm52-pipeline-and-p1-lock`
- HEAD: `3058814a103077cb33ac74465f81d6523d68a4f1`
- Full75 raw manifest SHA-256:
  `544163a74793f78f739649a5a23328ea0d161ed4d667309f99623987b01ba082`
- Full75 audit SHA-256:
  `c65f98ed2328cbd765d5cee46c6709ae72fa6d716186335f336ea24c0ae9e638`
- Full75 reevaluation SHA-256:
  `6f959103caa9b8bcfa707bc1f1cfa75946dda01d4a02dc75845afbf653aebed1`
- Recovered attribution SHA-256:
  `f4db48e00e32b6b1df502bfa7e91902b3a9c008160dd45285d49a66e97d67ee8`
- Full75 decision SHA-256:
  `e35585956a75a1c7df7ed5d5cdc65547dabe813fc29ca51585a50ce6a9aa52d3`
- Recovered worst-eight layer order: `[75, 74, 76, 72, 73, 77, 71, 70]`.
- Latest named verification profile before the stopped heavy run: 434 passed,
  1 deselected, `commit_ready=true`; report SHA-256
  `6dc00319a3546ffa6f2094e178606427c5f3f558deb2e9eb2c718e722b409ed4`.

The Full75 candidate remains diagnostic-only and fails the frozen quality gate.
The stopped worst8 transition produced no candidate artifact and changes no
release verdict.

## Hard guards

- No push, merge, publish, Hugging Face upload, rented hardware, paid service,
  or public release action without explicit approval.
- Leave custom MLX wired-memory limits unset.
- Only selection data may guide optimization; holdout and report remain
  evaluation-only.
- Heavy model operations must hold `.keep-heavy-job.lock`; never overlap them.
- Preserve `runs/`, existing root handoffs, unrelated RAMP plans, and all
  unrelated worktree changes.
- Never describe diagnostic, synthetic, or incomplete evidence as release-ready.
