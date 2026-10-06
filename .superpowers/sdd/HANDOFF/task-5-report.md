# Task 5 report — DeepSeek-V4-Flash `mtp-train` teacher targets

Status: **COMPLETE / AUDITED PASS**. The producer reached atomic terminal state,
and all 120 session files passed the independent content and inventory audit.

## Scope and authority

- Baseline: `05af9a89a5050b08920f63bfd9ee1b4abcbdfa08` on local `main`.
- Authority: `docs/HANDOFF.md` and `.superpowers/sdd/HANDOFF/task-5-brief.md`.
- Output: `/Users/jack.mazac/keep-artifacts/dsv4-teacher-mtp-train`.
- Log: `/Users/jack.mazac/keep-artifacts/dsv4-teacher-mtp-train.nohup.log`.
- The 37-session holdout was not selected, validated, emitted, or otherwise
  consumed. No Task 6/7, cloud/external system, push, custom wired limit,
  page-cache mode, compression, or reduced run was used.

## Focused RED → GREEN

The existing producer already owned layer-sequential execution, MTP capture,
atomic session writes, heartbeat, manifest, validation, and resumability. The
smallest shared-path fixes were:

1. Filter rows by requested split/prompt before reading or validating their
   token payload, so an unselected sealed holdout row is not consumed.
2. Authenticate the pack model ID and the pinned checkpoint revision,
   config SHA-256, index SHA-256, profile/config policy, and KEEP completion
   record before constructing the source workload.
3. Include `lm_head_slice`, I/O thread count, F_NOCACHE policy, and compression
   policy in the generation identity; require the stored generation hash,
   split, and token count to agree before any resume skip is earned.
4. Refuse the repository wired-limit environment variables as well as the
   existing legacy variables, including empty-valued presence.

RED was six focused failures: sealed holdout validation, wrong pack model ID,
four missing generation fields/resume binding, and two repository wired-limit
variables. GREEN: `61 passed`, with only the two existing SWIG deprecation
warnings. Ruff check passed with the file's two pre-existing `TRY004` findings
ignored; `git diff --check` passed.

## Authenticated preflight

- Source model: `deepseek-ai/DeepSeek-V4-Flash-0731`.
- Revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.
- Config SHA-256: `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`.
- Index SHA-256: `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`.
- Download-completion SHA-256:
  `68d973a8b79801a90f888c4edc5d7739721aaf2e5dc3698787964d64467cf4eb`.
- Pack SHA-256: `168eb4cb9251d1cb01a79a0d61205480f02252a0e6d2c96f596847b21e7f82a5`.
- Selected prompt-inventory SHA-256:
  `c532bc6a451e9a1c7c8000a4484a7778b704f7f42c7c1ea631554ba4f92405bd`.
- Exact selection: 120 `mtp-train` sessions, 5,552,257 input tokens,
  1,502,378 supervised positions, zero other-split sessions.
- Output was absent; heavy lock was free; all seven repository/legacy wired
  variables were absent; disk free was 990,265,167,872 bytes.
- Pinned configuration: `mode=mtp-targets`, `split=mtp-train`, chunk 1024,
  top-K 2048, LM-head slice 2048, width 5, 8 I/O threads, F_NOCACHE enabled,
  uncompressed output.

## Launch and runtime identity

The controller approved the exact command before launch:

```bash
nohup env -u GLM_MLX_WIRED_LIMIT_GB \
  -u GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB \
  -u GLM_REQUIRED_WIRED_MB \
  -u MLX_METAL_DEBUG \
  -u IOGPUWiredLimitMB \
  -u MLX_WIRED_LIMIT_MB \
  -u MLX_RELAXED_WIRED_LIMIT \
  UV_CACHE_DIR=/tmp/keep-uv-cache \
  uv run --group dev python benchmarks/produce_dsv4_teacher_cache.py run \
  --mode mtp-targets --split mtp-train \
  --out-dir /Users/jack.mazac/keep-artifacts/dsv4-teacher-mtp-train \
  --checkpoint /Users/jack.mazac/models/DeepSeek-V4-Flash-0731 \
  --pack /Users/jack.mazac/models/teich/dsv4-coding-agent-v1-20260811.json \
  --chunk 1024 --top-k 2048 --lm-head-slice 2048 \
  --mtp-draft-width 5 --io-threads 8 \
  > /Users/jack.mazac/keep-artifacts/dsv4-teacher-mtp-train.nohup.log 2>&1 &
```

- Retained exec session: `16027`.
- Retained interactive zsh PID: `50595`.
- Disowned uv wrapper PID: `50765`.
- Python producer child PID: `50768`.
- Lock/status PID: `50768`; lock label: `dsv4-teacher-mtp-targets`.
- Direct child-environment inspection found all seven named wired variables
  absent.
- Launched CLI SHA-256:
  `0e9c536163ace0540587f564ae2e2f1dc31f5b66c47d9f4d3caaf5b59fe93ca1`.
- Launched teacher-runner SHA-256:
  `c552cb8687f4e33258d639e752d9678fce18b42f26bdbcde1d87d45b7745cad6`.
- Run-manifest SHA-256:
  `129edd5266ced76c4a00f3ea797674f5c0a1f56ad90987338e79582ba8073d45`.
- Generation-config SHA-256:
  `6cac8ee4e95efe949d9a7d3b0843e002ef0886d7e52be1d66fad210c9a195997`.
- Resident bind: 191.74 seconds, 1,271 resident tensors, 390 FP8 decoded
  tensors, 54.4 GB resident parameter accounting, 93.79 GB initial MLX peak.

The preflight estimate was approximately 18 hours, obtained by scaling the
prior backbone run and adding the runner's documented rough MTP overhead. The
first measured ETA was 201,253 seconds (55.9 hours), so the estimate was wrong
and was immediately superseded by measured progress rather than retained as a
claim.

## First real session proof

The producer continued to session 2 while the first atomic file was loaded and
audited independently:

- Prompt: `teich_claude_agent-a3622521df8a9137d`.
- 3,428 input tokens; 1,087 supervised positions.
- 124.3 seconds; 27.6 tok/s.
- File bytes: 111,383,385.
- File SHA-256:
  `6a8968830cfe695c108747747ea39e3d057340ffbbbe6669200c375251587b0b`.
- Array-contract SHA-256:
  `8c086cfe13e06ebb8ad781ef88adb65b9fc6b383be8aaaa945e840152fedcede`.
- Full schema/dtypes/shapes passed. Every captured logit was finite and sorted
  within the stored fp16 tolerance. Every one of the 2,048 IDs was distinct in
  every position/slot and in vocabulary range. Targets and validity were
  independently rederived from the selected pack. Positions, next-token
  targets, token hash, split, chunk, generation/source identity, and width all
  agreed. Final hidden states were finite and nonconstant (`std=0.25078565`).
  The atomic temp path was absent.

## Progress and terminal audit

| sessions | input tokens | supervised | cumulative tok/s | last tok/s | ETA h | payload bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 10/120 | 68,674 | 29,117 | 40.2 | 51.0 | 37.88 | 2,983,476,531 |
| 20/120 | 275,067 | 83,858 | 55.2 | 84.3 | 26.57 | 8,592,465,614 |
| 30/120 | 572,132 | 142,612 | 60.3 | 60.0 | 22.93 | 14,612,639,196 |
| 40/120 | 967,902 | 250,721 | 60.5 | 63.0 | 21.05 | 25,689,874,143 |
| 50/120 | 1,402,696 | 358,892 | 59.7 | 59.0 | 19.32 | 36,773,461,636 |
| 60/120 | 1,869,096 | 491,067 | 58.6 | 63.5 | 17.47 | 50,316,571,533 |
| 70/120 | 2,381,922 | 638,437 | 55.2 | 41.4 | 15.95 | 65,416,606,795 |
| 80/120 | 2,948,347 | 805,806 | 48.5 | 29.1 | 14.92 | 82,565,799,710 |
| 90/120 | 3,562,317 | 930,828 | 47.6 | 52.9 | 12.56 | 95,375,991,288 |
| 100/120 | 4,191,650 | 1,093,000 | 47.9 | 54.4 | 8.11 | 111,992,683,388 |
| 110/120 | 4,846,614 | 1,237,642 | 48.3 | 52.0 | 4.09 | 126,813,199,062 |
| 120/120 | 5,552,257 | 1,502,378 | 44.9 | 29.7 | 0.00 | 153,938,907,002 |

At 10 sessions, observed payload density was 102,465.1 bytes per supervised
position, projecting to 153,941,323,065 bytes across the exact split. The
status log had exactly ten completion rows, no temp files existed, and the
lock remained held by PID 50768. Stream counters were 430 layer reads,
1,471.697 GB read, 8.56 GB/s device throughput, and 6.13 seconds blocked;
reported peaks were 93.79 GB MLX and 81.02 GB host RSS.

At 20 sessions, observed payload density was 102,464.5 bytes per supervised
position, projecting to 153,940,367,100 bytes across the exact split. The
status log had exactly 20 completion rows, no temp files existed, the launched
source SHA-256 values were unchanged, and the lock remained held by PID 50768.

At 30 sessions, observed payload density was 102,464.3 bytes per supervised
position, projecting to 153,940,114,787 bytes across the exact split. The
status log had exactly 30 completion rows, no temp files existed, the launched
source SHA-256 values were unchanged, and the lock remained held by PID 50768.
The earlier 189-second sparse-heartbeat alert during session 23's MTP-draft
phase resolved without action: the process remained CPU-active and advanced
atomically to session 24 with a healthy lock and no temp output.

At 40 sessions, observed payload density was 102,464.0 bytes per supervised
position, projecting to 153,939,645,005 bytes across the exact split. The
status log had exactly 40 completion rows, no temp files existed, the launched
source SHA-256 values were unchanged, and the lock remained held by PID 50768.

At 50 sessions, observed payload density was 102,463.9 bytes per supervised
position, projecting to 153,939,457,402 bytes across the exact split. The
status log had exactly 50 completion rows, no temp files existed, the launched
source SHA-256 values were unchanged, and the lock remained held by PID 50768.

At 60 sessions, observed payload density was 102,463.8 bytes per supervised
position, projecting to 153,939,299,742 bytes across the exact split. The
status log had exactly 60 completion rows, no temp files existed, the launched
source SHA-256 values were unchanged, and the lock remained held by PID 50768.

At 70 sessions, observed payload density was 102,463.7 bytes per supervised
position, projecting to 153,939,184,107 bytes across the exact split. The
status log had exactly 70 completion rows, no temp files existed, the launched
source SHA-256 values were unchanged, and the lock remained held by PID 50768.

At 80 sessions, observed payload density was 102,463.6 bytes per supervised
position, projecting to 153,939,088,362 bytes across the exact split. The
status log had exactly 80 completion rows, no temp files existed, the launched
source SHA-256 values were unchanged, and the lock remained held by PID 50768.
The 900-second monitor alert for session 71 was classified as active compute:
the runnable producer was using CPU, its lock was held, no temp file existed,
and its 22,937-position capture later completed atomically without action. The
monitor-only MTP-draft threshold was raised to 1,800 seconds for the subsequent
large captures; the producer command and configuration were not changed.

The user then paused the run after 80 atomic sessions. The controller sent
SIGTERM only to Python PID 50768; uv PID 50765 exited, the lock became free,
and there were exactly 80 completion rows, 80 atomic files, and no temp output.
The interrupted session 81 had no atomic file. On explicit continuation, the
identical command resumed from retained zsh PID 50595 as uv PID 89184 and
Python/lock/status PID 89186. All seven wired variables remained absent; 80
identity-bound files were skipped; and session 81 restarted from scratch.

At 90 sessions, the global totals (including the 80 pre-pause sessions) were
3,562,317 input tokens and 930,828 supervised positions. Observed payload
density was 102,463.6 bytes per supervised position, projecting to
153,939,063,972 bytes across the exact split. The status log had exactly 90
completion rows, no temp files existed, source SHA-256 values were unchanged,
and the lock remained held by PID 89186. Because resumed runtime counters reset
after the 80 skips, the raw status ETA incorrectly included already-skipped
work; the 12.56-hour table value is the remaining 1,989,940 tokens divided by
the measured resumed 44.0 tok/s and remains only a throughput estimate.

At 100 sessions, global totals were 4,191,650 input tokens and 1,093,000
supervised positions. Payload density was 102,463.6 bytes per supervised
position, projecting to 153,939,015,264 bytes across the split. The status log
had exactly 100 completion rows, no temp files existed, source SHA-256 values
were unchanged, and the lock remained held by PID 89186. The 8.11-hour table
value is the remaining 1,360,607 tokens divided by resumed throughput of
46.6 tok/s; the raw reset-counter status ETA remained unsuitable after skips.

At 110 sessions, global totals were 4,846,614 input tokens and 1,237,642
supervised positions. Payload density was 102,463.6 bytes per supervised
position, projecting to 153,938,990,742 bytes across the split. The status log
had exactly 110 completion rows, no temp files existed, source SHA-256 values
were unchanged, and the lock remained held by PID 89186. Effective global
throughput was 48.26 tok/s; the 4.09-hour table value is the remaining 705,643
tokens divided by resumed throughput of 47.9 tok/s. The raw reset-counter
status ETA remained unsuitable after skips.

At 120 sessions, the global totals exactly matched the authenticated split:
5,552,257 input tokens and 1,502,378 supervised positions. Captured session
compute was 60,832.0 seconds before the pause and 62,867.7 seconds after
resume, or 123,699.7 seconds total and 44.88 input tok/s overall. The final
session ran at 29.7 tok/s. The producer wrote terminal phase `complete` with
40 produced and 80 revalidated skips in the resumed process. Python PID 89186
and uv PID 89184 exited. A nonblocking exclusive `flock` then proved the heavy
lock free; the lock file intentionally retained only its last diagnostic PID
and label.

## Terminal inventory and content audit

The independent audit acquired the same heavy-job lock, loaded the exact
`mtp-train` selection through the pack validator, and checked every file one at
a time. Result: **120/120 PASS**, with zero missing, extra, duplicate,
temporary/partial, symlink, or other-split outputs. The append-only status log
contained exactly 120 uniquely ordered session-complete rows plus one terminal
row, and `status.json` exactly matched that terminal row.

For every uncompressed regular NPZ, the audit required the exact 20-array key,
dtype, and shape contract; exact prompt/split/token/chunk/generation identity;
pack-identical positions and next-token targets; independently rederived
width-5 target tokens and validity; finite descending top-2,048 values using
the writer's fp16 tolerance; all 2,048 IDs distinct in every position and slot
and in the 129,280-token vocabulary; finite log-normalizers no lower than the
top logit; finite tail mass in `[0, 1)`; and finite nonconstant 4,096-wide final
hidden states. Every archive member was stored rather than compressed. Every
session file received a streaming SHA-256 and array-contract SHA-256.

Actual storage was:

- Session payload logical bytes: **153,938,907,002**.
- Session payload allocated bytes: **153,939,152,896**.
- Manifest/status metadata logical bytes: **173,027**.
- Complete output-tree logical bytes: **153,939,080,029**.
- Complete output-tree allocated bytes: **153,939,398,656**.

The logical payload was 61,092,998 bytes (0.0397%) below the brief's approximate
154,000,000,000-byte estimate. The 120-row file set is bound by canonical
contract SHA-256
`c60bf865b639d37c78d56b637b3398834b7ee94793cfe36ba2ce812dcc379244`.
The audit contract SHA-256 is
`a6f9247a5ff6a799793df292b1ab49a01a25947bbd0c23905893168f1c729203`.

Compact evidence is
`artifacts/quality/dsv4-task5-20260819/terminal-audit.json`, file SHA-256
`d2ac01f8fa02de942d38d108665ac7457d09b564295a643fc065100595bc801d`,
with its adjacent `terminal-audit.sha256` verifier. It contains all 120 file
hashes and per-file contract summaries. Terminal runtime hashes are:

- Resumed run manifest:
  `f3819b6ba69832a5aef1893bb82ebe1537f7eb420603c3a1a4ed8994b62a1bf5`.
- Run summary and terminal nohup log:
  `f49063e8328c87de29c75bbc725d78747542e4eb71e748976702dc09ac17c2c6`.
- Terminal status:
  `3f0ef9d77a190153894cd47b358885a32defa7f96d4f63cbfae9364165800d50`.
- Append-only status log:
  `2f88b72587ef17d1f520843828b3a9bc00682befb4c4a1601ffdf0b2b5f8163a`.

The manifest is intentionally rewritten by the existing resume rail after
source binding; its terminal file hash therefore differs from the first-launch
hash recorded above because `resident_bind_seconds` was remeasured. The pinned
generation, pack, checkpoint authority, selected-session inventory, and all
per-file identities remained identical.

## Final verification

- Focused tests: `61 passed`, with only the two existing SWIG deprecation
  warnings (`PYTHONPATH=. UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev
  pytest -q tests/test_dsv4_teacher_runner.py`).
- A first fresh invocation without `PYTHONPATH=.` failed during collection
  because `benchmarks/` is not an installed package; the repository-root import
  path above is the established focused command, and no code was changed for
  that invocation-only issue.
- Ruff: all three owned code/test paths passed with the file's two pre-existing
  `TRY004` findings ignored.
- `git diff --check`: passed.
- The CLI and teacher-runner source hashes still exactly matched the launched
  bytes after the terminal audit.
- The evidence sidecar verified `terminal-audit.json: OK`.

## Limitations and self-review

No downstream MTP-head recovery training is in Task 5 scope, so these targets
prove capture integrity, not recovery quality. The exact approved resume
command used `>` and therefore replaced the first process's nohup log; the
append-only status log, report checkpoints, and preserved terminal summary
carry the lifecycle evidence across both processes. Resumed heartbeat counters
reset after 80 authenticated skips, so the runtime's raw ETA remained invalid;
all progress and terminal totals in this report explicitly combine the captured
pre-pause and resumed counters. The task did not read or emit holdout data,
start Task 6, mutate cloud/external systems, raise a wired-memory limit,
compress outputs, overwrite a session file, or push.
