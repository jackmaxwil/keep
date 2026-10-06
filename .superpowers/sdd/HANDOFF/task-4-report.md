# Task 4 — DeepSeek-V4-Flash MTP verify runtime and authenticated headline

Date: 2026-08-19 (America/Los_Angeles)

Baseline: `27e067a9aa8fc53d9546e51a4b018b0e1b421c79` on local `main`

Initial Task 4 commit: `aec8d8e883081ac6644c73a75581b549c7cee974`

Scope: Task 4 only. Task 2 remains a reviewed negative speed gate/STOP. The
37-session holdout stayed sealed, Task 5 was not started, and no cloud or
external system was used.

## Result

PASS for the Task 4 decision rule on the final authenticated series, with
material timing volatility and all dirty rows retained.

The accepted same-machine series contains five fresh-process pairs. Three
pairs passed both the pre-load quiet gate and timed pageout/swapout gate; two
pairs were dirty and excluded from the decision. Both execution orders occur
among the clean pairs. Their compressed-AR/speculative ratios are
`1.7700938193895241x`, `3.2972944650096374x`, and
`1.4479585874602416x`; the median is `1.7700938193895241x`, above the
`> 1.0` threshold.

This is a narrow same-machine result for one deterministic report-split
prefix, not a stable absolute-speed or portable prompt-distribution claim.
Clean baseline times span `5.583616999996593` to `26.2373436249909` seconds,
and clean ratios span `1.4479585874602416x` to
`3.2972944650096374x`. An earlier authenticated clean pair lost at
`0.709039442x`; a later dirty pair also lost at `0.374804004x`. Those rows
remain on disk and are reported below.

The original `artifacts/benchmarks/dsv4-task4-20260819` compact evidence is
superseded. Its rows predate the content receipt and exact benchmark/runtime
source-byte hashes, so it is not used for this decision and was not relabelled
as authenticated evidence.

## Minimum reused implementation seam

No parallel speculative framework or generic benchmark framework was added.
The runtime reuses:

- `DeepseekV4FlashVQModel.dspark_forward`, `dspark_markov`,
  `dspark_append_context`, and `make_mtp_cache` for drafting;
- `PoolingCache` for compressed-attention rollback;
- MLX-LM rotating-cache behavior through the DSV4-specific undo/replay seam;
- the existing production non-VQ and VQ artifact binders;
- `gather_vqmm` production dispatch, with a context-local trace around target
  verification;
- existing `heavy_job_lock`, metric, VM-stat, MLX-memory, teacher-pack, and
  quiet-window rails.

Task 4 implementation paths are:

- `src/ramp/models/deepseek_v4_flash_adapter.py`
- `src/mlx_vq/ops/vq_switch.py`
- `src/mlx_vq/models/dsv4_composite_loader.py`
- `src/mlx_vq/quality/dsv4_mtp_runtime.py`
- `benchmarks/bench_dsv4_mtp_headline.py`
- `tests/test_deepseek_v4_flash_adapter.py`
- `tests/test_dsv4_composite_loader.py`
- `tests/test_bench_dsv4_mtp_headline.py`

The first review remediation changed only the composite loader, benchmark,
corresponding focused tests, compact evidence, and this report. The second
review remediation changes only the benchmark validator, its focused test,
and this report; no evidence packet or runtime source is rewritten.

## RED → GREEN evidence

The initial Task 4 implementation began with missing runtime, composite
loader, and benchmark imports. The focused implementation suite subsequently
passed, and the existing logits test was explicitly rebaselined to the
observed contract: eligible parameters remain BF16 while prompt and decode
logits are FP32 because the existing FP32 hyper-connection accumulators feed
the shared head. Production arithmetic was not cast to satisfy the old test.

The review remediation began from two focused RED groups:

```text
payload authentication regressions:
3 failed, 2 warnings in 0.98s

evidence validation regressions:
6 failed, 3 passed, 2 warnings in 0.87s
```

The payload REDs covered same-size byte mutation, an unexpected runtime
safetensors file, and an unexpected tensor. The evidence REDs covered missing
identity, negative/inconsistent counters, fake NAX declarations with no real
dispatch records, and fail-open compact validation.

Focused GREEN after the minimum root-cause changes:

```text
UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONPATH=.:src \
  uv run --group dev python -m pytest \
  tests/test_dsv4_composite_loader.py \
  tests/test_bench_dsv4_mtp_headline.py -q

16 passed, 2 warnings in 0.95s
```

The first broad run found one owned test expecting `ValueError` where the
stricter mapping contract correctly raises `TypeError`, plus a non-owned
adapter token-parity flake. The owned expectation was corrected. The adapter
test passed in isolation and in prefix reruns. Broad GREEN before the real
series was:

```text
UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONPATH=.:src \
  uv run --group dev python -m pytest \
  tests/test_deepseek_v4_flash_adapter.py \
  tests/test_dsv4_composite_loader.py \
  tests/test_gather_vqmm_verify_rows.py \
  tests/test_bench_dsv4_mtp_headline.py -q

321 passed, 2 warnings in 206.56s
```

The two warnings are the existing SWIG deprecation warnings. Fresh final
focused, broad, static, and evidence-reconstruction results are recorded in
the verification section below.

The second review remediation used three independent RED groups:

```text
bind-proof anchoring:
3 failed, 9 deselected, 2 warnings in 3.44s

strict verdict fields:
10 failed, 12 deselected, 2 warnings in 14.33s

deterministic prompt rederivation:
1 failed, 2 warnings in 7.17s

exact prompt-policy field types:
2 failed, 4 passed, 2 warnings in 1.64s
```

The anchoring REDs showed that raw and compact validation did not accept an
external bind-proof authority. The verdict REDs showed that Python equality
allowed `False` and `0.0` to act like clean zero deltas, while malformed quiet
records were merely excluded. The prompt RED replaced the real prefix and
recomputed every dependent digest without rejection.

The minimum changes require a bind-proof path plus externally supplied file
SHA-256, revalidate the proof's current payload receipt using the existing
loader rail, compare row artifact and historical harness mappings to that
proof, validate exact counter types before classification, and rederive the
64-token deterministic report prefix from the authenticated current pack.
Focused GREEN after all three changes was `39 passed, 2 warnings in 10.58s`.

The third review remediation separated row schema validation from cleanliness
and removed pair-role short-circuiting. The first focused RED was `8 failed,
2 passed, 32 deselected, 2 warnings in 2.84s`: three malformed speculative
rows escaped validation behind a dirty baseline, while five synthetic
field-mutation cases raised instead of remaining dirty. The targeted GREEN
was `10 passed, 32 deselected, 2 warnings in 5.15s`. A second RED established
an intermediate zero-attempt interpretation; its focused GREEN was `1 passed,
42 deselected, 2 warnings in 0.95s`. The later producer-contract audit below
supersedes that interpretation because the producer never emits zero attempts
in an enabled record.

The minimum change validates and caches cleanliness for every row during the
row-validation pass. Exact field types, negative counters, and invalid windows
still fail closed. Well-formed failed producer states and nonzero preflight or
timed pageout/swap counters return false, so the row stays visible as dirty and
the pair cannot contribute to the verdict.

The fourth/fifth review remediation then matched that cleanliness validator to
the exact `wait_for_memory_quiet` producer states. Direct producer-output and
malformed-neighbor REDs were `5 failed, 6 passed, 42 deselected, 2 warnings in
1.45s`; the expanded impossible-state slice was `3 failed, 4 passed, 47
deselected, 2 warnings in 0.71s`. Targeted GREEN was `12 passed, 37 deselected,
2 warnings in 0.68s`.

The validator now accepts only the producer's three state-dependent schemas:
the exact disabled record, the complete unavailable record with `None` VM
deltas, or the complete active record with exact nonnegative integer deltas.
Disabled and unavailable records are dirty. For active records, the quiet flag
must equal the zero-delta result; a consistent false/nonzero record is dirty.
Missing, extra, mistyped, negative, zero-attempt enabled, or internally
impossible records fail closed. Timed row counters are validated before any
inactive-state return, preserving independent validation of every row.

## Payload authentication and real bind proof

The composite loader now hashes every payload that can be bound and checks
the exact resident and routed safetensors/tensor inventories. It authenticates
the resident `file_sha256` and every routed file `sha256`, rejects symlinks,
extra files, extra or missing tensors, size/hash drift, and index/manifest
mapping drift. The receipt records each file's SHA-256, tensor inventory
digest, and current `{device, inode, size, mtime_ns, ctime_ns}` identity.

Each timing child receives the bind-proof path and literal receipt SHA-256.
Before binding, it validates the canonical receipt, exact file/header sets,
declarations, and current file identities. A same-size mutation or replacement
therefore fails closed before model load. The artifact identity is derived
from pinned declarations plus the authenticated content-receipt digest.

Pinned identities:

- source revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- source config SHA-256:
  `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- source index SHA-256:
  `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- resident manifest SHA-256:
  `35fa6361691326aa805a5ff59a95508af1740660b98844071da59856ae8009d5`
- resident package-set SHA-256:
  `75b63dee580fc4038be088db63ef9e5dd2cb0256bb722abb77d94651ee51abe9`
- routed manifest SHA-256:
  `a996c4bedc514f49c54fe15d5d9b2373f839678e55b241861f2fda60cb4bf8b3`
- routed inventory SHA-256:
  `4bda0cf1a635123d08f711cd38ad4bb3f716b41714883495ec91cfd71540e843`

Exact bind command, launched by retained zsh PID `93082`:

```bash
nohup env -u GLM_MLX_WIRED_LIMIT_GB \
  -u GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB \
  UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONPATH=.:src \
  uv run python -m benchmarks.bench_dsv4_mtp_headline bind-proof \
  --source /Users/jack.mazac/models/DeepSeek-V4-Flash-0731 \
  --resident-dir /Users/jack.mazac/keep-artifacts/dsv4-residents-release-precision-20260819 \
  --vq-dir /Users/jack.mazac/keep-artifacts/dsv4-vq-e8p-g512 \
  --output /Users/jack.mazac/Developer/keep/artifacts/benchmarks/dsv4-task4-fix1-20260819/bind-proof.json \
  > /tmp/keep-task4-fix1-bind-proof.log 2>&1 &
task4_bind_wrapper_pid=$!
disown
```

Wrapper PID `95973` and Python/lock-owner PID `95975` completed in
`51.12925029100734` seconds. Both wired-limit variables were absent. The
flock was released at terminal; the reusable lock inode remains.

Bind results:

| Check | Result |
|---|---:|
| Authenticated files | 149 |
| Resident / routed files | 11 / 138 |
| Exact resident / routed tensors | 1,661 / 414 |
| Resident parameters bound exactly once | 1,271 |
| Backbone / MTP residents | 1,199 / 72 |
| Backbone layers / MTP stages | 0–42 / 0–2 |
| Runtime tensors / storage bytes | 1,685 / 90,990,750,712 |
| Dense routed parameters | 0 |
| Unbound backbone / MTP routed modules | 0 / 0 |

Receipt and bind identities:

- content receipt SHA-256:
  `b9c0b1afc40356649c4b5761f815acdd1bc49c65a0f8c04fdcf22b16c8931450`
- composite identity SHA-256:
  `8d30e9f61d81192c04684645a7dc4439fcb38e2d233a81e4fe371806f2daa15d`
- harness identity SHA-256:
  `01eda624a465d1d6904ae8af9e8f8211015be4591807376791cfe74b002ecdac`
- bind-proof file SHA-256:
  `647501aa70a4531c3c0cedabc866025f7c2072c5c4bf671e70c83c8605df6115`

The load/bind process recorded pageouts `+443` and swapouts `+90444`; this is
artifact materialization evidence only and is excluded from timing.

## Headline evidence validation

The summarizer no longer trusts row claims. For every row it recomputes or
demands:

- the exact content-bound artifact mapping and its canonical digest;
- the bind-proof-recorded benchmark and runtime generator hashes and their
  mapping digest;
- the current pack SHA-256;
- prompt token IDs/count/digest and output token IDs/count/digest;
- generation configuration mapping/digest;
- `campaign_split: report`, `holdout_used: false`, and wired-limit absence;
- nonnegative integer counters and exact draft, acceptance, rejection,
  verify-pass, emission, rate, and per-pass conservation;
- implementation names/counts, total calls, and NAX presence derived from
  actual raw dispatch records.

Raw and compact validation now require the committed bind-proof path and its
externally supplied file SHA-256. Validation checks the proof bytes, canonical
receipt, current file/header identities, composite identity, and exact
artifact/harness mappings before reading a verdict. The committed compact
representation carries counted unique dispatches; its standalone validator
recomputes totals and implementation counts and then reconstructs the full
summary contract. A declared NAX name with an empty or Metal-only dispatch
trace fails closed.

The accepted rows were generated by exact commit
`17fda71f4aefdd63fc9ef18212c1fb88c3b52cf0`. The bind proof records these
historical generator source-byte hashes, which remain authoritative even when
later validator-only edits change the current benchmark file:

- benchmark:
  `24c9fce7ba4c5c86a2f3f74cc4003bfe4e07fab2827fbc8ffc911d38f4f429fe`
- runtime:
  `1461d7b1f023a58a8d4aa4faf89f1b16f53e69886b66f26186b8058c5f51fb9c`

Other evidence identities:

- teacher pack SHA-256:
  `168eb4cb9251d1cb01a79a0d61205480f02252a0e6d2c96f596847b21e7f82a5`
- prompt ID: `teich_claude_agent-ab6dc2ab37c3c2c34:prefix-64`
- prompt-token SHA-256:
  `56abd36a1c85c08a75204573853754372a2af9de37520117b95f95e56c361030`
- generation-config SHA-256:
  `3c6b81edd82e70d3f1b3555407a09ebffcd9a290168741b8ef17511b2a351baa`
- output-token SHA-256, identical for AR and speculation:
  `fb84360de9a343c8da8ac2b2bf7b3b62651a5d1ce0eaee83f274cc362d8ee81c`

Each speculative row conserves `19 = 11 + 8` drafted/accepted/rejected
tokens, uses four verify passes, and emits `16 = 1 + 11 + 4` tokens.
Acceptance is `0.5789473684210527`, accepted/pass is `2.75`, and emitted/pass
is `4.0`. The actual 516-record trace derives `metal: 504` and
`nax_e8p_m32n64: 12`, across eight unique dispatch shapes. No other
implementation name occurs.

## Preserved run families

All commands used the same receipt, source/pack/prompt identities, 64 prompt
tokens, 16 emitted tokens, fresh child processes, both execution orders, and
the same `> 1.0` median rule. Every dirty row remains in its original output
directory and is excluded rather than substituted.

### `headline` — incomplete

- command shape: `series --output-dir .../headline --pairs 3
  --min-clean-pairs 3 --quiet-window-seconds 5 --quiet-max-attempts 3`
- wrapper/series PIDs: `97111` / `97113`
- pair 0 clean: baseline `5.337517834`, speculative `7.527815125`, ratio
  `0.709039442` (losing)
- pair 1 dirty: speculative `9.341392417` with pageouts `+297`; baseline
  `38.693361042` with pageouts `+72`; raw ratio `4.142140627`
- pair 2 dirty: baseline `27.432755584`; speculative `14.088725708` with
  swapouts `+4012`; raw ratio `1.947142428`
- terminal status: one clean pair and only one clean order; summary/compact
  paths were intentionally absent

### `headline-quiet30-r2` — incomplete

- command shape: `series --output-dir .../headline-quiet30-r2 --pairs 3
  --min-clean-pairs 3 --quiet-window-seconds 30 --quiet-max-attempts 3`
- wrapper/series PIDs: `17300` / `17308`
- pair 0 clean: baseline `42.166256666`, speculative `9.123434125`, ratio
  `4.621752740058964`
- pair 1 clean: speculative `4.449169083`, baseline `37.926337750`, ratio
  `8.524364222270481`
- pair 2 dirty: baseline `5.637076750`; speculative `15.040065459` with
  pageouts `+69`; raw ratio `0.374804004`
- terminal status: `INCOMPLETE`, two clean and one dirty; clean median
  `6.573058481164723` is not a headline
- summary SHA-256:
  `4a1c0872bbed830450999c6c5a151fec01ac656cccea0b7df9ac4c23c7b4b035`
- compact SHA-256:
  `a74b30f208fd982462fa09d9ee6d56b07ca968644a13b80d1d5f7f16ce8c01f8`

### `headline-quiet60-r3` — accepted

Exact command, launched from retained zsh PID `93082`:

```bash
nohup env -u GLM_MLX_WIRED_LIMIT_GB \
  -u GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB \
  UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONPATH=.:src \
  uv run python -m benchmarks.bench_dsv4_mtp_headline series \
  --source /Users/jack.mazac/models/DeepSeek-V4-Flash-0731 \
  --resident-dir /Users/jack.mazac/keep-artifacts/dsv4-residents-release-precision-20260819 \
  --vq-dir /Users/jack.mazac/keep-artifacts/dsv4-vq-e8p-g512 \
  --content-receipt /Users/jack.mazac/Developer/keep/artifacts/benchmarks/dsv4-task4-fix1-20260819/bind-proof.json \
  --content-receipt-sha256 b9c0b1afc40356649c4b5761f815acdd1bc49c65a0f8c04fdcf22b16c8931450 \
  --pack /Users/jack.mazac/models/teich/dsv4-coding-agent-v1-20260811.json \
  --output-dir /Users/jack.mazac/Developer/keep/artifacts/benchmarks/dsv4-task4-fix1-20260819/headline-quiet60-r3 \
  --pairs 5 --min-clean-pairs 3 \
  --prompt-tokens 64 --max-new-tokens 16 \
  --quiet-window-seconds 60 --quiet-max-attempts 3 \
  > /tmp/keep-task4-fix1-headline-quiet60-r3.log 2>&1 &
task4_r3_wrapper_pid=$!
disown
```

Wrapper PID `32606` and series/lock-owner PID `32608` completed. Both wired
variables were absent, every child reauthenticated receipt/source identity,
and the flock was released at terminal.

| Pair | Order | Baseline PID / s | Spec PID / s | Raw ratio | Decision gate |
|---:|---|---:|---:|---:|---|
| 0 | baseline first | 32611 / 7.026068207982462 | 33638 / 3.969319666008232 | 1.7700938193895241 | clean |
| 1 | speculative first | 38425 / 33.34343258300214 | 34343 / 7.211334958992666 | 4.623753129290198 | dirty: baseline pageouts +196 |
| 2 | baseline first | 40396 / 26.2373436249909 | 43351 / 7.9572340000013355 | 3.2972944650096374 | clean |
| 3 | speculative first | 46183 / 5.583616999996593 | 44239 / 3.8561993750045076 | 1.4479585874602416 | clean |
| 4 | baseline first | 47769 / 23.439610750006977 | 49584 / 7.9883577919972595 | 2.9342214458006364 | dirty: baseline pageouts +221 |

All ten rows recorded swapouts `0`. Every clean row recorded timed pageouts
`0`. The decision uses pairs 0, 2, and 3 only: three clean pairs, two
baseline-first and one speculative-first, exact token parity, consistent
acceptance counters, median `1.7700938193895241x`, verdict `PASS`.

Accepted compact evidence:

- summary SHA-256:
  `f0b9f2f098f68b24581409cec16ba22abf888992c5c4565b3090e6fa72a04dc1`
- compact evidence SHA-256:
  `78e30ca0af61351a0237d27ac6b87fa1cb554e31f18178bfa9de6b837acded0e`

The compact files preserve the row fields needed to reconstruct the decision
and counted unique dispatches. They do not contain raw-row file SHA-256
references. The large repeated raw traces and transient logs remain local and
are not committed.

## Final verification

The final verification set is:

```bash
UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONPATH=.:src \
  uv run --group dev python -m pytest \
  tests/test_dsv4_composite_loader.py \
  tests/test_bench_dsv4_mtp_headline.py -q

UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONPATH=.:src \
  uv run --group dev python -m pytest \
  tests/test_deepseek_v4_flash_adapter.py \
  tests/test_dsv4_composite_loader.py \
  tests/test_gather_vqmm_verify_rows.py \
  tests/test_bench_dsv4_mtp_headline.py -q

UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev ruff check \
  benchmarks/bench_dsv4_mtp_headline.py \
  src/mlx_vq/models/dsv4_composite_loader.py \
  tests/test_bench_dsv4_mtp_headline.py \
  tests/test_dsv4_composite_loader.py

UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev ruff format --check \
  benchmarks/bench_dsv4_mtp_headline.py \
  src/mlx_vq/models/dsv4_composite_loader.py \
  tests/test_bench_dsv4_mtp_headline.py \
  tests/test_dsv4_composite_loader.py

UV_CACHE_DIR=/tmp/keep-uv-cache uv run python -m compileall -q \
  benchmarks/bench_dsv4_mtp_headline.py \
  src/mlx_vq/models/dsv4_composite_loader.py

UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONPATH=.:src \
  uv run python -m benchmarks.bench_dsv4_mtp_headline validate-compact \
  --input artifacts/benchmarks/dsv4-task4-fix1-20260819/headline-quiet60-r3/headline-evidence.json \
  --bind-proof artifacts/benchmarks/dsv4-task4-fix1-20260819/bind-proof.json \
  --bind-proof-sha256 647501aa70a4531c3c0cedabc866025f7c2072c5c4bf671e70c83c8605df6115 \
  --min-clean-pairs 3

UV_CACHE_DIR=/tmp/keep-uv-cache PYTHONPATH=.:src \
  uv run python -m benchmarks.bench_dsv4_mtp_headline validate-compact \
  --input artifacts/benchmarks/dsv4-task4-fix1-20260819/headline-quiet30-r2/headline-evidence.json \
  --bind-proof artifacts/benchmarks/dsv4-task4-fix1-20260819/bind-proof.json \
  --bind-proof-sha256 647501aa70a4531c3c0cedabc866025f7c2072c5c4bf671e70c83c8605df6115 \
  --min-clean-pairs 3

git diff --check
```

First-review terminal results before commit `17fda71f`:

- focused: `16 passed, 2 warnings in 1.31s`;
- broad: `321 passed, 2 warnings in 152.91s` with wrapper PID `69880`, uv
  child PID `69882`, and terminal status `0`;
- Ruff check: `All checks passed!`;
- Ruff format: `4 files already formatted`;
- compileall: exit `0` through the repository `uv` environment;
- compact reconstruction: exit `0`, `PASS`, five pairs, three clean, two
  dirty, median `1.7700938193895241`, Metal `504`, NAX `12`;
- `git diff --check`: exit `0`;
- unsupported-word scan: no matches;
- frozen benchmark/runtime and all five compact artifact hashes match the
  values recorded above;
- no active headline process or heavy-lock holder remained.

Second-review terminal results immediately before the local commit:

- focused: `39 passed, 2 warnings in 10.58s`;
- broad: `344 passed, 2 warnings in 180.70s` with wrapper PID `40483`, uv
  child PID `40485`, and terminal status `0`;
- Ruff check: `All checks passed!`;
- Ruff format: `2 files already formatted`;
- compileall and `git diff --check`: exit `0`;
- accepted r3 compact validation against bind-proof SHA-256
  `647501aa70a4531c3c0cedabc866025f7c2072c5c4bf671e70c83c8605df6115`:
  exit `0`, `PASS`, five pairs, three clean, two dirty, both execution orders,
  median `1.7700938193895241`, Metal `504`, NAX `12`;
- preserved r2 compact validation against the same proof: exit `2`,
  `INCOMPLETE`, three pairs, two clean, one dirty, both execution orders;
- the authenticated canonical payload receipt remains
  `b9c0b1afc40356649c4b5761f815acdd1bc49c65a0f8c04fdcf22b16c8931450`;
- unsupported-word scan: no matches;
- no evidence packet was rewritten and no active headline process or
  heavy-lock holder remained.

Third-review terminal results immediately before the local commit:

- focused on the final formatted bytes: `43 passed, 2 warnings in 4.25s`;
- broad on the same bytes: `355 passed, 2 warnings in 133.46s` with wrapper
  PID `91445`, uv child PID `91447`, and terminal status `0`;
- Ruff check: `All checks passed!`;
- Ruff format: `2 files already formatted`;
- compileall and `git diff --check`: exit `0`;
- accepted r3 compact validation against the unchanged bind proof: exit `0`,
  `PASS`, five pairs, three clean, two dirty, and both execution orders;
- preserved r2 compact validation against the same proof: exit `2`,
  `INCOMPLETE`, three pairs, two clean, one dirty, and both execution orders;
- no evidence packet was rewritten and no active headline process or
  heavy-lock holder remained.

Fourth/fifth-review terminal results immediately before the local commit:

- focused on the final formatted bytes: `49 passed, 2 warnings in 3.92s`;
- broad on the same bytes: `361 passed, 2 warnings in 134.54s` with wrapper
  PID `13666`, uv child PID `13672`, and terminal status `0`;
- Ruff check: `All checks passed!`;
- Ruff format: `2 files already formatted`;
- compileall and `git diff --check`: exit `0`;
- accepted r3 compact validation against the unchanged bind proof: exit `0`,
  `PASS`, five pairs, three clean, two dirty, and both execution orders;
- preserved r2 compact validation against the same proof: exit `2`,
  `INCOMPLETE`, three pairs, two clean, one dirty, and both execution orders;
- no evidence packet was rewritten and no active headline process or
  heavy-lock holder remained.

## Limitations and boundary audit

- One deterministic 64-token report prefix and 16 emitted tokens cannot
  establish prompt-distribution or cross-machine performance.
- Timing is volatile even with quiet/pageout/swap gates. Both earlier losing
  rows and all later winning/dirty rows remain visible.
- The retained evidence proves 504 of 516 traced calls used Metal and 12 of
  516 used NAX. It does not retain a per-call eligibility explanation, prove
  NAX ownership of the verify window, or exclude a remaining Metal bottleneck.
- Load/bind pageouts and swapouts are never timing evidence.
- The drafter has no independent numerical reference. Repeated acceptance
  signatures and exact AR/spec token parity are consistency evidence only.
- Task 2 remains STOP. No threshold, split, row, output identity, or source
  byte was changed during the accepted series. No dirty row was reclassified.
- No custom wired-memory limit, holdout access, Task 5 work, external system,
  model-payload commit, artifact overwrite, or push occurred.
