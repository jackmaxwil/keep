# Task 3 — DeepSeek-V4-Flash resident artifact report

Date: 2026-08-19 (America/Los_Angeles)

Baseline: `789242a7da4f98d0003ac32f77635cbf4d6027a5` on local `main`

Scope: Task 3 only. Task 2 remains a reviewed negative speed gate/STOP; the
user's authorization to continue was not interpreted as a Task 2 pass.

## Result

PASS for Task 3's resident-artifact scope.

The real resident artifact is:

`/Users/jack.mazac/keep-artifacts/dsv4-residents-release-precision-20260819`

It is a new 13-file strict tree: 11 safetensors shards, one safetensors index,
and one manifest. It contains every mapped non-routed source tensor from the
pinned DeepSeek-V4-Flash checkpoint, including backbone and `mtp.{0,1,2}`
residents, exactly once. It contains no routed-expert tensor. The existing
routed VQ artifact was read for identity/size only and was not mutated.

The package preserves the released/tested precision split rather than
inventing an all-FP8 conversion:

- 390 affine weights remain `F8_E4M3` with 390 `F8_E8M0` 128×128 scale
  companions.
- Embeddings, output head, norms, routers, hyper-connection values, routing
  tables, and other higher-precision source tensors retain their source dtype.
- Routed backbone and MTP expert `.weight`/`.scale` tensors are deliberately
  excluded because `/Users/jack.mazac/keep-artifacts/dsv4-vq-e8p-g512` owns
  them.

## Implementation

Owned code/test paths:

- `src/mlx_vq/convert/dsv4_resident.py`
  - DSV4-specific pinned authority and inventory oracle.
  - Raw-range repack using the existing GLM52 non-VQ safetensors
    plan/write/hash/audit primitives.
  - Atomic new-directory publication; refuses to overwrite an existing target.
  - Full source-relative byte audit and strict tree enforcement.
  - CLI-held cooperative `.keep-heavy-job.lock` and fail-closed wired-limit
    environment check.
- `src/mlx_vq/convert/glm52_non_vq.py`
  - One-line addition of `F8_E8M0: 1` to the shared safetensors dtype-size map.
- `tests/test_dsv4_resident_pack.py`
  - One end-to-end raw-pack regression covering released mixed precision,
    block-scale companions, MTP inclusion, routed exclusion, inventory
    accounting, and source-relative audit.

No parallel conversion framework, dependency, config layer, cloud rail, or
Task 4 runtime was added.

## RED → GREEN evidence

RED:

```text
env -u GLM_MLX_WIRED_LIMIT_GB \
  -u GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB \
  UV_CACHE_DIR=/tmp/keep-uv-cache \
  uv run --group dev python -m pytest tests/test_dsv4_resident_pack.py -q
```

Initial terminal result: collection failed with
`ModuleNotFoundError: No module named 'mlx_vq.convert.dsv4_resident'`.

After the DSV4 wrapper existed, the same test found the missing shared raw
dtype support:

`ValueError: unsupported safetensors dtype 'F8_E8M0'`

The minimal additional production change was the one-line `F8_E8M0` byte-size
entry in the existing raw-pack dtype table.

GREEN:

```text
env -u GLM_MLX_WIRED_LIMIT_GB \
  -u GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB \
  UV_CACHE_DIR=/tmp/keep-uv-cache \
  uv run --group dev python -m pytest tests/test_dsv4_resident_pack.py -q
```

Terminal result: `1 passed, 2 warnings in 1.33s`.

## Source identity and exact inventory

Pinned source:

- Path: `/Users/jack.mazac/models/DeepSeek-V4-Flash-0731`
- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- `config.json` SHA-256:
  `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- `model.safetensors.index.json` SHA-256:
  `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- Source shards: 48
- Source index tensors: 72,317

Exact resident partition:

| Item | Count / bytes |
|---|---:|
| Retained source tensors | 1,661 |
| Logical model-parameter tensors | 1,271 |
| FP8 scale companions | 390 |
| Backbone retained tensors | 1,564 |
| `mtp.{0,1,2}` retained tensors | 97 |
| Logical resident parameters (scales excluded) | 7,827,675,070 |
| Resident storage elements (scales included) | 7,828,059,838 |
| Resident tensor payload bytes | 9,441,141,496 |
| Excluded routed `.weight`/`.scale` tensors | 70,656 |
| Excluded routed source payload bytes | 157,437,394,944 |
| Missing resident tensors | 0 |
| Duplicate resident tensors/targets | 0 |
| Routed tensors in resident artifact | 0 |
| Unmapped non-routed source tensors | 0 |

Dtype tensor counts:

| Dtype | Tensors | Payload bytes |
|---|---:|---:|
| `BF16` | 445 | 2,967,134,976 |
| `F32` | 433 | 150,966,520 |
| `F8_E4M3` | 390 | 6,304,038,912 |
| `F8_E8M0` | 390 | 384,768 |
| `I64` | 3 | 18,616,320 |

The pinned plan refuses production publication unless all of these counts,
bytes, dtypes, config facts, revision facts, and hashes match exactly.

## Build lifecycle and lock proof

Output absence, lock availability, wired-limit absence, and disk capacity were
checked before launch. The output directory did not exist and 972 GiB was free.

Two initial non-interactive detached-shell attempts returned before starting
the Python process. Both left a zero-byte log, no JSON, no final artifact, and
no partial artifact directory. They were not treated as evidence or success.

The successful run was launched from a persistent interactive zsh using:

```text
nohup env -u GLM_MLX_WIRED_LIMIT_GB \
  -u GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB \
  UV_CACHE_DIR=/tmp/keep-uv-cache \
  uv run --group dev python -m mlx_vq.convert.dsv4_resident \
  --source-dir /Users/jack.mazac/models/DeepSeek-V4-Flash-0731 \
  --output-dir /Users/jack.mazac/keep-artifacts/dsv4-residents-release-precision-20260819 \
  --output-json /Users/jack.mazac/Developer/keep/artifacts/quality/dsv4-task3-20260819/package-run.json \
  --max-shard-payload-bytes 1000000000 \
  --heavy-lock-path /Users/jack.mazac/Developer/keep/.keep-heavy-job.lock \
  > /Users/jack.mazac/Developer/keep/artifacts/quality/dsv4-task3-20260819/package-run.log 2>&1 \
  & disown
```

Observed process: PID 51841 (`uv`), worker PID 51843. The CLI held
`/Users/jack.mazac/Developer/keep/.keep-heavy-job.lock` across pack and audit.
The terminal JSON records `wired_limit_variables_absent: true`.

Lifecycle:

- Started: `2026-08-20T02:05:32.471229+00:00`
- Completed: `2026-08-20T02:06:13.725889+00:00`
- Status: `complete`
- Partial publication directory: none remains
- Existing payload overwritten: none

The 1 GB shard target was exceeded only by two indivisible 1,059,061,856-byte
single-tensor shard files (embedding/head). No source tensor was split,
converted, or re-encoded.

## Artifact identity and audit

- Resident manifest:
  `/Users/jack.mazac/keep-artifacts/dsv4-residents-release-precision-20260819/resident-manifest.json`
- Manifest SHA-256:
  `35fa6361691326aa805a5ff59a95508af1740660b98844071da59856ae8009d5`
- Package-set SHA-256:
  `75b63dee580fc4038be088db63ef9e5dd2cb0256bb722abb77d94651ee51abe9`
- Shards: 11
- Artifact files: 13
- Artifact tree bytes: 9,442,336,557
- Tensor payload bytes: 9,441,141,496

The manifest contains every retained source tensor name, target name, dtype,
shape, source/output shard and byte offsets, parameter/storage counts, and
payload SHA-256, plus every output shard's SHA-256 and the output index hash.

The build-time audit and a fresh post-format audit both passed:

```text
{"audit_pass": true,
 "checks": {"accounting": true, "index": true, "lineage": true,
 "payload_hashes": true, "physical_extents": true, "selection": true,
 "source_byte_identity": true, "source_inventory": true,
 "strict_tree": true},
 "artifact_tree_bytes": 9442336557,
 "retained_tensor_count": 1661,
 "tensor_payload_bytes": 9441141496}
```

Fresh audit runtime: 26.7 seconds.

## Production loader/binder and numerical proof

The proof used the production resident binder:

`ramp.models.deepseek_v4_flash_adapter.bind_deepseek_v4_flash_non_vq_weights`

It loaded the resident artifact's own `model.safetensors.index.json`, discovered
all files, decoded the block-FP8 residents into BF16, preserved direct and
integer tensors, and bound the full backbone plus all three MTP resident
modules. It was run under the same cooperative heavy-job lock with both custom
wired-limit variables absent, via `nohup ... & disown` from a persistent zsh.

Terminal proof:

| Check | Result |
|---|---:|
| Bound parameters | 1,271 |
| Backbone bound | 1,199 |
| MTP bound | 72 |
| FP8 block-decoded weights | 390 |
| Direct tensors | 878 |
| Preserved integer tensors | 3 |
| Allowed `wo_a` reshapes | 46 |
| Missing model parameters | 0 |
| Unmatched source tensors | 0 |
| Dense routed parameters | 0 |
| Bind time | 10.1703 s |
| Total proof time | 11.5880 s |
| MLX active bytes | 17,504,406,868 |
| MLX peak bytes | 18,326,523,356 |

Routed backbone and MTP experts remained unbound as expected for a resident-only
proof. That is a scope boundary, not a missing resident parameter.

Representative source-paired numerical validation used the same production
binder on the pinned source for layers 0 and 42 plus all MTP residents. All
values were finite in BF16. The documented bound was max absolute error `<= 0`;
the observed maximum and mean error were exactly `0.0` for all six tensors:

- `model.layers.0.attn.wq_a.weight`
- `model.layers.0.ffn.shared_experts.gate_proj.weight`
- `model.layers.42.attn.wq_b.weight`
- `model.layers.42.ffn.shared_experts.down_proj.weight`
- `mtp_drafter.blocks.0.main_proj.weight`
- `mtp_drafter.blocks.2.attn.wo_b.weight`

The representative numerical proof complements, rather than replaces, the
full 1,661-tensor source-byte audit.

## Measured resident and combined sizes

Existing routed artifact identity:

- Path: `/Users/jack.mazac/keep-artifacts/dsv4-vq-e8p-g512`
- Manifest SHA-256:
  `a996c4bedc514f49c54fe15d5d9b2373f839678e55b241861f2fda60cb4bf8b3`
- Runtime routed payload: 138 safetensors files, 75,246,105,792 bytes
  (75.246105792 GB / 70.078396976 GiB)

Measured sizes:

| Scope | Bytes | Decimal GB | GiB |
|---|---:|---:|---:|
| Resident tensor payload | 9,441,141,496 | 9.441141496 | 8.792748205 |
| Resident artifact tree | 9,442,336,557 | 9.442336557 | 8.793861193 |
| Routed runtime payload | 75,246,105,792 | 75.246105792 | 70.078396976 |
| Combined tensor payload | 84,687,247,288 | 84.687247288 | 78.871145181 |
| Resident tree + routed payload | 84,688,442,349 | 84.688442349 | 78.872258169 |

The measured combined payload, 84.687 GB, is inside the prior 82.5–85.1 GB
estimate. It is 2.187 GB above the lower bound and 0.413 GB below the upper
bound. This is a measurement from real artifact bytes, not reworked accounting.

The routed size intentionally uses the routed manifest's
`audit.artifact_payload_bytes` (the 138 runtime safetensors files), excluding
the calibration cache and logs that happen to live beside that payload.

## Compact committed evidence

- `artifacts/quality/dsv4-task3-20260819/package-run.json`
  - Full manifest/inventory, run lifecycle, lock/env proof, and build-time audit.
  - SHA-256:
    `2d7b4c928c5370997bb79f1548d778e4ed37ad1669199d084032ac961c14c4d0`
- `artifacts/quality/dsv4-task3-20260819/bind-proof.json`
  - Full production bind counts, memory, and six numerical rows.
  - SHA-256:
    `88e5b9970a366187d1a2d704103c54aed156bb53bd3b1cce8ccf5160e2deed49`
- `artifacts/quality/dsv4-task3-20260819/size-summary.json`
  - Resident/routed identities and byte arithmetic.

The 9.44 GB resident payload itself is not committed.

## Final verification commands and outcomes

Focused implementation/integration/checkpoint coverage:

```text
env -u GLM_MLX_WIRED_LIMIT_GB \
  -u GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB \
  UV_CACHE_DIR=/tmp/keep-uv-cache \
  uv run --group dev python -m pytest \
  tests/test_dsv4_resident_pack.py \
  tests/test_glm52_non_vq_pack.py \
  tests/test_deepseek_v4_flash_adapter.py::test_bind_layer_zero_residents_from_the_real_checkpoint \
  tests/test_deepseek_v4_flash_adapter.py::test_checkpoint_tensors_and_model_parameters_are_a_bijection \
  -q
```

Outcome: `36 passed, 2 warnings in 1.43s`.

Static checks:

```text
UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev ruff check \
  src/mlx_vq/convert/dsv4_resident.py tests/test_dsv4_resident_pack.py
UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev ruff format --check \
  src/mlx_vq/convert/dsv4_resident.py tests/test_dsv4_resident_pack.py
UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m compileall -q \
  src/mlx_vq/convert/dsv4_resident.py tests/test_dsv4_resident_pack.py
git diff --check
```

Outcomes: Ruff `All checks passed!`; format `2 files already formatted`;
compileall exit 0; diff check exit 0.

Broader adapter observation:

```text
uv run --group dev python -m pytest \
  tests/test_dsv4_resident_pack.py tests/test_deepseek_v4_flash_adapter.py -q
```

Outcome: `76 passed, 1 failed`. The failure is
`test_forward_is_finite_in_bfloat16`: the test expected BF16 logits but the
current forward returned F32. A fresh isolated run failed identically. Task 3
does not modify forward execution or output dtypes, so this existing out-of-scope
failure was not changed or relabeled as green.

## Full relevant command history

Read-only inspection included:

- `git status --short --branch`, `git rev-parse HEAD`, and `git log -6`.
- Reading `AGENTS.md`, `docs/HANDOFF.md`, the Task 3 brief, the production
  binder, precision policy, GLM52 non-VQ pack/audit rail, and their tests.
- `rg` searches for resident, DSV4, FP8/block-scale, manifest, loader, and binder
  symbols/callers.
- `shasum -a 256` over pinned source config/index and final evidence/manifests.
- Header-only Python inventory over all 48 source shards.
- A production plan-only preflight that returned 1,661 retained tensors,
  7,827,675,070 logical parameters, 9,441,141,496 payload bytes, 11 shards,
  and the exact expected dtype/backbone/MTP/routed partition.
- `df -h`, wired-limit environment inspection, output-absence checks, process
  polls, shard size polls, lock-holder checks, and final partial-directory check.

Mutating commands were limited to:

- `apply_patch` for the owned code, regression, size evidence, and this report.
- `ruff format` on the two new Python files only.
- `mkdir -p artifacts/quality/dsv4-task3-20260819`.
- The successful locked/nohup resident build shown above.
- Copying the valid JSON bind log to the committed `.json` evidence path.
- Explicit Git staging/commit described below; no push.

No holdout path, cloud API, external system, Task 4 runtime, routed payload
writer, custom MLX wired limit, macOS wired-limit mutation, push, branch switch,
or destructive command was used.

## Self-review and limitations

- The wrapper deliberately reuses the existing GLM52 raw-range safetensors
  primitives, including private same-package helpers. That keeps byte writing,
  hashing, physical-extent checks, and source-relative audit single-sourced,
  but a future rename of those internal helpers must update this wrapper.
- The artifact is resident-only. It has not been composed with the routed VQ
  payload or exercised by Task 4's full speculative runtime; routed experts are
  intentionally unbound in the Task 3 load proof.
- The numerical check is representative (six early/late/MTP affine tensors),
  while the independent byte audit is exhaustive over all 1,661 retained
  source tensors.
- No numerical approximation was introduced by packaging: the artifact copies
  source bytes exactly, and the production binder's decoded representatives
  matched source at a zero max-absolute-error bound.
- The sealed 37-session holdout was not inspected.
- Task 2's negative speed gate remains negative and is not contradicted by this
  storage/load result.

## Review fix round 1: fail-closed publication

The resident pack now runs its mandatory exhaustive audit against the private
staging tree before the atomic rename to the official output path. The audit
checks the staging bytes against the manifest's intended published path. If it
fails, packaging raises, removes that unique staging tree, and never creates the
official output. The existing post-publication CLI audit remains as an
additional success-path check. The already-built real artifact was not rebuilt;
it had already passed that post-publication audit.

### RED evidence

Command:

```text
env -u GLM_MLX_WIRED_LIMIT_GB -u GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_dsv4_resident_pack.py::test_failed_mandatory_audit_never_publishes_official_output -q
```

Output before the production fix:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_________ test_failed_mandatory_audit_never_publishes_official_output __________

tmp_path = PosixPath('/private/var/folders/cv/yvk5v9ss6hb4vgrv_70hbtxh0000gp/T/pytest-of-jack.mazac/pytest-28/test_failed_mandatory_audit_ne0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x118d826c0>

    def test_failed_mandatory_audit_never_publishes_official_output(
        tmp_path, monkeypatch
    ):
        source, output, index, config_sha, index_sha = _fixture_source(tmp_path)

        def fail_audit(*args, **kwargs):
            raise ValueError("forced mandatory audit failure")

        monkeypatch.setattr(resident_module, "audit_dsv4_resident_package", fail_audit)

>       with pytest.raises(ValueError, match="forced mandatory audit failure"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ValueError

tests/test_dsv4_resident_pack.py:132: Failed
=============================== warnings summary ===============================
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_dsv4_resident_pack.py::test_failed_mandatory_audit_never_publishes_official_output
1 failed, 2 warnings in 1.19s
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

This is the intended RED: the forced mandatory audit was never called, so the
pack returned rather than raising and had already published the official path.

### GREEN evidence

Focused command (identical to RED):

```text
env -u GLM_MLX_WIRED_LIMIT_GB -u GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_dsv4_resident_pack.py::test_failed_mandatory_audit_never_publishes_official_output -q
```

Output:

```text
.                                                                        [100%]
=============================== warnings summary ===============================
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 2 warnings in 0.68s
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

The regression forces the audit to raise and proves both that the official
output does not exist and that no `resident.partial-*` tree remains.

Directly relevant pack suites:

```text
env -u GLM_MLX_WIRED_LIMIT_GB -u GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_dsv4_resident_pack.py tests/test_glm52_non_vq_pack.py -q
```

Output:

```text
...................................                                      [100%]
=============================== warnings summary ===============================
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
35 passed, 2 warnings in 1.12s
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```
