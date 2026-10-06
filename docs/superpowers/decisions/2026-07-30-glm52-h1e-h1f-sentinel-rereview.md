# 2026-07-30 H.1e / H.1f identity sentinel re-review

Status: **accepted by the owner at 2026-07-30T21:32:44Z.**

Handoff `docs/superpowers/plans/2026-07-27-glm52-full-campaign-run-handoff.md`
§6 requires: *"Confirm all preexisting accepted H.1e and H.1f identity sentinels
remain exact, or document and re-review any deliberately replaced dependency."*
They do not remain exact. This is that documentation and re-review.

## How the drift was found

An earlier check compared sentinel occurrences between `HEAD` and the working
tree and reported PASS. That method was wrong: it can only prove the working
tree did not change them, and cannot see drift already committed at `HEAD`. The
correct gate recomputes each pinned path's whole-file SHA-256 against the
accepted report inventories.

- H.1e immutable sentinels, `.superpowers/sdd/goal-objective/task-3ph1e-production-generation-authority-report.md`
  lines 267–286: **20 pinned → 13 matched, 7 drifted**.
- H.1f frozen dependencies and owned identities, `.superpowers/sdd/goal-objective/task-3ph1f-production-fence-audit-report.md`
  lines 69–94: **18 pinned → 12 matched, 6 drifted**.

13 pins over 10 distinct files. Three files are pinned by both reports.

## Provenance

Every drifted file is clean in the working tree. The drift is committed history
from 2026-07-29, predating the 2026-07-30 remediation session, and was produced
by the enforcement-plane wave that migrated this code into the native
`glm52_enforcement` package.

| Commit | Subject | Drifted paths |
|---|---|---|
| `c9e01e3a` | feat: implement GLM-5.2 production dec… | `glm52_sky_production_submission.py`, `glm52_sky_submission_modes.py`, `glm52_sky_production_generation.py`, `glm52_sky_production_fence.py` |
| `ea166a79` | fix: close guarded GLM-5.2 production submission | `submit_sky_campaign.py`, `submit_sky_campaign.sh`, `test_glm52_sky_submission_integration.py` |
| `7f72301a` | fix: propagate GLM-5.2 worker launch authority | `glm52_sky_campaign.py`, `glm52_gpu_spend_snapshot.py` |
| `a5378865` | feat: execute guarded GLM-5.2 campaign routes | `test_glm52_h100_qualification.py` |

## Exact identity inventory, pinned to current

Machine-readable authority: `docs/superpowers/decisions/2026-07-30-glm52-h1e-h1f-sentinel-manifest.json`  
Manifest SHA-256: `3f931ed1dd7195c897ee91adb20975ce0fb7ef58022be00a7c273c342696d1c8`

The manifest records all 38 report rows across 30 distinct paths, including the 20 unchanged paths. The table below is the exact 10-path drift set. `Pinned` and `Current` are full SHA-256 digests; line and byte counts bind both historical and current byte sequences.

| Path | Pinned SHA-256 | Pinned size | Current SHA-256 | Current size |
|---|---|---:|---|---:|
| `aws/glm52-gpu/scripts/submit_sky_campaign.py` | `11d6d8d5d44e4d008f6fadd1f3e2d6647f36ea6f7879f4fb3f73af2abc8fee10` | 7181 lines / 256656 bytes | `113c327825017b8c53b85959104910ca653c33a7b537db689b9977b64ca02c25` | 7704 lines / 276096 bytes |
| `aws/glm52-gpu/scripts/submit_sky_campaign.sh` | `9dc16be14d60769aef6f1d1c2f84c2da3aab1be9c38a3852ef8e3bbf5f37d84d` | 39 lines / 1355 bytes | `33a4520edeef8cf907fc637fd3119fba227f9f9cf81c41ca1b1d5a8abbf791c3` | 82 lines / 2841 bytes |
| `src/mlx_vq/quality/glm52_gpu_spend_snapshot.py` | `b805a75030717c4d67791eca991c03217193ec690e2c26034cddd7036a8806c1` | 834 lines / 30478 bytes | `99ead043e67d0e924efed5d0b3600e56b484a728b6f3b7dd2fdae977f4de0dfb` | 39 lines / 1405 bytes |
| `src/mlx_vq/quality/glm52_sky_campaign.py` | `f34ecc4ed5293dc36f98dc172021f991e4354d3b3ea6d32f7bf0b9ee3cb7a40a` | 1106 lines / 42146 bytes | `77412115f6b8ce325177864418d838e0335d536329e8e7565d7811bcc806cd94` | 133 lines / 4439 bytes |
| `src/mlx_vq/quality/glm52_sky_production_fence.py` *(H.1f owned)* | `28e7584e1cdba0f4403385388e08d8e76e6176e3cae63069f14d2b6692c2e9b8` | 2209 lines / 76461 bytes | `1180ed7e0e9f6527282c14a07fad0a382c9dd1c25b7dd80d74fa9bb8c3d7ff88` | 6 lines / 179 bytes |
| `src/mlx_vq/quality/glm52_sky_production_generation.py` | `768dbd4edb53a494cf04d3c1e6d6e37cf9918e254f534fd30fbb0f523a1ff79e` | 3482 lines / 140208 bytes | `7180b65da01ea0b337d0892e544c5e9fadb8e798beec1f23fdb0f8d0d58d3632` | 6 lines / 189 bytes |
| `src/mlx_vq/quality/glm52_sky_production_submission.py` | `c5d0a566842bdecadb1d48010aba6f74f967f25e60f9fc0f98e13ad1caf5a918` | 1650 lines / 63272 bytes | `4e8da5d19cc2791e4992aafc6a4a63e0c04f473e26bba5a9b1a39e25f5de9f57` | 6 lines / 192 bytes |
| `src/mlx_vq/quality/glm52_sky_submission_modes.py` | `75e32add1d89bd2a26c4617568e2a45f9297804a1955056137eea6146cc516c7` | 281 lines / 7276 bytes | `52c4aad2810883b3628ef173b649019ce79e7d799b28a7fd45117a668000d35d` | 6 lines / 192 bytes |
| `tests/test_glm52_h100_qualification.py` | `02004f721b3e86a3d374cfc4eaa3dc83f79935091b616852f29b7e3944e89347` | 559 lines / 21166 bytes | `fce95da719342703cadb1853879e0b8ff6147f1abbfe882e15bcb5e211d463d0` | 568 lines / 21579 bytes |
| `tests/test_glm52_sky_submission_integration.py` | `b24c721a72a13f047813f4f659409065166cdf873fcb972884aac8c69cc6901b` | 7097 lines / 236773 bytes | `41a7ae27fb753afb6cff7d74a3f8f309b3dd09292f339ad4cb838210d917e93a` | 7392 lines / 246136 bytes |

### Recovery coverage

The manifest's pinned blob references are Git-recoverable for **28 of 30**
distinct paths. Two exact, unchanged H.1f dependencies are exceptions:

- `.superpowers/sdd/goal-objective/task-3ph1e-production-generation-authority-brief.md`
- `.superpowers/sdd/goal-objective/task-3ph1e-production-generation-authority-report.md`

Those ignored `.superpowers` documents have no Git commit or blob reference and
their bytes are not embedded in the manifest. Their current workspace bytes
rehash exactly to the pinned SHA-256 values and match the recorded line/byte
counts, so the manifest binds their present identities; it **cannot reconstruct
them after workspace loss**. Owner acceptance of this re-review does not create
a durable recovery copy for those two files.

## Semantic classification

Eight of ten changes are **benign supersession** — the legacy module became a
compatibility facade over the enforcement-native implementation, or a test
gained coverage. Notably `glm52_sky_production_fence.py`, an H.1f *owned*
identity, differs from the native fence only by docstring and relative-import
relocation; its fail-closed logic and its no-I/O, no-authority surface are
intact.

Two are genuine **contract changes**, both in the guarded submission entrypoint:

- `aws/glm52-gpu/scripts/submit_sky_campaign.py` now requires a Task 13 launch
  bridge, reserves an identity-safe outcome file, rejects direct production
  task/config/Sky inputs, asserts no raw-effect surface, and validates an exact
  pinned Sky executable, version, and parser before launch.
- `aws/glm52-gpu/scripts/submit_sky_campaign.sh` appends environment-sourced
  authority explicitly as `--production-authority`, retaining the existing
  authority/H100 binding and account/SNS guards.

Both tighten the production contract. Neither loosens it.

## Governing-constraint audit

Checked adversarially for any relaxation. **No constraint risk found.**

| Constraint | Finding |
|---|---|
| Account `246813579024` | Touched only by a new equality guard; remains exact, no widening |
| Region `us-west-2` | Same guard; no alternate-region path added |
| One on-demand `p5.48xlarge` | No broadened shape, count, or market rule |
| 24 h / `$1,320.96` GPU envelope | Untouched; spend-snapshot change is import plumbing |
| 900 s / `$13.76` residual reserve | Untouched |
| 300-GiB root tail ≤ `$0.01` | Untouched |
| Spot / Capacity Blocks / alt shapes / raw launch | **No deny rule relaxed.** The entrypoint change hardens raw-launch control; tests retain Spot and Capacity Block absence checks |
| Exact JSON `null` semantics | No schema-defined null field or validator changed |

## Behavioural reproduction

Every verification command the two reports name was re-run and reproduced its
pinned baseline exactly:

| Command | Baseline | Observed |
|---|---|---|
| H.1f focused: `test_glm52_sky_production_fence.py` + `…_fence_audit.py` | 175 passed | **175 passed** |
| H.1f compatibility A: `…submission_modes.py` + `…production_generation.py` | 351 passed | **351 passed** |
| H.1f compatibility B: `…production_submission.py` + `…production_acquisition.py` | 513 passed | **513 passed** |
| H.1e sentinel-file suite, all 20 pinned paths' tests | — | **1242 passed** |

Static verification on H.1f's four owned files: `compileall` pass, Python 3.9
`py_compile` of the audit module pass, `ruff --select E4,E7,E9,F` clean,
`git diff --check` clean.

So the accepted behavioural contracts hold at their documented numbers even
though the file bytes changed.

## Owner acceptance

At `2026-07-30T21:32:44Z`, the owner replied **“Accept. Execute next
steps.”** This accepts:

1. the stricter production-entrypoint interface — a Task 13 launch bridge and
   prior exact Sky control-plane validation are mandatory for production
   submission; and
2. supersession of the 13 historical whole-file pins by the current identities
   recorded in the machine-readable manifest.

No governing-constraint exception was accepted. Handoff §6's identity-sentinel
item is closed, and Phase Two stands at **9 of 9**.
