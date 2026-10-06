# 2026-07-30 GLM-5.2 Task 13 controller signing-key rotation (v2 → v3)

## Why

The pinned Task 13 controller signing private key was stored only at
`$FULL_RUN_WORK/controller-authority-ed25519` under `/private/tmp`. That
directory was cleared, so the key is **unrecoverable**. It was never escrowed.

`ControllerAuthoritySigner` derives the public half with `ssh-keygen -y` and
rejects any drift from the pinned public key and fingerprint
(`src/glm52_enforcement/task13_controller_authority.py`). Exactly one private
key can satisfy that contract, so a generated replacement could not be adopted
silently and the lost key could not be worked around.

Consequences observed before rotation:

- `issue_controller_execution_authority` / `issue_controller_operation_capability`
  could not sign. Callers on the live path are
  `aws/glm52-gpu/scripts/run_glm52_task13_campaign.py:181`,
  `aws/glm52-gpu/scripts/issue_glm52_task13_controller_authority.py:202`, and
  `src/glm52_enforcement/task13_campaign_runner.py:627`.
- `tests/test_glm52_task13_campaign_runner.py` and
  `tests/test_glm52_task13_controller_authority.py` could not be collected,
  because both resolve the key at import time.
- Verification was unaffected: it needs only the public pin.

## Blast radius, measured before rotating

- Checkout: the pin appears in exactly two source locations plus one test
  assertion — `src/glm52_enforcement/task13_controller_authority.py`,
  `aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py`,
  `tests/test_glm52_task13_controller_authority.py`. `_PINNED_PUBLIC_CORE` is
  derived, not a separate literal.
- Remote: scanned `s3://keep-glm52-models-246813579024-us-west-2/campaigns/glm52-sky-20260724/authorities/`
  and all of `task13/` — 12 JSON objects, **0** containing the v2 public key,
  fingerprint, or `controller-v2` label. Three `.zip` bundles were not scanned.
- DynamoDB: no tables exist in `us-west-2`, so no ledger records reference it.
- `task13/gates/t01-t25.json` and `task13/gates/transport-22-mutants.json`
  contain **0** references to controller/signer/signature, so the sealed
  transport gates do not require resealing for this rotation.

Nothing signed under v2 was found. Historical v2 verification is therefore
retired rather than preserved as a dual-pin path.

## Decision

Rotate the key identity only. The signing namespace
(`keep-glm52-task13-controller-authority-v2`) and the authority/capability
record types are protocol identifiers, not key identity, and are unchanged.

- New key comment: `keep-glm52-task13-controller-v3`
- New public key: `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDGJndv7GTsFzrh/fGs82pm/W6MmUvOsnPlP28IKugex keep-glm52-task13-controller-v3`
- New fingerprint: `SHA256:8nCALZDwL4X59iIu+znZj5QBKJCUi/c5tjkMR39Q5mM`
- Generated with the pinned `/usr/bin/ssh-keygen`, whose SHA-256 was verified
  equal to `OPENSSH_EXECUTABLE_SHA256`
  (`3ebf8d762494bb9e23be1955a8e136a6f3853512ea1f693fab96e0986bb9304f`).

The module-level guard that recomputes the fingerprint from the pinned public
key and raises `RuntimeError` on drift passes on import.

## Custody contract the key must satisfy

From `ControllerAuthoritySigner.__init__`:

- `full_run_work` absolute, not a symlink, `resolve(strict=True)` equal to
  itself, a directory, mode exactly `0o700`, owned by the calling uid.
- key at `<full_run_work>/controller-authority-ed25519`, absolute, not a
  symlink, regular file, mode exactly `0o600`, owned by the calling uid,
  `st_nlink == 1`.
- `.pub` sibling: not a symlink, regular file, mode exactly `0o600`, owned by
  the calling uid, `st_nlink == 1`, contents exactly
  `PINNED_SIGNER_PUBLIC_KEY + "\n"`.

Because the key must be a real file at an exact path with exact permissions, a
durable master plus controlled per-run materialization is required. A symlink,
hardlink, or relocated key is rejected.

## Custody, as implemented and exercised

The durable master is AWS Secrets Manager in the pinned account. There is no
long-lived local copy: the plaintext master used to seed escrow was removed
after the escrow restore was proven to produce a byte-identical key.

- Secret: `keep/glm52/task13/controller-authority-ed25519-v3`
- ARN: `arn:aws:secretsmanager:us-west-2:246813579024:secret:keep/glm52/task13/controller-authority-ed25519-v3-vzKBDU`
- Version id: `5381d275-1ae2-4e0f-bc3b-cbfb4a7c1152`
- Public fingerprint: `SHA256:8nCALZDwL4X59iIu+znZj5QBKJCUi/c5tjkMR39Q5mM`

No private material is recorded here or in any evidence artifact.

`aws/glm52-gpu/scripts/restore_glm52_task13_controller_key.py` materializes the
key per run. It pins account `246813579024` and region `us-west-2` and refuses a
foreign account before fetching the secret; validates the caller's raw path with
`lstat` before resolving, so a symlinked work directory cannot pass; refuses to
touch a directory that is not an exact `0o700` private directory; refuses to
overwrite existing custody material; creates both halves with
`O_CREAT|O_EXCL|O_NOFOLLOW` at `0o600` so creation itself is the exclusivity
guarantee rather than check-then-replace; removes only halves that call actually
created, so losing an `O_EXCL` race never deletes another process's file; proves
custody by constructing `ControllerAuthoritySigner`; and offers `--wipe`, which
validates the same exact `0o700` directory before deleting anything.

Round trip exercised, not merely documented: restore into a fresh `0o700`
directory, custody verified at mode `0600` / `nlink 1` / correct uid for both
halves, a real 464-byte base64 SSHSIG produced through the signer boundary, then
wiped clean. `tests/test_glm52_task13_controller_key_restore.py` pins these
invariants with seven tests, using an ephemeral key and a fake session; it never
reads the production key and never calls AWS.

## Test proof boundary

The clean-checkout gate no longer needs the production private key.
`tests/glm52_task13_signer_support.py` generates an ephemeral Ed25519 key in a
`0o700` directory and patches the module's complete pin set --
`PINNED_SIGNER_PUBLIC_KEY`, `PINNED_SIGNER_FINGERPRINT`, and derived
`_PINNED_PUBLIC_CORE` -- because `issue_controller_execution_authority` records
the module-level fingerprint and then calls
`verify_controller_execution_authority`, whose `_verify_signature` reads the
module-level pinned key. Constructor-only injection was attempted, found
insufficient for that reason, and reverted rather than left as a production
bypass. `_PINNED_PUBLIC_FINGERPRINT` is used only by the import-time consistency
check and is deliberately not patched. The production factory remains pinned to
v3 with no injection point. The pin-literal assertion in
`tests/test_glm52_task13_controller_authority.py` still asserts the real v3
literals, captured before patching.

## What is NOT yet done

**New archive/activation lineage is required.** Rotating the pin changed the
bytes of `aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py`, and
therefore its self-SHA and the campaign package identity that binds it. The
coordinator SHA is now
`259e490d2f02380c6a33b79608660d1b65eea56b08469772ce64f72d00ac4723`. The
already-staged `s3://.../task13/archive/repo-tar.json` and any package,
approval, or rehearsal bound to the previous archive **cannot authorize this
checkout**. Do not overwrite or reuse that frozen archive. Create a new
activation and archive lineage, bind approvals and a fresh clean rehearsal to
its new hashes, and preserve the existing S3 objects for audit.

## Verification performed

Baseline before this rotation, for contrast: the enforcement gate could not run
as written -- `tests/test_glm52_task13_campaign_runner.py` and
`tests/test_glm52_task13_controller_authority.py` failed collection, and the
suite stood at 3 failed / 8937 passed with those two files excluded.

Final, after rotation and the test-boundary split:

- Focused campaign gate (handoff §6 exact 10-file list): **577 passed, 0 failed**.
- Enforcement gate with **no exclusions**
  (`tests/test_glm52_enforcement_*.py tests/test_glm52_task*.py
  tests/test_glm52_h1g_stack_migration.py`): **2558 passed, 0 failed**.
- The two formerly uncollectable Task 13 files: **109 passed**, so the gate now
  does cover controller-authority issuance logic.
- `tests/test_glm52_task13_controller_key_restore.py`: **7 passed**.
- Full suite, no exclusions: **2 failed, 9055 passed, 108 skipped**. Both
  failures are pre-existing legacy GLM-4.5-Air assertions outside every named
  gate.
- Ruff `E4,E7,E9,F` over `src/glm52_enforcement` and `aws/glm52-gpu/scripts`:
  clean. Python 3.9 compilation of `src/glm52_enforcement`: pass.

Not covered by the above: an end-to-end issuance run using the **escrowed
production key** against a canonical package. Escrow restore, custody, and a
real SSHSIG through `ControllerAuthoritySigner.sign` are proven; the full
`issue_*` / `verify_*` chain is proven only with ephemeral keys. Closing that
gap requires a canonical package, which the archive-lineage rebuild below must
regenerate, so it is tracked there rather than satisfied with a synthetic
fixture.

## Archive lineage status

Open. The coordinator SHA is now
`259e490d2f02380c6a33b79608660d1b65eea56b08469772ce64f72d00ac4723`, so the
staged `s3://keep-glm52-models-246813579024-us-west-2/task13/archive/repo-tar.json`
and every package, approval, or rehearsal bound to the previous archive cannot
authorize this checkout. No new archive has been built and no S3 object has been
overwritten.

## AWS mutation performed by this rotation

One write, no campaign compute or controller changes: the Secrets Manager secret
recorded above. It costs roughly `$0.40` per month plus API calls, is encrypted
with the AWS-managed key rather than a customer master key, and carries the
ownership tags listed above. No recovery window was configured, so
`delete-secret` would apply the 30-day default; deleting it without a restored
copy destroys the only remaining copy of the v3 signing authority.
