# Task 10 Report — mount-free worker, reboot-safe deadline, and guarded production CLI

Date: 2026-07-29

## Status

`DONE`

Task 10 now provides a genuine guarded `production` route, one canonical
mount-free SkyPilot `0.13.0` task and one-wire request body, an authenticated
Task 8/9 worker-bootstrap wrapper, exact worker-side instance/tag readback,
three immutable systemd units, a fixed authenticated runtime-authority
entrypoint, persistent monotonic signed deadline state, synchronous
post-authentication systemd readiness, a durable hash-linked BOOTSTRAP ledger
barrier, exact-version initial S3 reads, and durable local-timer plus retained
SSM graceful-stop evidence.

Fix round 2 closes all five independently reported findings:

1. the campaign service is `Type=notify`, and READY is sent only after a real
   authenticated `CampaignController` is constructed; bootstrap authenticates
   H.1c ancestry plus exact unit, PID/argv, timer/drop-in, and next-edge
   readback;
2. signed T-60/T-50 progress is monotonic across wall-clock rollback and
   reboot, so STOP/start refusal never regress;
3. bootstrap cannot return until the controller's sole BOOTSTRAP ledger
   record, its distinct immutable record object, and the durable receipt have
   exact-version readback;
4. wrapper, archive, approval, intent, and H.1c reads use opaque VersionIds,
   expected owner `246813579024`, checksum proof, and one-attempt clients; and
5. both local timer and retained SSM paths publish or exactly adopt
   generation/allocation-scoped `WORKER_GRACEFUL_STOP.json`. The retained
   observer first exact-reads a durable closed Task 12 authority that binds
   the Task 9 allocation, action/audit identities, exact worker artifacts,
   command, instance, parameterless document, and numeric non-`$LATEST`
   document version before accepting matching `GetCommandInvocation`.

Fix round 3 closes the retained-provenance job-name fail-open found by the
second independent review. A worker observation must now name the exact
frozen production job `glm52-sky-20260724`, and runtime authority explicitly
cross-binds that name to the authenticated worker descriptor before Task 12
or retained SSM provenance can consume it.

No live AWS, Sky POST, EC2 launch, systemd mutation, SSM command, deployment,
termination, billing effect, model work, installed-Sky mutation, or Git
mutation occurred.

## Owned files

- `src/glm52_enforcement/task10_worker.py`
- `src/glm52_enforcement/task10_production.py`
- `src/glm52_enforcement/task10_durable_s3.py`
- `tests/test_glm52_task10_worker.py`
- `tests/test_glm52_task10_production_route.py`
- `aws/glm52-gpu/skypilot/production/keep-glm52-campaign.service`
- `aws/glm52-gpu/skypilot/production/keep-glm52-deadline.service`
- `aws/glm52-gpu/skypilot/production/keep-glm52-deadline.timer`
- `aws/glm52-gpu/skypilot/bootstrap_production_campaign.sh`
- `aws/glm52-gpu/scripts/authenticate_task10_worker_inputs.py`
- `aws/glm52-gpu/scripts/build_task10_production_task.py`
- `aws/glm52-gpu/scripts/collect_task10_ssm_graceful_stop.py`
- `aws/glm52-gpu/scripts/download_task10_runtime_inputs.py`
- `aws/glm52-gpu/scripts/glm52_deadline_guard.py`
- `aws/glm52-gpu/scripts/install_task10_worker_descriptors.py`
- `aws/glm52-gpu/scripts/materialize_task10_graceful_stop.py`
- `aws/glm52-gpu/scripts/materialize_task10_worker_observation.py`
- `aws/glm52-gpu/scripts/observe_task10_ssm_graceful_stop.py`
- `aws/glm52-gpu/scripts/publish_task10_bootstrap_ledger_receipt.py`
- `aws/glm52-gpu/scripts/publish_task10_bootstrap_ready.py`
- `aws/glm52-gpu/scripts/run_task10_production_campaign.py`
- `aws/glm52-gpu/scripts/wait_task10_bootstrap_ledger.py`
- `aws/glm52-gpu/cloudformation/KeepGlm52GracefulStopV1.json`
- `aws/glm52-gpu/cloudformation/worker-drain-signal-policy.json.in`
- `.superpowers/sdd/2026-07-27-glm52-full-campaign-run-handoff/task-10-report.md`

## Integration files

- `aws/glm52-gpu/scripts/submit_sky_campaign.py`
- `aws/glm52-gpu/scripts/submit_sky_campaign.sh`
- `aws/glm52-gpu/scripts/run_campaign.sh`
- `benchmarks/run_glm52_campaign.py`

## Implemented contracts

### Guarded production route

- The shell and Python entrypoints accept only production
  `validate-only`, `start`, and `reconcile`.
- Production rejects every qualification/cache-seed-only argument.
- The typed production authority binds the current consumed action,
  activation ordinal, generation, executor epoch, exact production job,
  execution deadline, exact GPU allocation identity, Task 8
  live/spend/reserve identities, Task 9
  launch/admission/custody identities, H100 resume authority, workflow
  version/execution, attestation/seal/relay identities, and exact task/body
  hashes.
- The request carries the typed worker-bootstrap descriptor. Before boundary
  inspection or start, it cross-binds wrapper file/body, campaign,
  activation id/ordinal, generation, action key, job, deadline, archive,
  approval, intent, GPU allocation, and Task 8/9 identities.
- The route exposes only injected `inspect`, `start_once`, and `reconcile`
  operations. Raw Sky, HTTP/POST, subprocess, RunInstances, SendCommand, and
  public `jobs_launch` methods are rejected.
- Validation has zero start/reserve effect. Start makes one injected
  `start_once` call. Ambiguous results are reconciliation-only and never
  trigger a second start.
- Without a separately deployed production boundary, the real CLI returns
  `deployment-required`; it does not fall back to a public Sky client.

### Mount-free task and worker authority

- The canonical task is exactly one AWS `us-west-2` on-demand
  `p5.48xlarge`, max `$55.04/hour`, `api_server_access: false`, no file
  mounts/workdir, bounded recovery, and no application retry.
- The production Sky job name is exactly `glm52-sky-20260724`, matching the
  accepted production-intent naming contract.
- Setup exact-reads the wrapper and repository archive by opaque VersionId,
  expected bucket owner, returned SHA-256 checksum, and one-attempt CLI
  policy, verifies file hashes, extracts only the authenticated archive, and
  invokes the tar-pinned bootstrap. The privileged handoff uses an explicit
  15-field `sudo env` projection; no setup authority is lost or inherited
  through a wildcard environment.
- The tar-pinned bootstrap exact-reads approval and intent into memory with
  opaque VersionIds, expected owner, returned checksum, HTTP status, and
  one-attempt SDK policy, then writes either both authenticated byte strings
  or neither.
- The wrapper is deliberately distinct from accepted H.1c descriptor v2. It
  exact-references that descriptor's URI, VersionId, file/body identity and
  campaign identity while binding the Task 8 deadline, exact lowercase GPU
  allocation SHA-256, spend authority, and Task 9 runtime coordinates.
- The wrapper contains no task-YAML or request-body hash. This breaks the
  otherwise impossible wrapper-hash/task-hash fixed-point cycle.
- On the worker, the closed task environment is reconstructed, the exact YAML
  and `JobsLaunchBody` hashes are recomputed, and those hashes must equal the
  Task 9 instance tags. IMDSv2 plus a one-attempt EC2 self-read binds the
  account, region, instance, activation, ordinals, generation, action, job,
  campaign, and allocation.
- The accepted H.1c descriptor is exact-read by its wrapper VersionId and is
  installed separately at `/etc/keep-glm52/campaign.json`.
- The production intent is file-hashed, canonical-JSON checked, delegated to
  the existing full production-intent validator, and compared to
  `GLM52_SUBMISSION_INTENT_BODY_SHA256` before systemd starts.

### Systemd, deadline, and evidence

- Exactly three units are installed. The campaign unit freezes `Type=notify`,
  `NotifyAccess=main`, `TimeoutStartSec=300`, `Restart=no`,
  `KillMode=control-group`, `TimeoutStopSec=1200`, `SendSIGKILL=no`, fixed
  paths, fixed pre-start/entrypoint, production mode, and the heavy-job lock.
- `run_campaign.sh` can invoke only the tar-pinned
  `run_task10_production_campaign.py`. That entrypoint accepts no arguments,
  validates canonical wrapper, accepted H.1c, and worker-observation files,
  authenticates the signed deadline state against the actual wrapper file
  hash and persistent 32-byte key, cross-binds instance/allocation/deadline,
  replaces all three runtime-authority variables with wrapper-derived values,
  and then execs only the fixed benchmark argv.
- The persistent deadline state is HMAC-signed and binds wrapper file
  identity, exact worker instance/allocation, T-60, T-50, and retained T-30
  wall-clock edges. The persistent timer receives immutable `OnCalendar`
  entries for T-60/T-50 and uses `Persistent=true`.
- Deadline evaluation takes the maximum of authenticated
  `last_completed_edge` and the observed wall-clock edge. A signed T-60 or
  T-50 completion therefore recreates STOP and refuses start even if the
  clock rolls backward.
- Before T-60, a stop marker is premature. At/after T-60, `/run` STOP is
  atomically reconstructed and new work is refused. At/after T-50, only
  `systemctl stop keep-glm52-campaign.service` is issued. T-30 never grants
  restart or force-kill.
- The production campaign installs a fixed SIGTERM handler before restore.
  Systemd SIGTERM becomes STOP plus status 75, so the existing final
  checkpoint-tree sync and resumable drain path runs; duplicate SIGTERM is
  ignored while draining.
- A successful local T-50 stop materializes canonical
  `WORKER_GRACEFUL_STOP.json` with exact unit/script hashes, actual systemd
  success fields, empty cgroup, stop/checkpoint/latest/terminal identities,
  and exact-null SSM command identity, then conditionally publishes or adopts
  its immutable generation/allocation object with exact VersionId, checksum,
  metadata, owner, and bytes before writing the local result.
- The parameterless SSM document emits the same post-stop canonical
  observation. Its retained observer accepts no arguments or environment
  coordinates; it exact-reads a fixed-key durable Task 12 worker-drain
  authority and cross-binds the embedded Task 9 worker observation, wrapper,
  exact unit/script hashes, instance, command, document name/version, role,
  action, and audit identities to a successful `GetCommandInvocation`.
  Only then may it publish/adopt SSM-authoritative evidence.
- Bootstrap authenticates H.1c and wrapper ancestry, unit bytes, the
  persistent state's HMAC and exact wrapper-file binding, `Type=notify`
  enabled/active state, exact MainPID/ExecMainPID and `/proc` argv, exact
  timer drop-in bytes/hash/two calendars/next T-60 elapse, descriptor/archive
  VersionIds and hashes, instance/allocation, resume roots, and
  marker-before-ledger ordering before publishing `BOOTSTRAP_READY`.
- The controller sends READY only after construction succeeds. It then owns
  the sole local BOOTSTRAP transition. A distinct immutable ledger-record
  object and a hash-linked `BOOTSTRAP_LEDGER.json` receipt are published and
  exact-read by VersionId; the bootstrap waiter independently reconstructs
  both and fails immediately if the controller dies before completion.
- The parameterless `KeepGlm52GracefulStopV1` document has no caller command,
  path, unit, signal, shell, or parameter. Its policy admits only the retained
  role's exact document/tagged-worker SendCommand surface and denies every
  SSM session operation.

## Literal RED to GREEN

Initial named RED before a production parser/route existed:

```text
$ uv run --offline pytest -q tests/test_glm52_task10_production_route.py::test_production_mode_has_genuine_guarded_validate_start_and_reconcile_route
FAILED tests/test_glm52_task10_production_route.py::test_production_mode_has_genuine_guarded_validate_start_and_reconcile_route
SystemExit: 64
1 failed
```

The parser rejected `production` as an invalid mode. After implementing the
real route:

```text
$ uv run --offline pytest -q tests/test_glm52_task10_production_route.py::test_production_mode_has_genuine_guarded_validate_start_and_reconcile_route
1 passed, 2 warnings in 0.95s
```

No-cycle executable worker proof:

```text
$ uv run --offline pytest -q tests/test_glm52_task10_worker.py::test_real_wrapper_breaks_hash_cycle_and_initializes_from_accepted_h1c
1 passed, 2 warnings in 0.56s
```

Fix-round-1 typed allocation RED:

```text
$ uv run --offline pytest -q tests/test_glm52_task10_production_route.py::test_validate_only_authenticates_but_cannot_reserve_or_post
FAILED tests/test_glm52_task10_production_route.py::test_validate_only_authenticates_but_cannot_reserve_or_post
TypeError: ProductionAuthority.__init__() got an unexpected keyword argument 'gpu_allocation_sha256'
1 failed in 0.04s
```

After extending and cross-binding both typed authorities:

```text
$ uv run --offline pytest -q tests/test_glm52_task10_production_route.py::test_validate_only_authenticates_but_cannot_reserve_or_post tests/test_glm52_task10_production_route.py::test_foreign_worker_wrapper_identity_fails_before_inspect_or_start
16 passed in 0.05s
```

The fixed runtime-entrypoint RED was the absent tar-pinned executable:

```text
$ uv run --offline pytest -q tests/test_glm52_task10_worker.py::test_fixed_runtime_entrypoint_derives_authority_before_real_controller_init
FAILED tests/test_glm52_task10_worker.py::test_fixed_runtime_entrypoint_derives_authority_before_real_controller_init
FileNotFoundError: aws/glm52-gpu/scripts/run_task10_production_campaign.py
1 failed, 2 warnings in 0.74s
```

The publisher-authentication RED proved both unsafe paths reached publication:

```text
$ uv run --offline pytest -q tests/test_glm52_task10_worker.py::test_bootstrap_publisher_authenticates_deadline_state_and_wrapper_before_publish
FAILED ...[corrupt-signature] - Failed: DID NOT RAISE SystemExit
FAILED ...[foreign-wrapper] - Failed: DID NOT RAISE SystemExit
2 failed, 2 warnings in 0.61s
```

Runtime and publisher GREEN:

```text
$ uv run --offline pytest -q tests/test_glm52_task10_worker.py::test_fixed_runtime_entrypoint_derives_authority_before_real_controller_init tests/test_glm52_task10_worker.py::test_fixed_runtime_entrypoint_rejects_mutated_authority_before_campaign_work
6 passed, 2 warnings in 0.60s

$ uv run --offline pytest -q tests/test_glm52_task10_worker.py::test_bootstrap_publisher_authenticates_deadline_state_and_wrapper_before_publish tests/test_glm52_task10_worker.py::test_graceful_stop_requires_checkpoint_service_and_empty_cgroup_evidence tests/test_glm52_task10_worker.py::test_local_timer_materializes_canonical_graceful_stop_evidence
4 passed, 2 warnings in 0.57s
```

Fix-round-2 review-selector RED, captured before implementation:

```text
$ uv run --offline pytest -q tests/test_glm52_task10_worker.py -k fix2
7 failed, 2 passed in 0.81s
```

Those failures proved premature `Type=simple` readiness, signed-edge
rollback, marker-before-ledger completion, inexact/default-retry S3 reads,
and absent durable local/SSM graceful publication. The final expanded Fix 2
matrix covers the five findings plus privileged environment handoff, exact
fake-client read/write/adoption, Task 12/Task 9 provenance, numeric document
version, artifact-hash drift, and zero-put failure cases:

```text
$ uv run --offline pytest -q tests/test_glm52_task10_worker.py tests/test_glm52_task10_production_route.py -k fix2
18 passed, 74 deselected, 2 warnings in 0.76s
```

Fix-round-3 literal RED proved that a canonical re-self-hashed observation
could name a foreign Sky job and reach the retained observer's durable read:

```text
$ uv run --offline pytest -q tests/test_glm52_task10_worker.py -k fix3
1 failed, 1 passed, 60 deselected, 2 warnings in 0.87s
```

The GREEN matrix covers the exact accepted name, syntactically valid foreign
name, wrong-case name, missing name, and a fully re-self-hashed Task 12/SSM
handoff that must fail before any durable get or graceful-evidence put:

```text
$ uv run --offline pytest -q tests/test_glm52_task10_worker.py -k fix3
2 passed, 60 deselected, 2 warnings in 0.58s
```

Final focused result:

```text
$ uv run --offline pytest -q tests/test_glm52_task10_worker.py tests/test_glm52_task10_production_route.py
94 passed, 2 warnings in 2.07s
```

## Verification

All commands were local/offline or used injected fakes.

| Gate | Result |
|---|---:|
| Task 10 focused | 94 passed |
| SkyPilot task + Sky campaign + submission integration | 333 passed |
| Production submission + production fence/audit + terminal state | 480 passed |
| Enforcement aggregate + H.1g migration + frozen CloudFormation | 1,085 passed |
| Campaign runtime descriptor + pinned rehearsal regressions | 46 passed |
| Total distinct pytest cases | 2,038 passed, 0 failed |

Exact frozen aggregate commands:

```text
uv run --offline pytest -q tests/test_glm52_skypilot_task.py tests/test_glm52_sky_campaign.py tests/test_glm52_sky_submission_integration.py
uv run --offline pytest -q tests/test_glm52_sky_production_submission.py tests/test_glm52_sky_production_fence.py tests/test_glm52_sky_production_fence_audit.py tests/test_glm52_sky_terminal_state.py
uv run --offline pytest -q tests/test_glm52_enforcement_*.py tests/test_glm52_h1g_stack_migration.py tests/test_glm52_sky_cloudformation.py
uv run --offline pytest -q tests/test_glm52_campaign_runtime_descriptor.py tests/test_glm52_sky_pinned_rehearsal.py
```

Additional gates:

```text
Python 3.9 py_compile: passed
Python 3.12 py_compile: passed
Python 3.9 import-light with mlx/numpy blocked: passed
Python 3.12 import-light with mlx/numpy blocked: passed
Pinned SkyPilot 0.13.0 parser on the production task: passed
uvx --offline ruff check --select E4,E7,E9,F: All checks passed!
bash -n and zsh -n: passed
Executable-mode checks: passed
Explicit owned/integration-file trailing-whitespace/final-newline scan: passed
Checked-in unit/document/policy bytes equal pure renderers: passed
Canonical/non-overwrite output tests: passed
```

The ordinary SWIG `SwigPyPacked`, `SwigPyObject`, and `swigvarlink`
deprecation warnings were the only pytest warnings.

`git diff --check` was not run because the coordinating task imposed a
no-Git-command fence. The explicit owned/integration-file whitespace and
final-newline scan above was used instead.

## Frozen artifact SHA-256

All 28 non-self-referential owned/integration artifacts are represented. The
report itself is intentionally not self-hashed.

```text
72fc72d91d35f15d4f64c608aa1ba8b77e5c72732ac5931651d00adf65ccc8db  src/glm52_enforcement/task10_worker.py
b6ee468bfe597d09d7883367dee3e8c3712efc6791a321477aff1d1c65b12bff  src/glm52_enforcement/task10_production.py
1077083196349a19004bdb4f54c0eccd70b96b1b122df9d045715d5d1c4427a3  src/glm52_enforcement/task10_durable_s3.py
4a9eed390ccded7ad804b6917f90fe4658d5b79baf7330d06f32e565880d9001  aws/glm52-gpu/skypilot/production/keep-glm52-campaign.service
ce70b7499f3e62c52cb914acac8db6b01f554499af71efc88a826cd13a60ed2e  aws/glm52-gpu/skypilot/production/keep-glm52-deadline.service
65cceb5a95092b9122e15156958963b036fc706497c52f441e4501f7c11bfdcc  aws/glm52-gpu/skypilot/production/keep-glm52-deadline.timer
7a172768bf4feac8e132082b3f5ef94f90d834bb99caab9ab2c6721b92273ac0  aws/glm52-gpu/cloudformation/KeepGlm52GracefulStopV1.json
4f98e64324765a651b8c33e3e7ef532aa48468535867511e32c0d531e26ae7fc  aws/glm52-gpu/cloudformation/worker-drain-signal-policy.json.in
69fdb9700fad2d7eb0fd2074be3340ae266f8be67d150ec503eca3147b0fa07f  aws/glm52-gpu/skypilot/bootstrap_production_campaign.sh
c9c4d74e5ee6e5a0e7527946496a7f4f34cd0a899c6341b1476a762d89e83da1  aws/glm52-gpu/scripts/authenticate_task10_worker_inputs.py
7d79662cf5dea95493d7970d6305a02d94589f44eeb6193f7c73196ddc055e74  aws/glm52-gpu/scripts/build_task10_production_task.py
1178476f21d541114e544d3d70b4d6c87971c059d52acd95c5459dbf81324caf  aws/glm52-gpu/scripts/collect_task10_ssm_graceful_stop.py
9a9d7cd74898cb31102988846b677bbaed03ad79ebe11f749a3a580bba1eb485  aws/glm52-gpu/scripts/download_task10_runtime_inputs.py
97dd1f5019714a718d8c3f71d6fb72c498b27fc0e15499d6f936ab3348a9de3f  aws/glm52-gpu/scripts/glm52_deadline_guard.py
acfce4dd31016f0a1eb1f2a30cb534bf58198730f5b4e264050ef34533e1a1b9  aws/glm52-gpu/scripts/install_task10_worker_descriptors.py
87aea23871a5054ab4cdc710f1e63e7b56a6db46f014ba9503b9da78ec62c227  aws/glm52-gpu/scripts/materialize_task10_graceful_stop.py
f9e5604605d99c0f51900fd7e8019e5fbc2fb6f584d5f75003adcd1b1c8e9aef  aws/glm52-gpu/scripts/materialize_task10_worker_observation.py
c4f5d68baaee037d199727c50449281ef234f6a366e5f0210d2d535fdc7ebd0a  aws/glm52-gpu/scripts/observe_task10_ssm_graceful_stop.py
cc16b42d09fd2c07b173d2193b108cef33f4e3c597e19952c0dd2c27f168493f  aws/glm52-gpu/scripts/publish_task10_bootstrap_ledger_receipt.py
abd8e5aebd205105542557b28a2025af5ed97feb8d59137df2d76c2b1d59b504  aws/glm52-gpu/scripts/publish_task10_bootstrap_ready.py
b928e6c13f258a72c1bfc2747f35c8ed7b481897776ee5440b332f76bc6cfe5b  aws/glm52-gpu/scripts/run_task10_production_campaign.py
2b70a94dacb22d84c9fc04ec0c277543ca4ce391148794ab9a379c7a7341dc66  aws/glm52-gpu/scripts/wait_task10_bootstrap_ledger.py
575c2c7fdb43a9053930e9f625ff7334dc20924cc9d34c65cc254a51bf4ae8e4  aws/glm52-gpu/scripts/run_campaign.sh
33f3f1707448353f7493e1dda92ed732883fe479a1b4e39730d7598c455bf080  aws/glm52-gpu/scripts/submit_sky_campaign.py
c91121014c9389e7bc900e390e294e05b186f405f3d6047acb51800bbe182534  aws/glm52-gpu/scripts/submit_sky_campaign.sh
c75408649dc72ddb5b79752d84214bbccc5099e1f01b7422c3d516cda0e1df9c  benchmarks/run_glm52_campaign.py
6310df7bd8ca756e9c345820bf5e05ceb0d98565f24e4b992fa1075791c2e39c  tests/test_glm52_task10_worker.py
9efb2699c7b0d0dc06b7b99e1fffcca581449ddd14a14e078e5ced1f39ce1e13  tests/test_glm52_task10_production_route.py
```

## Self-review

- Verified the accepted H.1c descriptor is not incorrectly treated as a
  source for Task 8 deadline or Task 9 instance/allocation fields.
- Verified wrapper serialization has no task/body hash cycle and that a real
  wrapper file hash can be embedded into a task whose recomputed hashes match
  Task 9 tags.
- Verified every overlapping wrapper/authority coordinate, including
  activation ordinal, action key, deadline, GPU allocation, and exact
  production job, fails before injected inspection/start when foreign.
- Verified an exact-name observation is accepted, while re-self-hashed
  foreign, wrong-case, and missing worker job names fail before runtime
  authority; the retained Task 12/SSM form fails before durable get or put.
- Verified the production runtime with all three controller-authority
  variables absent: only the exact authenticated wrapper values reached a
  real `CampaignController.__init__`. Missing or mutated allocation,
  deadline, spend, or wrapper ancestry failed before campaign work.
- Verified corrupt deadline signatures and rehashed foreign wrappers produce
  zero bootstrap-publish calls and no local `BOOTSTRAP_READY` marker.
- Verified an authenticated completed T-60/T-50 edge never regresses across
  wall-clock rollback; STOP is recreated and campaign start remains refused.
- Verified invalid/foreign H.1c, wrong process argv/PID, wrong unit mode,
  wrong timer next edge, or changed drop-in bytes fail before bootstrap
  publication.
- Verified controller death after marker publication but before BOOTSTRAP
  transition leaves no durable ledger completion and makes the bootstrap
  waiter fail.
- Verified every initial S3 get carries exact owner, opaque VersionId,
  checksum mode, returned checksum verification, and one-attempt retry
  configuration.
- Verified intent body identity was not merely present in the environment:
  worker bootstrap parses and fully validates the canonical intent.
- Verified systemd SIGTERM reaches status-75 checkpoint/sync/drain rather than
  default-killing the Python controller.
- Verified successful local drain evidence uses the actual systemd
  `ExecMainCode=1`, `ExecMainStatus=0` success shape and refuses missing
  checkpoint, live cgroup, or failed service evidence.
- Verified local graceful evidence publishes once and exactly adopts an
  identical existing version; missing checkpoint, active service, or
  nonempty cgroup makes zero put calls.
- Verified the retained SSM observer rejects a merely self-hashed but
  unpublished Task 12 authority, `$LATEST`/wrong document version, foreign
  Task 9 ancestry, and changed worker artifact hashes before any graceful
  evidence put.
- Verified no public Sky retrying client, raw HTTP, EC2 launch script, caller
  YAML/path/command, automatic SIGKILL, or custom MLX wired-memory override
  was added to the production path.

## Limitations and handoff

- The exact production workflow boundary is injectable but not deployed in
  this task. Default CLI execution remains honestly `deployment-required`.
- AWS/S3/EC2/IMDS, systemd, SSM, and Sky effects were exercised only through
  pure contracts, monkeypatched adapters, deterministic simulators, or the
  installed local SkyPilot parser. None is live/deployed truth.
- Task 10 implements exact local and SSM graceful evidence
  publication/adoption plus the retained observer contract. Task 12 must
  durably publish the typed retained worker-drain authority, invoke/deploy the
  observer with its fixed handoff, and own terminal-v2, cancellation,
  controller-quiesced forced-remnant handling, and final drain.
- `BOOTSTRAP_READY` publication and later BOOTSTRAP ledger consumption are
  implemented and locally validated, but no marker was written to live S3.
- No worker was rebooted and no real child process was held beyond 1,200
  seconds; those paths use deterministic state/systemd simulations here and
  require deployed proof in the later campaign gate.

## Independent approval

Independent read-only review approved the final Task 10 implementation with no
Critical or Important findings. The final review verified the exact frozen
Sky job-name cross-binding for accepted, foreign, wrong-case, missing, and
retained-SSM provenance cases; `94/94` focused tests; `2,038/2,038` aggregate
checks; `28/28` artifact hashes; Ruff; Python 3.9/3.12 compilation; and clean
whitespace/diff checks.
