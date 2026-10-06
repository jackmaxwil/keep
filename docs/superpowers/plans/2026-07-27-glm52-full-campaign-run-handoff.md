# GLM-5.2 Full Campaign Run Handoff and Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to execute this handoff task by task. Use
> checkbox (`- [ ]`) status here as the durable operator ledger. Do not use
> parallel workers for live cloud mutations.

**Goal:** Take the current KEEP checkout from its closed, zero-GPU Job 3 state
through one authenticated qualification-cache seed, real cross-node NVIDIA
H100 recovery qualification, full teacher-cache generation, guarded adapter
training and evaluation, authenticated drain, cloud teardown, cost
reconciliation, and a manual promotion or no-promotion decision.

**Architecture:** SkyPilot owns on-demand worker provisioning, replacement,
and normal worker teardown in Amazon Web Services account `246813579024`,
region `us-west-2`. KEEP remains authoritative for immutable campaign
identity, cumulative spend, checkpoints, cache integrity, phase transitions,
training gates, terminal evidence, and orphan detection. The owner has stopped
further architecture-hardening iterations: candidate thirteen is the
implementation basis, but live work remains fail-closed until its executable
enforcement, local verification, artifact audit, approvals, and real hardware
qualification exist.

**2026-07-30 owner decision — superseding SCP clauses:** Do not create,
attach, enumerate, or prove any bootstrap, baseline, or maintenance-seal AWS
Organizations Service Control Policy (SCP). Use existing member-account
permissions and member-account lifecycle controls instead. This supersedes the
candidate-v13 SCP clauses; it retains fresh STS proof for account
`246813579024`, exact `DescribeOrganization` identity, region
`us-west-2`, immutable reviewed artifact/version/checksum, approvals and the
spend ledger, the one-worker limit, on-demand `p5.48xlarge` only,
watchdog/termination/reconciliation, and the prohibition on Spot, Capacity
Blocks, and raw launch. This is an owner decision for future execution
requirements, not an assertion of current deployed truth.

**Tech Stack:** Python 3.9 and 3.12 compatibility, `pytest`, CloudFormation,
DynamoDB, Amazon Simple Storage Service, Lambda, Step Functions, EventBridge,
Systems Manager, SkyPilot `0.13.0`, on-demand `p5.48xlarge`, eight NVIDIA H100
accelerators, CUDA, systemd, and the existing KEEP teacher/cache/training
modules.

## Global Constraints

- Work only in `/Users/jack.mazac/Developer/keep`.
- Preserve the dirty feature worktree. Do not reset, clean, switch branches,
  commit, push, merge, publish, or remove protected `runs/` or root handoff
  files without a new explicit instruction.
- Every Amazon Web Services command uses the explicit profile
  `keep-gpu`; every paid or mutable command first proves account
  `246813579024`.
- Region is exactly `us-west-2`.
- Provision at most one active `p5.48xlarge`; use on-demand capacity only.
- Do not purchase a Capacity Block. Do not use Spot. Do not use another
  region, cloud, instance type, or raw manual launch path.
- The existing graphics-processing-unit approval is twenty-four cumulative
  hours and `$1,320.96` at no more than `$55.04/hour`. Recovery instances and
  qualification consume the same balance. Resubmission never resets it.
- No production worker may use the dirty checkout. Every paid worker uses one
  exact immutable repository archive that passed clean-room rehearsal and
  direct Amazon Simple Storage Service audit.
- Keep `GLM_MLX_WIRED_LIMIT_GB` and
  `GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB` unset.
- No model is automatically promoted, published, uploaded, or installed as
  the accepted baseline.
- Cost Explorer, budgets, and anomaly detection are delayed observations.
  The authenticated append-only spend ledger is the immediate authority.
- Pending SkyPilot time does not consume graphics-processing-unit approval.
  An active Amazon Elastic Compute Cloud worker does.

---

## 1. Owner Decision and Exact Handoff Snapshot

The owner instruction on 2026-07-27 is:

> Stop. We have hardened enough. Write handoff for full run.

This closes further candidate-thirteen architecture-review iteration as a
prerequisite. Do not spend another session seeking three architecture
`READY` verdicts. Reopen the architecture only if implementation or testing
finds a concrete contradiction that prevents a safe executable contract.

The exact architecture snapshot accepted as the implementation basis is:

```text
path:
.superpowers/sdd/goal-objective/task-3ph1g-production-enforcement-architecture.md
SHA-256:
1daaa963d42e5fb94ed712c07d83c6b43f16fbcc9e89b0df0c7721587d8c051e
Markdown fences:
94, balanced
status inside document:
REVISED DESIGN CANDIDATE v13 — REVIEW REQUIRED; NOT AN IMPLEMENTATION BRIEF
```

The status line is historically accurate: candidate thirteen did not receive
fresh independent `READY` reviews. The owner decision permits moving forward
from these bytes; it does not transform unimplemented design into deployed
protection.

The current worktree snapshot has nine tracked changed paths and 149
untracked paths. Treat all of them as user-owned. Record a fresh
`git status --short --branch` before touching any file, but do not normalize
the tree.

The earlier cache-seed SkyPilot Job 3 is closed, not pending:

- Immutable cancellation evidence authenticates.
- No `p5.48xlarge` existed at the read-only closeout.
- No graphics-processing-unit allocation or spend-ledger interval existed.
- The old controller `i-0511af4e31aa5406a` was stopped.
- Its attached 50-GiB root volume
  `vol-04d1a5b0f39076220` may still incur storage cost.
- Direct SkyPilot queue proof of `CANCELLED` was unavailable while the
  controller was stopped.

Those facts are from the 2026-07-27 read-only report and are not current cloud
authority. Refresh them before any mutation.

The present executable boundary is not production-ready:

- `aws/glm52-gpu/scripts/submit_sky_campaign.sh` accepts only
  `--qualification`.
- `submit_sky_campaign.py` has only `qualification` and `cache-seed` modes.
- The default live CloudFormation inspector intentionally returns no success
  path and reports deployment required.
- The thirteenth enforcement architecture is not implemented.
- The cancelled seed produced no accepted qualification-cache marker.
- No authenticated `H100_RESUME_READY.json` exists.
- No full 257-session schema-version-three teacher cache exists.
- No current campaign adapter candidate exists.
- No `GPU_RESIDUAL_LIABILITY_APPROVAL.json` exists in this checkout.

These are mandatory implementation or approval gates, not documentation
warnings.

## 2. Authoritative Inputs

Read these in this order before implementation or live work:

1. Full campaign objective:
   `/Users/jack.mazac/.codex/attachments/ec77be41-4865-4b70-bab6-db421670d080/goal-objective.md`
2. This handoff:
   `docs/superpowers/plans/2026-07-27-glm52-full-campaign-run-handoff.md`
3. Candidate-thirteen architecture:
   `.superpowers/sdd/goal-objective/task-3ph1g-production-enforcement-architecture.md`
4. Accepted production-generation authority:
   `.superpowers/sdd/goal-objective/task-3ph1e-production-generation-authority-report.md`
5. Accepted production-fence audit:
   `.superpowers/sdd/goal-objective/task-3ph1f-production-fence-audit-report.md`
6. Historical Job 3 closeout:
   `.superpowers/sdd/goal-objective/job3-closeout-readonly-report.md`
7. Amazon Web Services and artifact runbook:
   `aws/glm52-gpu/README.md`
8. SkyPilot recovery runbook:
   `aws/glm52-gpu/SKYPILOT_BREAK_GLASS.md`
9. Account isolation:
   `aws/glm52-gpu/AWS_ACCOUNT_PROFILES.md`
10. Scientific campaign state machine:
    `benchmarks/run_glm52_campaign.py`

When documents disagree, use this priority:

1. A fresh direct cloud read for current resource state.
2. Exact immutable, version-pinned object bytes and their strict validator.
3. The current candidate-thirteen architecture snapshot.
4. Accepted H.1e and H.1f contracts.
5. This execution handoff.
6. Historical reports and the old objective opening.

The objective's opening statement that Job 3 is pending is historical.

## 3. Definition of Done

The full run is done only when all of the following are proven:

- [ ] The exact campaign account, region, market, instance type, spend
  approval, support-cost approval, and residual-liability approval
  authenticate.
- [ ] The candidate-thirteen enforcement plane is implemented and its local
  test and rehearsal gates pass.
- [ ] The exact repository archive and every model/campaign object pass
  version, size, full-object checksum, and tensor-range audit.
- [ ] The one-row qualification cache authenticates.
- [ ] A source H100 worker is intentionally terminated and a different H100
  worker resumes its exact checkpoint successfully.
- [ ] The real two-step training smoke stays below 70 GiB peak device memory.
- [ ] `H100_RESUME_READY.json` authenticates.
- [ ] Production generates exactly 257 teacher sessions and exactly
  2,500,735 supervised positions, or drains with an authenticated resumable
  deadline before completion.
- [ ] The schema-version-three cache audit passes and
  `TEACHER_CACHE_READY.json` authenticates before training.
- [ ] Training either completes under its gates or publishes an authenticated
  `TRAINING_DEFERRED.json`.
- [ ] Evaluation completes without automatic promotion.
- [ ] `CAMPAIGN_DRAINED.json`, the final campaign ledger, the final spend
  ledger, and `TERMINAL_VERIFIED.json` authenticate.
- [ ] No campaign P5 worker, unintended Elastic Block Store volume, Elastic
  IP address, temporary bucket, or unintended controller remains billable.
- [ ] The final cost report accounts for every allocation and retained
  control/storage resource.
- [ ] The owner records a manual reject, further-experiment, or promote
  decision.

## 4. Phase Zero — Re-establish Current Authority

- [ ] **Read the full objective and all authoritative inputs.**

  Do not rely on a condensed conversation summary.

- [x] **Capture the local snapshot without changing it.**

  ```bash
  cd /Users/jack.mazac/Developer/keep
  git status --short --branch
  shasum -a 256 \
    .superpowers/sdd/goal-objective/task-3ph1g-production-enforcement-architecture.md
  ```

  The architecture digest must be
  `1daaa963d42e5fb94ed712c07d83c6b43f16fbcc9e89b0df0c7721587d8c051e`.
  If it differs, stop and explain the drift before using it as authority.

- [x] **Authenticate the Research and Development account.**

  ```bash
  cd /Users/jack.mazac/Developer/keep
  aws login --profile keep-gpu
  export AWS_PROFILE=keep-gpu
  export AWS_REGION=us-west-2
  export AWS_DEFAULT_REGION=us-west-2
  aws/glm52-gpu/scripts/assert_rnd_aws_account.sh
  aws sts get-caller-identity \
    --profile keep-gpu \
    --region us-west-2 \
    --query Account \
    --output text
  ```

  Both guards must print `246813579024`. A different account is an immediate
  stop.

- [x] **Perform a read-only cloud reconciliation.**

  ```bash
  aws ec2 describe-instances \
    --profile keep-gpu \
    --region us-west-2 \
    --filters \
      'Name=tag:project,Values=keep-glm52' \
      'Name=instance-state-name,Values=pending,running,stopping,stopped' \
    --query 'Reservations[].Instances[].{Id:InstanceId,Type:InstanceType,State:State.Name,Run:Tags[?Key==`campaign-run-id`]|[0].Value}' \
    --output json

  aws ec2 describe-volumes \
    --profile keep-gpu \
    --region us-west-2 \
    --filters 'Name=tag:project,Values=keep-glm52' \
    --query 'Volumes[].{Id:VolumeId,State:State,Size:Size,Attachments:Attachments}' \
    --output json

  aws ec2 describe-addresses \
    --profile keep-gpu \
    --region us-west-2 \
    --filters 'Name=tag:project,Values=keep-glm52' \
    --output json

  aws cloudformation describe-stacks \
    --profile keep-gpu \
    --region us-west-2 \
    --stack-name keep-glm52-gpu \
    --output json
  ```

  Record exact output and timestamps. Do not start the stopped controller to
  improve a historical queue proof. Do not delete its volume as part of this
  read-only phase.

- [x] **Inventory immutable campaign objects without writing.**

  ```bash
  aws s3api list-objects-v2 \
    --profile keep-gpu \
    --region us-west-2 \
    --bucket keep-glm52-models-246813579024-us-west-2 \
    --prefix campaigns/glm52-sky-20260724/ \
    --output json
  ```

  Preserve the result as evidence. Filename presence is not authentication.

## 5. Phase One — Implement the Production Enforcement Plane

The owner has ended architecture iteration, not implementation. Complete this
phase before any new worker can be launched.

### 5.1 Pure package and durable records

- [ ] Create a standalone import-light `glm52_enforcement` package that does
  not import `mlx_vq.quality.__init__`.
- [ ] Implement the exact activation index, rollover, execution, action,
  recovery, finalization, snapshot-cleanup, worker-launch, launch-liability,
  liability-action, liability-settlement, post-terminal-allocation, operator
  disposition, and terminal-version-two schemas from candidate thirteen.
- [ ] Implement canonical encoding and hash-chain verification.
- [ ] Implement invocation-nonce ownership, conditional state transitions,
  exact duplicate adoption, coherent readback, and owner takeover.
- [ ] Add the no-launch
  `ACTIVATION_ROLLOVER_DEPLOYMENT_FAILED_NO_LAUNCH` terminal outcome.
- [ ] Implement the immutable snapshot-cleanup transition chain and
  rollover-time atomic lineage condition checks.
- [ ] Implement the nominal 900-second graphics-processing-unit reserve,
  separate root-volume tail, and residual-liability approval validator.

The first test run must fail because these interfaces do not exist. The
subsequent run must pass without importing model backends.

### 5.2 CloudFormation ownership and network

- [ ] Split the stack into retained foundation, parameterless production
  fence, and ephemeral support ownership.
- [ ] Make CloudFormation own one host-egress private subnet, two isolated
  private subnets, routes, one network address translation gateway, one
  Elastic IP address, gateway endpoints, the exact Secrets Manager endpoint,
  security groups, roles, logs, alarms, dead-letter queue, and retained
  ledger resources.
- [ ] Make the production bucket-policy fence unconditional across every
  legal parameter combination.
- [ ] Constrain both the CloudFormation executor and the service role.
- [ ] Add explicit denies for Spot, Capacity Blocks, alternate regions,
  alternate worker shapes, raw controller launch, and unapproved role
  passing.
- [ ] Enforce policy-size, resource-count, event-history, invocation, log,
  network-byte, and lifecycle budgets.

### 5.3 Launch and liability custody

- [ ] Patch the pinned SkyPilot provisioner so it writes only a durable launch
  intent. It must never call `RunInstances`.
- [ ] Make the exact retained same-token completer the sole sender for attempt
  one and every later same-token call.
- [ ] Require one liability action and shared atomic attempt counter before
  each call.
- [ ] Cap same-token completion at six zero-library-retry calls in six
  minutes.
- [ ] Keep the combined controller host denied `RunInstances`,
  `TerminateInstances`, `StartInstances`, and `PassRole`.
- [ ] Implement exact worker shape enforcement: approved image, one
  on-demand `p5.48xlarge`, required tags, metadata service version two,
  encrypted delete-on-termination 300-GiB gp3 root at 3,000 input/output
  operations per second and 125 MiB/s, and no Elastic Block Store data disk.
- [ ] Implement one-minute liability observation as a service-level
  objective, not a hard cloud guarantee.
- [ ] Keep exact-instance termination authority alive until the settlement
  record authenticates, including after campaign authorization exhaustion.

### 5.4 SkyPilot identity and one-wire submission

- [ ] Pin SkyPilot `0.13.0` and the exact patched source hash.
- [ ] Implement authenticated API-server identity, role-based access control,
  closed loopback relay, launch admission, and effective-consolidation probes.
- [ ] Retain one bounded combined host role and eliminate the inert
  `c6a.xlarge` controller pin contradiction.
- [ ] Make exactly one action-consumed production submission possible per
  activation.
- [ ] Add production mode to the guarded submission entrypoint; do not expose
  raw `sky jobs launch` as an operator path.
- [ ] Keep cache-seed, qualification, and production as distinct immutable
  submissions under one cumulative spend authority.

### 5.5 Deadlines, terminal evidence, and teardown

- [ ] Add distinct phase names and explicit Step Functions Task timeouts.
- [ ] Implement the measured decision-closure budget and its twenty-run test,
  including five cold starts.
- [ ] Implement runtime observation, exact numeric binding, request and worker
  cardinality, cancellation, controller quiescence, worker drain, allocation
  closure, terminal-version-two publication, and retained finalization.
- [ ] Implement reboot-safe `T−60m`, `T−50m`, and `T−30m` behavior with
  `SendSIGKILL=no`.
- [ ] Implement post-drain forensic snapshot cleanup and post-rollover
  authority.
- [ ] Implement final orphan and retained-cost audit.

### 5.6 Implementation acceptance

- [ ] Run the exact twenty-five deferred transport rows and twenty-two
  transport mutants.
- [ ] Add explicit tests for:
  - original provisioner launch unreachability;
  - first-call liability action consumption;
  - ambiguous same-token response readback;
  - writer death after committed rollover;
  - exact duplicate CloudFormation callback adoption;
  - rollback-to-no-launch terminal closure;
  - cleanup versus rollover atomic race;
  - three-activation cleanup lineage;
  - exact settlement creation and lost-response adoption;
  - delayed or multiple instances outside the nominal reserve;
  - missing residual-liability approval;
  - complete post-rollover cleanup permissions;
  - the honest DynamoDB sort-key boundary;
  - support-plane price-model completeness.

Do not claim Phase One complete merely because pure model tests pass. The
actual adapters, policies, workflow definitions, and disabled-stack
rehearsals must be covered.

## 6. Phase Two — Local Verification and Exact Archive

- [ ] Run focused campaign tests:

  ```bash
  cd /Users/jack.mazac/Developer/keep
  python3 -m pytest -q \
    tests/test_glm52_sky_campaign.py \
    tests/test_glm52_sky_submission_integration.py \
    tests/test_glm52_sky_production_fence.py \
    tests/test_glm52_sky_production_fence_audit.py \
    tests/test_glm52_sky_cloudformation.py \
    tests/test_glm52_sky_terminal_state.py \
    tests/test_glm52_h100_qualification.py \
    tests/test_glm52_teich_checkpoint.py \
    tests/test_glm52_teich_training_cache.py \
    tests/test_glm52_teich_training_campaign.py
  ```

- [ ] Run every new enforcement test and confirm the exact expected count.
- [ ] Run Python 3.9 compilation for the standalone enforcement package.
- [ ] Run Python 3.12 Amazon Linux target-import proof.
- [ ] Run Ruff selected `E4,E7,E9,F`.
- [ ] Run shell syntax checks over every campaign shell script.
- [ ] Run CloudFormation lint and policy denial tests.
- [ ] Run `git diff --check`.
- [ ] Confirm all preexisting accepted H.1e and H.1f identity sentinels remain
  exact, or document and re-review any deliberately replaced dependency.

Build the repository archive once. Never repack between seed, qualification,
and production:

```bash
cd /Users/jack.mazac/Developer/keep
export FULL_RUN_WORK
FULL_RUN_WORK=$(mktemp -d /tmp/glm52-full-run-20260727.XXXXXX)
printf '%s\n' "$FULL_RUN_WORK" > /tmp/glm52-full-run-current
chmod 600 /tmp/glm52-full-run-current
aws/glm52-gpu/scripts/build_sky_repository_tar.sh \
  "$FULL_RUN_WORK/repo.tar.gz"
shasum -a 256 "$FULL_RUN_WORK/repo.tar.gz" \
  > "$FULL_RUN_WORK/repo.tar.gz.sha256"
```

The removal above is restricted to the exact dedicated temporary directory.
Do not point `FULL_RUN_WORK` at the repository, `runs/`, or an authority
source.

## 7. Phase Three — Approval Closure

Before any possible worker send, authenticate all three separate authorities:

1. Existing graphics-processing-unit approval:
   - approver: Alex Approver;
   - one on-demand `p5.48xlarge`;
   - `us-west-2`;
   - `$55.04/hour`;
   - twenty-four cumulative hours;
   - `$1,320.96`;
   - qualification and recovery included.
2. Production support-plane approval:
   - exact resources and cardinalities;
   - 68-hour work, 71-hour delete request, 72-hour incident threshold;
   - one network address translation gateway and Elastic IP address;
   - exact logs, invocations, workflow history, secrets, endpoint, storage,
     and seven-day forensic snapshot;
   - dated price-model identity.
3. Residual launch-liability approval:
   - nominal 900 graphics-processing-unit seconds and `$13.76` deducted from
     the existing graphics-processing-unit envelope;
   - separately priced 300-GiB root-volume tail at most `$0.01`;
   - delayed request visibility, delayed control-plane calls, delayed
     termination, and unexpected same-token multiplicity;
   - termination-only continuation until positive settlement;
   - explicit acknowledgment that Amazon Web Services supplies no hard
     post-acceptance billing cap.

If either the support-plane or residual-liability approval is absent,
continue local tests and artifact audit but stop before `POSSIBLY_SENT`.

## 8. Phase Four — Artifact Inventory and Clean Rehearsal

Use:

```bash
export AWS_PROFILE=keep-gpu
export AWS_REGION=us-west-2
export RUN_ID=glm52-sky-20260724
export BUCKET=keep-glm52-models-246813579024-us-west-2
export FULL_RUN_WORK
FULL_RUN_WORK=$(cat /tmp/glm52-full-run-current)
test -d "$FULL_RUN_WORK"
mkdir -p \
  "$FULL_RUN_WORK/cache-seed" \
  "$FULL_RUN_WORK/post-seed" \
  "$FULL_RUN_WORK/production"
```

The campaign descriptor and both qualification driver manifests are mutable
between attempts but immutable within one activation. They MUST use these
activation-scoped keys; the former global keys are invalid inputs:

| Artifact kind | Exact key |
| --- | --- |
| `PRODUCTION_DESCRIPTOR` | `task13/activations/$ACTIVATION_ID/inputs/campaign-descriptor-v2.json` |
| `QUALIFICATION_CACHE_SEED_INPUT` | `task13/activations/$ACTIVATION_ID/qualification/cache-seed-input.json` |
| `H100_QUALIFICATION_INPUT` | `task13/activations/$ACTIVATION_ID/qualification/h100-input.json` |

Pass the same activation identity through descriptor publication and driver
materialization:

```bash
test -n "$ACTIVATION_ID"

aws/glm52-gpu/scripts/publish_glm52_task13_reviewed_artifact.py \
  --artifact-kind PRODUCTION_DESCRIPTOR \
  --activation-id "$ACTIVATION_ID" \
  --source "$FULL_RUN_WORK/inputs/campaign-descriptor-v2.json" \
  --expected-file-sha256 "$DESCRIPTOR_FILE_SHA256" \
  --expected-body-sha256 "$DESCRIPTOR_BODY_SHA256" \
  --bucket "$BUCKET" \
  --coordinate-output "$FULL_RUN_WORK/coordinates/production-descriptor.json"

aws/glm52-gpu/scripts/build_glm52_task13_driver_request.py h100 \
  --activation-id "$ACTIVATION_ID" \
  --campaign-descriptor "$FULL_RUN_WORK/inputs/campaign-descriptor-v2.json" \
  --output "$FULL_RUN_WORK/inputs/h100-driver-request.json"

aws/glm52-gpu/scripts/materialize_glm52_task13_fixed_artifacts.py \
  publish-driver \
  --request "$FULL_RUN_WORK/inputs/h100-driver-request.json" \
  --bucket "$BUCKET" \
  --coordinate-output "$FULL_RUN_WORK/coordinates/h100-input.json"
```

The cache-seed request uses the same `build_glm52_task13_driver_request.py
cache-seed --activation-id "$ACTIVATION_ID" ...` contract. Package validation
MUST reject any coordinate whose activation segment differs from the package
activation, even when the key and object are otherwise well formed.

- [ ] Derive the production object-authority list from exact current
  version-pinned source, non-vector-quantized package, accepted baseline,
  prompt packs, and repository archive.
- [ ] Build the production inventory:

  ```bash
  aws/glm52-gpu/scripts/build_production_s3_inventory.py \
    --profile keep-gpu \
    --region us-west-2 \
    --bucket "$BUCKET" \
    --run-id "$RUN_ID" \
    --output "$FULL_RUN_WORK/production-inventory.json"
  ```

- [ ] Start and collect the Amazon Simple Storage Service full-object
  checksum audit using the exact three commands in
  `aws/glm52-gpu/README.md`.
- [ ] Run `audit_s3_campaign_artifacts.py` against the completed checksum
  authority.
- [ ] Require every safetensors header, tensor byte range, object length,
  version identifier, full-object checksum, and campaign identity to pass.
- [ ] Refuse zero-length, truncated, multipart-incomplete, foreign-run,
  unversioned, or mismatched objects.
- [ ] Build a new immutable bundle from the already-created
  `repo.tar.gz`; set `REPO_TAR_SOURCE` and
  `EXPECTED_REPO_TAR_SHA256` so the builder cannot repack the checkout.
- [ ] Stage content first, descriptor and readiness last.
- [ ] Rehearse the exact staged descriptor and repository archive in a new
  clean temporary directory.
- [ ] Stop the rehearsal exactly where CUDA and a real H100 become
  mandatory.
- [ ] Prove the extracted archive, not the dirty checkout, contains every
  required script, unit file, policy, absolute path, model authority,
  checkpoint prefix, and resume directory.

Any artifact drift requires a new bundle identity and another rehearsal.
Never edit a staged descriptor in place.

## 9. Phase Five — Deploy Disabled Support Infrastructure

This is a live cloud mutation and requires an explicit live-change
authorization at execution time.

- [ ] Re-run the account and credential-expiry guard immediately before the
  change set.
- [ ] Create and inspect the exact no-execute CloudFormation change set.
- [ ] Verify the rendered template, role, parameters, resource graph, policy
  size, costs, and disabled launch state.
- [ ] Execute only that inspected change set.
- [ ] Require the stack to reach its expected complete state.
- [ ] Confirm `operator@example.com` has a confirmed Amazon Simple Notification
  Service subscription.
- [ ] Confirm the watchdog, dead-letter queue, alarms, log retention, state
  machines, schedules, roles, and private networking exist but no launch
  action is armed.
- [ ] Run disabled-stack denial tests from real role credentials.

Do not combine support deployment and worker activation in one opaque step.

## 10. Phase Six — Qualification-Cache Seed

The cancelled Job 3 is never resumed or reactivated. Create a fresh immutable
cache-seed submission under the same campaign history and current activation
authority.

- [ ] Confirm zero active campaign P5 workers and no open allocation.
- [ ] Confirm the spend ledger has the full authorized balance minus any
  newly proven prior consumption.
- [ ] Create a fresh twelve-hour `must_start_by`.
- [ ] Build the cache-seed bundle into
  `$FULL_RUN_WORK/cache-seed`, using the exact repository archive, then
  rehearse that descriptor.
- [ ] Validate the submission with no cloud launch.
- [ ] Arm and submit exactly one on-demand `p5.48xlarge` cache-seed job.
- [ ] Start the ten-minute watchdog.
- [ ] Cancel safely if no worker starts before the immutable deadline.
- [ ] Once running, confirm account, region, on-demand lifecycle, tags,
  status checks, Systems Manager, systemd, one-worker cardinality, and spend
  allocation.
- [ ] Generate exactly one full-version-two teacher row.
- [ ] Authenticate its four-layer checkpoints, router targets, layer-77
  hidden probe, normalized hidden state, and language-model-head slices.
- [ ] Publish immutable cache objects, manifest, and
  `TEACHER_CACHE_READY.json`, then publish
  `QUALIFICATION_CACHE_SEED_READY.json` last.
- [ ] Close the allocation and authenticate SkyPilot teardown.
- [ ] Verify no P5, unattached volume, or Elastic IP address remains.

The seed may consume at most six cumulative graphics-processing-unit hours.
Application or integrity failure preserves checkpoints and stops before
qualification.

## 11. Phase Seven — Post-Seed Bundle and Real H100 Recovery

- [ ] Authenticate the seed with
  `authenticate_qualification_cache_seed.py`.
- [ ] Build the post-seed bundle into `$FULL_RUN_WORK/post-seed`; its
  descriptor binds the real cache prefix and manifest digest.
- [ ] Reuse the exact same `repo.tar.gz`; do not repack.
- [ ] Re-run inventory, full-object audit, staging, and clean rehearsal.
- [ ] Confirm at least four graphics-processing-unit hours remain available
  for qualification and at least one hour would remain afterward.
- [ ] Submit one qualification Managed Job.
- [ ] On the first worker, run CUDA imports, topology, dense-prefix parity,
  long-context dynamic sparse attention parity, and a real four-layer-boundary
  checkpoint.
- [ ] Publish `SOURCE_NODE_READY.json`.
- [ ] Authenticate the source marker and exact active instance.
- [ ] Publish `QUALIFICATION_TERMINATION_REQUESTED.json`.
- [ ] Terminate that exact source instance through the guarded qualification
  supervisor.
- [ ] Let SkyPilot provision a different worker.
- [ ] Require the new instance identifier to differ.
- [ ] Resume the authenticated source checkpoint without duplicating or
  omitting work.
- [ ] Compare the resumed capture to the uninterrupted reference within the
  frozen tolerance.
- [ ] Run two real adapter forward/backward steps against the one-row cache.
- [ ] Require peak device memory below 70 GiB.
- [ ] Publish and authenticate `H100_RESUME_READY.json`.
- [ ] Close both qualification allocations and verify teardown.

Exit code `137`, out-of-memory failure, corruption, identity drift, or
schedule mismatch is terminal. Only exact exit code `75` receives up to two
bounded local retries.

## 12. Phase Eight — Production Submission

- [ ] Compute remaining authorization from the append-only spend ledger.
- [ ] Refuse production when one hour or less remains.
- [ ] Build the production bundle into `$FULL_RUN_WORK/production`, with a
  fresh twelve-hour descriptor if the qualification descriptor's launch
  window is no longer fresh.
- [ ] Bind:
  - `H100_RESUME_READY.json`;
  - real qualification cache;
  - accepted baseline;
  - full prompt pack;
  - training configuration;
  - repository archive;
  - approval records;
  - exact remaining spend.
- [ ] Re-run clean rehearsal.
- [ ] Confirm exactly zero active campaign P5 workers.
- [ ] Consume one activation and one production-submission action.
- [ ] Submit exactly one Managed Job.
- [ ] Confirm the first worker allocation and start the watchdog.

Never use the legacy raw Amazon Elastic Compute Cloud launch scripts.

## 13. Phase Nine — Production Campaign

### Bootstrap and CUDA gate

- [ ] Authenticate all staged objects on the worker.
- [ ] Restore campaign and spend ledgers and any compatible checkpoints.
- [ ] Start `keep-glm52-campaign.service` with `Restart=no`.
- [ ] Publish `BOOTSTRAP_READY.json`.
- [ ] Re-run the full CUDA parity and checkpoint gate on the production
  worker.

### Teacher generation

- [ ] Distribute all 257 sessions across eight accelerators by token-balanced
  deterministic partitions.
- [ ] Persist for every supervised position:
  - 32-bit integer position;
  - 32-bit integer target token identifier;
  - top 2,048 logit identifiers;
  - half-precision top 2,048 logit values;
  - 32-bit log-sum-exp;
  - half-precision tail mass;
  - layer-77 half-precision hidden probe;
  - top-eight router identifiers and normalized weights for layers 70–77.
- [ ] Checkpoint every four layers, preserving bfloat16 hidden state and
  dynamic sparse attention previous top-index state.
- [ ] Persist normalized hidden state after layer 78.
- [ ] Persist language-model-head slices independently.
- [ ] Synchronize checkpoint and heartbeat evidence at least every five
  minutes.

### Cache audit

- [ ] Require exactly 257 authenticated sessions.
- [ ] Require exactly 2,500,735 supervised positions.
- [ ] Verify every tensor name, shape, type, file digest, prompt identity,
  token identity, and split.
- [ ] Prove zero overlap with the frozen 66-prompt regression pack.
- [ ] Produce deterministic session-disjoint train, validation, and holdout
  splits using seed `20260712`.
- [ ] Permit only train rows in the optimizer schedule.
- [ ] Publish the cache manifest and `TEACHER_CACHE_READY.json` last.

### Training and evaluation

- [ ] Authenticate cache, baseline, and schedule.
- [ ] Run twenty timing steps.
- [ ] Require peak memory below 70 GiB.
- [ ] Require projected completion before the execution deadline minus sixty
  minutes after multiplying by the 1.5 safety factor.
- [ ] If the gate fails, publish `TRAINING_DEFERRED.json`, preserve the cache,
  and drain successfully.
- [ ] If the gate passes, train the canonical layer-77, all-projection,
  rank-four adapter for at most one epoch.
- [ ] Checkpoint locally every fifty optimizer steps and to Amazon Simple
  Storage Service every 250 steps or five minutes.
- [ ] Evaluate fixed validation windows every 2,048 steps and stop after two
  consecutive regressions.
- [ ] Evaluate the best checkpoint on session-disjoint holdout data and the
  frozen 66-prompt pack.
- [ ] Expand to layers 75, 74, 76, 72, 73, 71, and 70 only if layer 77 passes
  and each additional candidate fits the remaining deadline.
- [ ] Record `automatic_promotion: false`.

## 14. Phase Ten — Deadline, Drain, and Break Glass

For every allocation:

- [ ] At sixty minutes before the execution deadline, stop assigning new
  chunks or training windows and create the persistent stop request.
- [ ] At fifty minutes before the deadline, send the controlled termination
  signal and force a boundary checkpoint.
- [ ] Finish final synchronization and marker authentication.
- [ ] At thirty minutes before exhaustion, require authenticated drain or
  force the retained termination path.
- [ ] At authorization exhaustion, permit no useful compute to remain.

Use:

```bash
cd /Users/jack.mazac/Developer/keep
export AWS_PROFILE=keep-gpu
export FULL_RUN_WORK
FULL_RUN_WORK=$(cat /tmp/glm52-full-run-current)
test -d "$FULL_RUN_WORK"
export CAMPAIGN_DESCRIPTOR="$FULL_RUN_WORK/production/campaign-descriptor-v2.json"
export SKY_BIN=/Users/jack.mazac/.local/share/keep/skypilot-0.13.0/bin/sky
export SKYPILOT_CONFIG=/Users/jack.mazac/.local/share/keep/skypilot-0.13.0/server-config.yaml
test -f "$CAMPAIGN_DESCRIPTOR"

aws/glm52-gpu/scripts/sky_campaign_break_glass.sh status
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh logs
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh inspect-worker
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh graceful-stop
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh verify
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh audit-orphans
```

The descriptor path must be the exact active local readback, not a guessed
path. Use `cancel` only after the graceful-stop request and checkpoint
evidence, unless immediate termination is required to prevent unauthorized
spend.

## 15. Phase Eleven — Terminal Verification and Cost

- [ ] Authenticate `CAMPAIGN_DRAINED.json`.
- [ ] Authenticate the final campaign and spend ledgers byte-for-byte against
  their version-pinned remote objects.
- [ ] Authenticate `TERMINAL_VERIFIED.json`.
- [ ] Close the final allocation only after terminal instance proof.
- [ ] Confirm SkyPilot job terminality.
- [ ] Confirm no campaign `p5.48xlarge` exists in any billable state.
- [ ] Inspect all campaign Elastic Block Store volumes, Elastic IP addresses,
  security groups, network interfaces, controllers, temporary buckets,
  schedules, state machines, dead-letter queues, and alarms.
- [ ] Resolve or explicitly retain the old stopped-controller root volume.
- [ ] Stop monitoring only after the drained marker authenticates.
- [ ] Produce a cost report with:
  - seed seconds and cost;
  - source qualification seconds and cost;
  - replacement qualification seconds and cost;
  - every production or recovery allocation;
  - total graphics-processing-unit seconds and cost;
  - remaining approval;
  - controller, storage, checksum, network, Lambda, logs, workflow, queue,
    notification, and root-volume costs;
  - every incident tail.

Cloud billing dashboards may lag. Do not let a delayed dashboard contradict a
complete immediate spend ledger, but reconcile both when delayed data becomes
available.

## 16. Phase Twelve — Manual Model Disposition

Present the authenticated evaluation package to the owner. Record exactly one
decision:

### Reject or no promotion

- Preserve teacher cache, checkpoints, schedule, evaluation, and spend
  evidence.
- Leave the accepted baseline unchanged.
- Close the campaign.

### Further experiment

- Define a new immutable candidate configuration.
- Reuse the durable teacher cache.
- Obtain additional graphics-processing-unit approval if the original
  envelope is exhausted.
- Build and rehearse a new descriptor before launch.

### Promote

- Obtain explicit promotion and publication authorization.
- Reconstruct the adapter from the authenticated best checkpoint.
- Verify baseline, cache, schedule, candidate, and evaluation identities.
- Update the accepted baseline through a distinct audited action.
- Upload to a registry only if that upload is separately authorized.

The full-run authorization does not itself authorize promotion.

## 17. Stop Conditions

Stop immediately and preserve evidence when any of these occurs:

- wrong Amazon Web Services account or region;
- Spot, Capacity Block, alternate instance type, or raw launch path;
- more than one active campaign P5 worker;
- missing or changed approval;
- support or residual-liability approval absent before possible send;
- repository archive, descriptor, artifact, or checkpoint identity drift;
- failed full-object checksum or tensor-range audit;
- unreviewed live change set;
- unconfirmed alert subscription;
- open or unreconciled prior allocation;
- cumulative runtime or cost exhaustion;
- hardware qualification failure;
- peak training memory at or above 70 GiB;
- cache count other than 257 sessions or 2,500,735 positions;
- heartbeat older than thirty minutes;
- systemd failure;
- launch, cancellation, termination, or ledger response ambiguity without
  coherent readback;
- inability to prove terminal worker and closed spend before settlement.

Do not convert a stop into an ad hoc launch, a new run identifier, a new
ClientToken, a different cloud, a relaxed validator, or a larger budget.

## 18. First Action for the Next Session

The next session starts at Phase Zero, not at a cloud launch. Its first
deliverable is a short evidence update containing:

1. exact current worktree status;
2. candidate-thirteen digest match;
3. current Research and Development account identity and credential expiry;
4. current controller, volume, P5, stack, alert, and campaign-object state;
5. an implementation gap matrix mapping every candidate-thirteen component
   to existing, missing, or contradictory code;
6. the first failing enforcement test selected from that matrix.

Only after that first failing test is real should implementation begin.
