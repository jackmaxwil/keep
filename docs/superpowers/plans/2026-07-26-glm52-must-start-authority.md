# GLM52 Exact Timely-Start Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add disabled-by-default exact timely-start authority for current
immutable Job 3 and future worker-published latches without permitting a late
or foreign submission to spend.

**Architecture:** Extend the AWS-independent must-start policy with strict,
hash-authenticated job-binding, observation, worker-latch, accepted-start, and
superseded records. The coordinator observes and durably binds one numeric
SkyPilot job before any exact-ID cancellation; future workers publish a
content-addressed identity latch before repository download or spend
allocation. CloudFormation keeps observation and active scheduling separately
gated and provides a one-time deadline schedule plus periodic reconciliation.

**Tech Stack:** Python 3.13 Lambda, Python 3 worker bootstrap, boto3, pinned
SkyPilot 0.13 controller APIs, Bash, CloudFormation, EventBridge Scheduler,
S3, SSM, SNS, pytest.

## Global Constraints

- Work only in `/Users/jack.mazac/Developer/keep`; preserve the dirty shared
  worktree and `runs/`.
- Do not commit, stage, push, merge, reset, clean, checkout, or mutate branches.
- Do not call live AWS/S3/SkyPilot/SSM/SNS/CloudFormation mutation APIs.
- Do not edit
  `aws/glm52-gpu/scripts/rehearse_staged_control_plane.sh` or the concurrent
  rehearsal-evidence test.
- Account authority is `246813579024`; region authority is `us-west-2`.
- Current live coordinates belong only in tests/explicit deployment
  parameters, never general defaults.
- `EnableSkyMustStartCancel=false` and observe-only deployment remain binding
  until independent review and live observation approve activation.

---

### Task 1: Strict authority records and compatibility decision table

**Files:**
- Modify: `src/mlx_vq/quality/glm52_sky_must_start.py`
- Modify: `tests/test_glm52_sky_must_start.py`

**Interfaces:**
- Produces:
  `build/validate_must_start_job_binding`,
  `build/validate_must_start_controller_observation`,
  `build/validate_worker_start_latch`,
  `build/validate_timely_start_accepted`,
  `build/validate_must_start_cancel_superseded`, and
  `decide_controller_start_authority`.
- Consumes the existing canonical authority and digest helpers.

- [ ] Write RED tests for exact fields, hashes, unknown-field rejection,
  foreign descriptor/submission rejection, positive IDs, workspace `default`,
  and current PENDING/STARTING/RUNNING/RECOVERING/terminal decisions.
- [ ] Run
  `.venv/bin/pytest -q tests/test_glm52_sky_must_start.py -k 'job_binding or controller_start or worker_latch or superseded'`
  and record the missing-interface failures.
- [ ] Implement only the strict records and pure decision function required by
  those tests.
- [ ] Rerun the focused policy tests and retain exact GREEN counts.

### Task 2: Current Job 3 observation, binding, and exact cancellation

**Files:**
- Modify: `aws/glm52-gpu/lambda/sky_must_start_cancel_handler.py`
- Modify: `tests/test_glm52_sky_must_start_lambda.py`

**Interfaces:**
- Consumes Task 1 records and decisions.
- Produces observation-only and active coordinator outcomes; publishes
  `JOB_BINDING.json`, content-addressed observations,
  `TIMELY_START_ACCEPTED.json`, and `CANCEL_SUPERSEDED.json`.

- [ ] Write RED fake-client tests with literal live-shaped history rows
  `[1, 2, 3]` for PENDING exact-ID cancellation, timely start suppression,
  bounded STARTING races, observe-only no-cancel, wrong authority no mutation,
  duplicate convergence, and superseded request.
- [ ] Run only the new tests and confirm missing behavior rather than fixture
  errors.
- [ ] Extend controller resolution to require exact instance ID, instance type,
  IAM profile, cluster tags, region, and SSM `Online`.
- [ ] Extend the pinned shim to observe exact row fields and to cancel only
  `cancel_jobs_by_id([target_id], current_workspace="default",
  graceful=False)` after an immediate identity reread.
- [ ] Implement the 20-second bounded STARTING grace through injected sleep
  and repeated exact observations.
- [ ] Rerun the Lambda behavior and package-import tests.

### Task 3: Future worker latch before download and spend

**Files:**
- Create: `aws/glm52-gpu/skypilot/publish_must_start_latch.py`
- Modify: `aws/glm52-gpu/skypilot/glm52-campaign.yaml`
- Modify: `aws/glm52-gpu/skypilot/run_managed_campaign.sh`
- Modify: `aws/glm52-gpu/scripts/submit_sky_campaign.sh`
- Create: `tests/test_glm52_sky_worker_must_start.py`
- Modify: `tests/test_glm52_skypilot_task.py`

**Interfaces:**
- Publisher consumes descriptor/submission bytes plus STS and IMDS identity,
  emits one canonical worker latch, and conditionally writes its
  content-addressed key.
- Coordinator accepts only S3 `LastModified <= must_start_by` and writes
  `TIMELY_START_ACCEPTED.json`.

- [ ] Write RED subprocess tests for exact STS/IMDS/descriptor/submission/repo
  binding, content-addressed keys, conditional convergence, expiry before
  spend, late S3 `LastModified`, same-submission recovery, and cross-submission
  rejection.
- [ ] Run the new worker tests and record missing-script/behavior RED.
- [ ] Implement the standalone publisher and mount it with the policy module
  into future Sky tasks so it runs after small authority downloads but before
  repository tar download.
- [ ] Add submission URI/body-SHA environment authority.
- [ ] Add a second accepted-latch check immediately before
  `manage_gpu_spend.py start`; expired workers without accepted authority exit.
- [ ] Rerun worker and Sky task tests.

### Task 4: Disabled scheduler, IAM, packaging, and deployment gates

**Files:**
- Modify: `aws/glm52-gpu/cfn/gpu-teacher-stack.yaml`
- Modify: `aws/glm52-gpu/scripts/package_sky_must_start_cancel_lambda.sh`
- Modify: `aws/glm52-gpu/scripts/deploy_sky_control_plane.sh`
- Modify: `tests/test_glm52_sky_cloudformation.py`
- Modify: `tests/test_glm52_sky_staging.py`

**Interfaces:**
- Produces an exact one-time EventBridge Scheduler target and retains periodic
  reconciliation.
- Supplies observe-only/controller identity and exact SHA-scoped S3 authority
  to Lambda.

- [ ] Write RED template tests for scheduler input, scheduler invoke role,
  reconciliation rule, observe-only default, positive target, exact controller
  parameters, SSM Online permission, and complete SHA-scoped marker paths.
- [ ] Run CloudFormation/staging tests and record expected failures.
- [ ] Add disabled-by-default parameters/resources and exact IAM paths.
- [ ] Require observe-only before active mode in the deploy script; do not call
  deployment commands during tests.
- [ ] Build and import the Lambda zip, run `cfn-lint`, and rerun focused tests.

### Task 5: Verification and evidence

**Files:**
- Create: `.superpowers/sdd/goal-objective/task-1b-report.md`

**Interfaces:**
- Records only commands actually run and evidence actually observed.

- [ ] Run the Task 1b new-surface regression.
- [ ] Run the existing must-start/watchdog/campaign/Sky task regression.
- [ ] Run shell syntax, Python compilation, Ruff, `cfn-lint`, package import,
  and `git diff --check`.
- [ ] Write exact RED/GREEN counts, changed files, deployment hold, deferred
  risks, and confirmation of no AWS/Git mutation.
