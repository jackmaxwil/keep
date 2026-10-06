# GLM-5.2 full campaign Phase Zero evidence — 2026-07-28

## Decision

`STOP_BEFORE_POSSIBLY_SENT`.

The local checkpoint was committed and fast-forwarded to local `main`, but the
full campaign cannot launch yet. The campaign prefix contains only the existing
GPU spend approval. It contains neither a production-support-plane approval nor
`GPU_RESIDUAL_LIABILITY_APPROVAL.json`. The execution handoff requires both
authorities before any request can become `POSSIBLY_SENT`.

The candidate-thirteen production enforcement plane is also not implemented.
The first prescribed focused test run proves that the guarded submitter remains
qualification-only.

## Local snapshot

- Repository: `/Users/jack.mazac/Developer/keep`
- Branch: `main`
- HEAD: `5fa13229` (`feat: add guarded GLM-5.2 campaign control plane`)
- Local relationship: `main...origin/main [ahead 173]`
- Feature branch retained: `keep-glm52-pipeline-and-p1-lock` at `5fa13229`
- Architecture SHA-256:
  `1daaa963d42e5fb94ed712c07d83c6b43f16fbcc9e89b0df0c7721587d8c051e`
- Expected architecture SHA-256:
  `1daaa963d42e5fb94ed712c07d83c6b43f16fbcc9e89b0df0c7721587d8c051e`
- Architecture digest result: `MATCH`
- No remote pull, push, or branch deletion was performed.

## AWS identity and login

- Login command: `aws login --profile keep-gpu`
- Login completion: 2026-07-28 at approximately 19:17 UTC
- Repository account guard: `246813579024`
- Direct STS account: `246813579024`
- Direct STS ARN:
  `arn:aws:sts::246813579024:assumed-role/AWSReservedSSO_AdministratorAccess_0123456789abcdef/operator@example.com`
- Profile region: `us-west-2`
- Newest observed login ID-token expiry: `2026-07-28T19:32:46Z`
- The AWS CLI login cache uses refresh credentials and does not expose the
  overall session expiry as a separate field. Re-run login, STS, and the
  repository account guard immediately before any future live mutation.

## Fresh cloud reconciliation

Observed at 2026-07-28 19:18–19:21 UTC using explicit profile
`keep-gpu` and region `us-west-2`:

- Active campaign `p5.48xlarge` workers: zero.
- Controller `i-0511af4e31aa5406a`:
  - type `c6a.xlarge`;
  - state `running`;
  - launch/start time `2026-07-28T14:31:58Z`;
  - EC2 system and instance reachability `ok`;
  - SSM `Online`;
  - instance profile
    `arn:aws:iam::246813579024:instance-profile/keep-glm52-skypilot-controller`.
- CloudTrail attributes the current controller start to the automated principal
  `246813579024-fcc-offhours`, event
  `a5ab70a4-e668-4806-8f28-e9c1f3695641`, at
  `2026-07-28T07:31:58-07:00`.
- Attached controller volume:
  `vol-04d1a5b0f39076220`, `gp3`, 50 GiB, `in-use`,
  `DeleteOnTermination=true`.
- Campaign Elastic IP addresses: zero.
- Stack `keep-glm52-gpu`: `UPDATE_COMPLETE`; last update
  `2026-07-26T12:57:37.257Z`.
- SNS email subscription for `operator@example.com`: confirmed ARN present.
- Eight `keep-glm52` CloudWatch alarms: all `OK`, actions enabled.
- EventBridge must-start cancellation rule: `DISABLED`.
- EventBridge watchdog rule: `ENABLED`, every ten minutes.
- Scheduler schedules with the `keep-glm52` prefix: zero.
- Pinned SkyPilot client: `0.13.0`, commit
  `b1431e52d97c22e9bb8fa8b67f162543754ddaf5`.
- Direct `sky jobs queue --all` result:
  `ClusterNotUpError: No in-progress managed jobs.`

The controller was not started by this run. The read-only SkyPilot query
started only the local loopback API server after finding no connected local
server.

## Campaign-object inventory

- Bucket:
  `keep-glm52-models-246813579024-us-west-2`
- Prefix: `campaigns/glm52-sky-20260724/`
- Exact object count: 620; listing not truncated.
- Approval-named objects: exactly one:
  `authorities/GPU_SPEND_APPROVAL-667d49b4096fbac5bb685a305219b4bb59b643dc9279d2b1180aaae2516c09ea.json`
  (1,510 bytes).
- Production-support-plane approval: absent.
- `GPU_RESIDUAL_LIABILITY_APPROVAL.json`: absent.
- Latest watchdog state: `alarm`.
- Latest watchdog finding:
  `SkyPilot must-start deadline reached with no running GPU`.
- Latest watchdog still sees the historical `SUBMITTED` cache-seed status, no
  campaign phase, no worker instances, and no campaign/spend/checkpoint
  progress. This is not a valid fresh submission authority.

## Candidate-thirteen implementation gap matrix

| Candidate-thirteen component | Classification | Current evidence |
| --- | --- | --- |
| Standalone import-light `glm52_enforcement` package | **MISSING** | No package directory, module, import, or test exists under `src`, `aws`, or `tests`. |
| Activation-index, rollover, execution, action, recovery, finalization, snapshot-cleanup, worker-launch, launch-liability, liability-action, liability-settlement, post-terminal-allocation, operator-disposition, and terminal-v2 schemas | **MISSING** | Exact candidate record names and the no-launch terminal outcome are absent from executable code and tests. |
| Canonical candidate hash chains, invocation-nonce ownership, conditional transitions, coherent readback, duplicate adoption, and owner takeover | **MISSING** | Older modules have separate canonical JSON and submission locks, but no candidate-thirteen state package or transaction contract exists. |
| Snapshot-cleanup chain, rollover atomic lineage, three-activation cleanup, and post-rollover cleanup authority | **MISSING** | No matching executable implementation or acceptance tests were found. |
| Nominal 900-second GPU reserve, root-volume tail, residual-liability validator, settlement, and termination-only continuation | **MISSING** | The terms occur only in the handoff/architecture. No validator exists, and the required approval artifact is absent. |
| Retained foundation, parameterless production-fence stack, and ephemeral support stack | **CONTRADICTORY** | The current deployment is one parameterized `gpu-teacher-stack.yaml` with conditional support/controller resources, not three ownership stacks. |
| Candidate three-subnet/NAT/EIP/gateway-endpoint/Secrets Manager endpoint network | **CONTRADICTORY** | Current template has the older VPC/subnet topology and an S3 gateway endpoint; no candidate NAT gateway, isolated-subnet set, or Secrets Manager endpoint exists. |
| Patched SkyPilot provisioner writes launch intent only; retained completer is sole `RunInstances` sender | **MISSING / CONTRADICTORY** | No completer exists. Current CloudFormation policies still contain `ec2:RunInstances`; no authenticated patched provisioner hash is wired. |
| Combined controller denied `RunInstances`, `TerminateInstances`, `StartInstances`, and `PassRole` | **CONTRADICTORY** | Current policies and roles retain launch/termination/pass-role paths for the older control plane. The live controller is the older `c6a.xlarge` design. |
| Exact one on-demand `p5.48xlarge` worker shape | **PARTIAL** | Existing descriptors/tests enforce the approved region, market, type, price, and single worker, but not through candidate-thirteen launch custody. |
| SkyPilot 0.13.0 pin and dedicated AWS-only local control environment | **EXISTING** | Pinned client and dedicated config exist and authenticate locally. Candidate patched-source hash and API identity/RBAC/consolidation probes do not. |
| One-wire guarded production submission | **MISSING** | `submit_sky_campaign.sh` accepts only `--qualification`; the Python CLI exposes qualification and cache-seed paths but no executable production route. |
| Distinct cache-seed, qualification, and production immutable intents | **PARTIAL** | Cache-seed and qualification orchestration exist. Production intent validation exists as a library, but it is not connected to the guarded submitter. |
| Candidate deadlines, terminal-v2, retained finalization, forensic snapshot cleanup, and orphan/cost closure | **PARTIAL / MISSING** | Older watchdog, spend-ledger, drain, and terminal-v1 code exists. Candidate support execution, terminal-v2, settlement, and cleanup lineage do not. |
| Twenty-five deferred transport rows and twenty-two transport mutants | **MISSING** | No candidate enforcement test suite or exact row/mutant ledger exists. |
| Candidate implementation acceptance and disabled-stack rehearsal | **MISSING** | Existing focused campaign tests exercise the pre-candidate control plane only. No three-stack disabled rehearsal exists. |

Unclassified candidate-thirteen capabilities are treated as missing.

## First failing test

Command:

```bash
.venv/bin/python -m pytest -q \
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

Result:

```text
558 passed, 1 failed, 2 warnings in 167.66s
```

Failing assertion:

```text
tests/test_glm52_h100_qualification.py:
test_qualification_worker_requires_real_replacement_and_two_step_smoke

assert "H100_RESUME_READY.json" in submit
```

Root cause: the guarded shell submitter is intentionally qualification-only
and delegates only to `submit_sky_campaign.py qualification`. Production
intent validation code is not an executable production submission path. Adding
the marker name as a comment would only mask the missing one-wire production
contract and is not an acceptable fix.

## Required external authorities before launch

1. An authenticated production-support-plane approval covering the exact
   resource cardinalities, 68/71/72-hour lifecycle, NAT/EIP, logs,
   invocations, workflow history, secrets, endpoint, storage, forensic
   snapshot, and dated price-model identity.
2. An authenticated residual launch-liability approval covering the nominal
   900 GPU seconds and `$13.76` reserve, separately priced 300-GiB root-volume
   tail up to `$0.01`, delayed visibility/control/termination and same-token
   multiplicity, termination-only continuation until settlement, and explicit
   acknowledgment that AWS provides no hard post-acceptance billing cap.

Neither authority may be synthesized from the general instruction to execute
the campaign.
