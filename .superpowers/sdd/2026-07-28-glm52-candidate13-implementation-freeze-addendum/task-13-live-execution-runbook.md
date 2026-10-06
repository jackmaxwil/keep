# Task 13 live execution runbook

Status: documentation-only operator runbook, prepared from the repository
sources present on 2026-07-29. This file is not live-change authorization and
does not prove that any AWS object, stack, measurement, H100 result, launch, or
terminal marker exists.

Do not run past a `BLOCKED` condition in this document. In particular, do not
replace a missing repository-owned materializer with an inline script, an
unversioned S3 object, a guessed identifier, or a raw SkyPilot/EC2 launch.

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

## 1. Closed scope and execution order

The only supported stage order is:

1. Materialize and publish the exact immutable reviewed artifacts for the
   21-coordinate predecessor. Do not publish clean-rehearsal evidence yet.
2. Build the 21-artifact `PREQUALIFICATION` request, reviewed-artifact list,
   and package.
3. Run the pure `validate` stage with no authority and no journals.
4. Run, with one fresh authority per stage:
   `deploy-disabled`, `collect-first-five`, `collect-remaining`, and
   `finalize`.
5. Preserve the exact finalizer invocation metadata, payload bytes, and
   immutable gate readback. Build CLEAN evidence from those captures and the
   unchanged predecessor/archive lineage, then publish it only at its
   activation/self-hash key.
6. Run, with one fresh authority per stage, `qualification-cache-seed` and
   `h100-qualification`.
7. Materialize `TASK10_PRODUCTION_AUTHORITY` from the authenticated H100
   result and build the 25-artifact `PRODUCTION` successor by adding exactly
   CLEAN plus the three Task 10 coordinates.
8. Run the pure `validate` stage again on that successor.
9. Run `launch`. The runner performs the Task 10 start and immediate durable
   reconciliation; an ambiguous result is reconciled by replaying the same
   Task 13 stage against the same journals, never by sending a second launch.
10. Run `monitor` as discrete, operator-requested readbacks.
11. After the terminal publishers have created all required generation
    `00000001` records, run `terminal`.

The prequalification and production packages must use the same three journal
paths. Every one of the 21 predecessor coordinates remains byte-identical.
The production package adds exactly `CLEAN_REHEARSAL`,
`TASK10_PRODUCTION_AUTHORITY`, `TASK10_WORKER_DESCRIPTOR`, and
`TASK10_TASK_INPUTS`; it does not replace or reissue a predecessor row. The
runner recognizes the predecessor identity when it proves that the six
preproduction stages were already committed.

The following actions are always forbidden:

- raw `sky jobs launch`, raw `ec2:RunInstances`, Spot, Capacity Block, another
  region, another instance type, or more than one active `p5.48xlarge`;
- repacking the repository between cache seed, H100 qualification, and
  production;
- reusing one controller authority for another stage;
- replacing any opaque S3 `VersionId` with `latest`, `null`, or a guessed
  value;
- truncating, copying, joining, or editing a journal;
- retrying a mutation after `POSSIBLY_SENT`;
- treating `monitor` success as terminal proof;
- treating this runbook as model-promotion or publication authority.

## 2. Exact constants and private local workspace

Run all commands from the exact repository root:

```bash
cd /Users/jack.mazac/Developer/keep
export TASK13_ROOT=/Users/jack.mazac/Developer/keep
export AWS_PROFILE=keep-gpu
export AWS_REGION=us-west-2
export AWS_PAGER=
export TASK13_ACCOUNT=246813579024
export TASK13_RUN_ID=glm52-sky-20260724
export TASK13_BUCKET=keep-glm52-models-246813579024-us-west-2
export TASK13_ACTIVATION_ID=replace-with-reviewed-activation-id

export FULL_RUN_WORK
test -f /private/tmp/glm52-full-run-current
test ! -L /private/tmp/glm52-full-run-current
test "$(stat -f '%Lp' /private/tmp/glm52-full-run-current)" = 600
FULL_RUN_WORK=$(cat /private/tmp/glm52-full-run-current)
test -d "$FULL_RUN_WORK"
test ! -L "$FULL_RUN_WORK"
test "$(stat -f '%Lp' "$FULL_RUN_WORK")" = 700

export TASK13_SIGNING_PRIVATE_KEY="$FULL_RUN_WORK/controller-authority-ed25519"
test -f "$TASK13_SIGNING_PRIVATE_KEY"
test ! -L "$TASK13_SIGNING_PRIVATE_KEY"
test -f "${TASK13_SIGNING_PRIVATE_KEY}.pub"
test ! -L "${TASK13_SIGNING_PRIVATE_KEY}.pub"

export TASK13_WORK="$FULL_RUN_WORK/task13-live-execution"
test ! -e "$TASK13_WORK"
mkdir -m 700 "$TASK13_WORK"
mkdir -m 700 \
  "$TASK13_WORK/authorities" \
  "$TASK13_WORK/coordinates" \
  "$TASK13_WORK/inputs" \
  "$TASK13_WORK/journals" \
  "$TASK13_WORK/packages" \
  "$TASK13_WORK/preflight" \
  "$TASK13_WORK/results"
```

On a resumed operator session, recover rather than recreate the workspace:

```bash
cd /Users/jack.mazac/Developer/keep
export TASK13_ROOT=/Users/jack.mazac/Developer/keep
export AWS_PROFILE=keep-gpu
export AWS_REGION=us-west-2
export AWS_PAGER=
export TASK13_ACCOUNT=246813579024
export TASK13_RUN_ID=glm52-sky-20260724
export TASK13_BUCKET=keep-glm52-models-246813579024-us-west-2
export TASK13_ACTIVATION_ID=replace-with-reviewed-activation-id

export FULL_RUN_WORK
test -f /private/tmp/glm52-full-run-current
test ! -L /private/tmp/glm52-full-run-current
test "$(stat -f '%Lp' /private/tmp/glm52-full-run-current)" = 600
FULL_RUN_WORK=$(cat /private/tmp/glm52-full-run-current)
test -d "$FULL_RUN_WORK"
test ! -L "$FULL_RUN_WORK"
test "$(stat -f '%Lp' "$FULL_RUN_WORK")" = 700

export TASK13_SIGNING_PRIVATE_KEY="$FULL_RUN_WORK/controller-authority-ed25519"
test -f "$TASK13_SIGNING_PRIVATE_KEY"
test ! -L "$TASK13_SIGNING_PRIVATE_KEY"
test -f "${TASK13_SIGNING_PRIVATE_KEY}.pub"
test ! -L "${TASK13_SIGNING_PRIVATE_KEY}.pub"

export TASK13_WORK="$FULL_RUN_WORK/task13-live-execution"
test -d "$TASK13_WORK"
test ! -L "$TASK13_WORK"
test "$(stat -f '%Lp' "$TASK13_WORK")" = 700
```

Set the three journal paths once. Never change them during the campaign:

```bash
export TASK13_FENCE_JOURNAL="$TASK13_WORK/journals/fence.jsonl"
export TASK13_SUPPORT_JOURNAL="$TASK13_WORK/journals/support.jsonl"
export TASK13_CAMPAIGN_JOURNAL="$TASK13_WORK/journals/campaign.jsonl"
test "$TASK13_FENCE_JOURNAL" != "$TASK13_SUPPORT_JOURNAL"
test "$TASK13_FENCE_JOURNAL" != "$TASK13_CAMPAIGN_JOURNAL"
test "$TASK13_SUPPORT_JOURNAL" != "$TASK13_CAMPAIGN_JOURNAL"
```

If a journal already exists, it must be a regular non-symlink file with mode
`0600`. A new journal is created by the runner with mode `0600`. Do not create
empty aliases or hard links.

## 3. Credential and account guard

Run this read-only guard immediately before every AWS publication or external
runner stage:

```bash
aws sts get-caller-identity \
  --profile keep-gpu \
  --region us-west-2 \
  --no-cli-pager \
  > "$TASK13_WORK/preflight/caller.json"

aws configure export-credentials \
  --profile keep-gpu \
  --format process \
  > "$TASK13_WORK/preflight/credentials.json"

python3 - "$TASK13_WORK/preflight/caller.json" \
  "$TASK13_WORK/preflight/credentials.json" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

caller = json.loads(Path(sys.argv[1]).read_bytes())
credentials = json.loads(Path(sys.argv[2]).read_bytes())
if caller.get("Account") != "246813579024":
    raise SystemExit("wrong AWS account")
expiration = credentials.get("Expiration")
if not isinstance(expiration, str):
    raise SystemExit("credential expiration is absent")
expires = datetime.fromisoformat(expiration.replace("Z", "+00:00"))
remaining = int((expires - datetime.now(timezone.utc)).total_seconds())
if remaining < 3600:
    raise SystemExit("credential lifetime is below 3600 seconds")
print(json.dumps(
    {"account_id": caller["Account"], "seconds_remaining": remaining},
    sort_keys=True,
    separators=(",", ":"),
))
PY
```

The production coordinator repeats the same account and credential-expiration
guard immediately before every mutation. A locally passing preflight does not
override a later coordinator refusal.

## 4. Immutable repository archive

Create the archive exactly once. The output path must not already exist:

```bash
test ! -e "$TASK13_WORK/repo.tar.gz"
aws/glm52-gpu/scripts/build_sky_repository_tar.sh \
  "$TASK13_WORK/repo.tar.gz"
shasum -a 256 "$TASK13_WORK/repo.tar.gz" \
  > "$TASK13_WORK/repo.tar.gz.sha256"
chmod 600 \
  "$TASK13_WORK/repo.tar.gz" \
  "$TASK13_WORK/repo.tar.gz.sha256"
```

The archive publisher creates two immutable versioned objects:

- the archive payload at
  `campaigns/glm52-sky-20260724/repository/keep-{archive_sha256}.tar.gz`;
- its reviewed manifest at the activation-scoped key
  `task13/activations/{activation_id}/archive/repo-tar.json`.

The manifest, not the payload coordinate alone, is the
`REPOSITORY_ARCHIVE` reviewed artifact. It must bind the exact activation,
payload `VersionId`, byte length, and file SHA-256.

After explicit S3-publication authorization and the credential guard, publish
the payload and create-only manifest through the repository-owned typed
boundary:

```bash
export TASK13_ARCHIVE_MANIFEST="$TASK13_WORK/coordinates/repository-archive-manifest.json"
export TASK13_ARCHIVE_COORDINATE="$TASK13_WORK/coordinates/repository-archive.json"
test ! -e "$TASK13_ARCHIVE_MANIFEST"
test ! -e "$TASK13_ARCHIVE_COORDINATE"
python3 \
  aws/glm52-gpu/scripts/materialize_glm52_task13_fixed_artifacts.py \
  publish-archive \
  --activation-id "$TASK13_ACTIVATION_ID" \
  --archive "$TASK13_WORK/repo.tar.gz" \
  --bucket "$TASK13_BUCKET" \
  --manifest-output "$TASK13_ARCHIVE_MANIFEST" \
  --coordinate-output "$TASK13_ARCHIVE_COORDINATE"
```

The CLI verifies bucket versioning, uses the exact account and region, admits
one total SDK attempt, and refuses to overwrite either local output. It
create-only persists the exact canonical manifest bytes already validated at
publication into `TASK13_ARCHIVE_MANIFEST`; both the fresh-coordinate join
and CLEAN evidence builder consume those same bytes.

## 5. Exact reviewed-artifact keys

`PREQUALIFICATION` contains the 21 common rows below and no CLEAN or Task 10
production row. `PRODUCTION` preserves those 21 coordinates byte-for-byte and
adds exactly the four production-only rows.

| Phase | Artifact kind | Exact key |
|---|---|---|
| Common | `TASK11_REVIEW_APPROVAL` | `reviews/task11/approval.json` |
| Common | `TASK12_REVIEW_APPROVAL` | `reviews/task12/approval.json` |
| Common | `RETAINED_FOUNDATION_TEMPLATE` | `task13/templates/retained-foundation.yaml` |
| Common | `RETAINED_PRE_SUPPORT_TEMPLATE` | `task13/templates/retained-pre-support.yaml` |
| Common | `RETAINED_TEMPLATE` | `task13/templates/retained.yaml` |
| Common | `FENCE_TEMPLATE` | `task13/migration/final-fence.json` |
| Common | `SUPPORT_TEMPLATE` | `task13/templates/support-disabled.yaml` |
| Common | `SUPPORT_INPUTS` | `task13/inputs/support-build-inputs.json` |
| Common | `BOOTSTRAP_TEMPLATE` | `task13/templates/container-bootstrap-v1.json` |
| Common | `QUALIFICATION_CACHE_SEED_INPUT` | `task13/qualification/cache-seed-input.json` |
| Common | `H100_QUALIFICATION_INPUT` | `task13/qualification/h100-input.json` |
| Common | `T01_T25_GATE` | `task13/gates/t01-t25.json` |
| Common | `TRANSPORT_22_MUTANT_GATE` | `task13/gates/transport-22-mutants.json` |
| Common | `REPOSITORY_ARCHIVE` | `task13/activations/{activation_id}/archive/repo-tar.json` |
| Common | `ACCEPTED_BASELINE` | `task13/inputs/accepted-baseline.json` |
| Common | `PROMPT_PACK` | `task13/inputs/prompt-pack.json` |
| Common | `TRAINING_CONFIGURATION` | `task13/inputs/training-configuration.json` |
| Common | `GPU_SPEND_APPROVAL` | `task13/approvals/gpu-spend.json` |
| Common | `SUPPORT_APPROVAL` | `task13/approvals/support-plane.json` |
| Common | `RESIDUAL_LIABILITY_APPROVAL` | `task13/approvals/residual-liability.json` |
| Common | `PRODUCTION_DESCRIPTOR` | `task13/inputs/campaign-descriptor-v2.json` |
| Production only | `CLEAN_REHEARSAL` | `task13/gates/clean-rehearsal/{activation_id}/{evidence_identity_sha256}.json` |
| Production only | `TASK10_PRODUCTION_AUTHORITY` | `task13/production/task10-production-authority.json` |
| Production only | `TASK10_WORKER_DESCRIPTOR` | `task13/production/task10-worker-descriptor.json` |
| Production only | `TASK10_TASK_INPUTS` | `task13/production/task10-task-inputs.json` |

Every coordinate has exactly these fields:

```text
artifact_kind
bucket
key
version_id
file_sha256
body_sha256
```

All fields must come from exact live publication/readback. A reviewed-artifact
list is sorted by `artifact_kind` by the package builder.

### 5.1 Transport gates

The committed gate files are:

```text
task13/gates/t01-t25.json
task13/gates/transport-22-mutants.json
```

They must be resealed only after every source file has stabilized. The
publication CLI performs a fresh independent pytest observation before any
S3 write, uses one conditional `If-None-Match: *` put per key, and reconciles
an ambiguous response without a second put.

After explicit live S3-publication authorization and the credential guard:

```bash
test ! -e "$TASK13_WORK/coordinates/transport-gates.json"
python3 \
  aws/glm52-gpu/scripts/publish_glm52_task13_transport_gates.py \
  --output-coordinates \
  "$TASK13_WORK/coordinates/transport-gates.json"
```

The output is a canonical two-coordinate array. Do not treat local gate
hashes as S3 `VersionId` evidence.

### 5.2 Cache-seed and H100 driver inputs

The reviewed driver records contain only the exact `operation_kind`, `argv`,
and `environment`. Their materialization requests additionally bind every
local source by absolute path, positive byte length, and file SHA-256.

The only accepted operation contracts are:

| Artifact kind | Operation kind | First argv item | Fixed key |
|---|---|---|---|
| `QUALIFICATION_CACHE_SEED_INPUT` | `qualification-cache-seed` | `aws/glm52-gpu/scripts/submit_sky_campaign.py` | `task13/qualification/cache-seed-input.json` |
| `H100_QUALIFICATION_INPUT` | `h100-qualification` | `aws/glm52-gpu/scripts/run_h100_qualification_campaign.sh` | `task13/qualification/h100-input.json` |

Each driver materialization request is canonical JSON plus LF with exactly:

```text
schema_version
record_type
account_id
region
profile
run_id
operation_kind
argv
environment
source_coordinates
canonical_identity_sha256
```

Every source coordinate has `role`, absolute `path`, positive `size_bytes`,
and `file_sha256`. Set these paths only after repository-owned builders have
created and reviewed the requests:

```bash
export TASK13_CACHE_DRIVER_REQUEST="$TASK13_WORK/inputs/cache-seed-driver-request.json"
export TASK13_H100_DRIVER_REQUEST="$TASK13_WORK/inputs/h100-driver-request.json"
export TASK13_CACHE_DRIVER_COORDINATE="$TASK13_WORK/coordinates/cache-seed-driver.json"
export TASK13_H100_DRIVER_COORDINATE="$TASK13_WORK/coordinates/h100-driver.json"
```

Bind the cache-seed request only after these exact local sources have been
created and reviewed. The staged-ready VersionId is the real opaque S3
VersionId returned by its publisher:

```bash
export TASK13_CAMPAIGN_DESCRIPTOR=replace-with-absolute-canonical-descriptor
export TASK13_CACHE_APPROVAL=replace-with-absolute-canonical-approval
export TASK13_STAGED_READY=replace-with-absolute-canonical-staged-ready
export TASK13_STAGED_READY_VERSION_ID=replace-with-real-opaque-version-id
export TASK13_REHEARSAL_EVIDENCE=replace-with-absolute-canonical-evidence
export TASK13_CACHE_TASK=replace-with-absolute-task-yaml
export TASK13_CACHE_CONFIG=replace-with-absolute-config
export TASK13_SKY_BIN=replace-with-absolute-sky-executable

test ! -e "$TASK13_CACHE_DRIVER_REQUEST"
python3 \
  aws/glm52-gpu/scripts/build_glm52_task13_driver_request.py \
  cache-seed \
  --campaign-descriptor "$TASK13_CAMPAIGN_DESCRIPTOR" \
  --approval "$TASK13_CACHE_APPROVAL" \
  --staged-ready "$TASK13_STAGED_READY" \
  --staged-ready-version-id "$TASK13_STAGED_READY_VERSION_ID" \
  --rehearsal-evidence "$TASK13_REHEARSAL_EVIDENCE" \
  --task "$TASK13_CACHE_TASK" \
  --config "$TASK13_CACHE_CONFIG" \
  --sky-bin "$TASK13_SKY_BIN" \
  --output "$TASK13_CACHE_DRIVER_REQUEST"

test ! -e "$TASK13_H100_DRIVER_REQUEST"
python3 \
  aws/glm52-gpu/scripts/build_glm52_task13_driver_request.py \
  h100 \
  --campaign-descriptor "$TASK13_CAMPAIGN_DESCRIPTOR" \
  --output "$TASK13_H100_DRIVER_REQUEST"
```

After explicit S3-publication authorization and a fresh credential guard:

```bash
test ! -e "$TASK13_CACHE_DRIVER_COORDINATE"
python3 \
  aws/glm52-gpu/scripts/materialize_glm52_task13_fixed_artifacts.py \
  publish-driver \
  --request "$TASK13_CACHE_DRIVER_REQUEST" \
  --bucket "$TASK13_BUCKET" \
  --coordinate-output "$TASK13_CACHE_DRIVER_COORDINATE"

test ! -e "$TASK13_H100_DRIVER_COORDINATE"
python3 \
  aws/glm52-gpu/scripts/materialize_glm52_task13_fixed_artifacts.py \
  publish-driver \
  --request "$TASK13_H100_DRIVER_REQUEST" \
  --bucket "$TASK13_BUCKET" \
  --coordinate-output "$TASK13_H100_DRIVER_COORDINATE"
```

Close those two published driver coordinates and the activation-scoped
archive into one fresh three-coordinate source before assembling
prequalification:

```bash
export TASK13_FRESH_COORDINATES="$TASK13_WORK/coordinates/fresh-archive-drivers.json"
test ! -e "$TASK13_FRESH_COORDINATES"
python3 \
  aws/glm52-gpu/scripts/build_glm52_fresh_archive_prequalification_coordinates.py \
  --activation-id "$TASK13_ACTIVATION_ID" \
  --repository-archive "$TASK13_WORK/repo.tar.gz" \
  --coordinator "$TASK13_ROOT/aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py" \
  --h100-campaign-descriptor "$TASK13_CAMPAIGN_DESCRIPTOR" \
  --cache-seed-campaign-descriptor "$TASK13_CAMPAIGN_DESCRIPTOR" \
  --approval "$TASK13_CACHE_APPROVAL" \
  --staged-ready "$TASK13_STAGED_READY" \
  --staged-ready-version-id "$TASK13_STAGED_READY_VERSION_ID" \
  --rehearsal-evidence "$TASK13_REHEARSAL_EVIDENCE" \
  --task "$TASK13_CACHE_TASK" \
  --config "$TASK13_CACHE_CONFIG" \
  --sky-bin "$TASK13_SKY_BIN" \
  --repository-archive-coordinate "$TASK13_ARCHIVE_COORDINATE" \
  --repository-archive-manifest "$TASK13_ARCHIVE_MANIFEST" \
  --h100-driver-coordinate "$TASK13_H100_DRIVER_COORDINATE" \
  --cache-seed-driver-coordinate "$TASK13_CACHE_DRIVER_COORDINATE" \
  --output "$TASK13_FRESH_COORDINATES"
```

The support input has a separate exact local materializer and then the same
fixed transport:

```bash
export TASK13_SUPPORT_INPUT_REQUEST="$TASK13_WORK/inputs/support-input-request.json"
export TASK13_SUPPORT_INPUT="$TASK13_WORK/inputs/support-build-inputs.json"
export TASK13_SUPPORT_INPUT_COORDINATE="$TASK13_WORK/coordinates/support-build-inputs.json"

test ! -e "$TASK13_SUPPORT_INPUT"
python3 \
  aws/glm52-gpu/scripts/materialize_glm52_task13_support_inputs.py \
  --request "$TASK13_SUPPORT_INPUT_REQUEST" \
  --output "$TASK13_SUPPORT_INPUT"

test ! -e "$TASK13_SUPPORT_INPUT_COORDINATE"
python3 \
  aws/glm52-gpu/scripts/materialize_glm52_task13_fixed_artifacts.py \
  publish-support-inputs \
  --inputs "$TASK13_SUPPORT_INPUT" \
  --bucket "$TASK13_BUCKET" \
  --coordinate-output "$TASK13_SUPPORT_INPUT_COORDINATE"
```

Both builders exact-read their local sources, pin byte length and SHA-256,
emit canonical JSON plus LF with mode `0600`, and refuse replacement. Do not
hand-convert driver argv or environment into an S3 coordinate.

## 6. Build the prequalification package

The base request is canonical JSON plus LF and has exactly these fields,
omitting `artifacts`:

```text
schema_version
record_type
account_id
region
profile
run_id
activation_id
collector_version_arn
task11_request
task11_boundary
staged_infrastructure_evidence
retained_stack_id
fence_stack_name
support_stack_name
fence_change_set_name
support_change_set_name
cloudformation_role_arn
monitor_descriptor_path
```

The constants are account `246813579024`, region `us-west-2`, profile
`keep-gpu`, run ID `glm52-sky-20260724`, and deployment role
`arn:aws:iam::246813579024:role/keep-glm52-h1g-cloudformation-deployment`.
The retained stack ID, collector version ARN, activation ID, change-set names,
Task 11 records, and monitor descriptor path are reviewed live inputs; never
fill them with examples from tests or old runs.

Set these paths only after their contents exist and have been independently
reviewed:

```bash
export TASK13_BASE_REQUEST="$TASK13_WORK/inputs/base-request.json"
export TASK13_PREQUAL_COORDINATES="$TASK13_WORK/coordinates/prequalification-21.json"
export TASK13_PREQUAL_REQUEST="$TASK13_WORK/inputs/prequalification-request.json"
export TASK13_PREQUAL_REVIEWED="$TASK13_WORK/inputs/prequalification-reviewed.json"
export TASK13_PREQUAL_PACKAGE="$TASK13_WORK/packages/prequalification-package.json"
```

Set the following to the exact canonical local Task 11 records and the exact
coordinate output from each authorized publisher. The transport-gate output
is one canonical two-coordinate array; every other source is one coordinate
object:

```bash
export TASK13_TASK11_REQUEST=replace-with-absolute-canonical-task11-request
export TASK13_TASK11_BOUNDARY=replace-with-absolute-canonical-task11-boundary
export TASK13_STAGED_INFRASTRUCTURE_EVIDENCE=replace-with-absolute-canonical-staged-infrastructure-evidence
export TASK13_COLLECTOR_VERSION_ARN=replace-with-pinned-collector-version-arn
export TASK13_RETAINED_STACK_ID=replace-with-live-retained-stack-id
export TASK13_FENCE_CHANGE_SET_NAME=replace-with-reviewed-fence-change-set
export TASK13_SUPPORT_CHANGE_SET_NAME=replace-with-reviewed-support-change-set
export TASK13_MONITOR_DESCRIPTOR="$TASK13_CAMPAIGN_DESCRIPTOR"

export TASK13_TASK11_REVIEW_COORDINATE=replace-with-absolute-coordinate
export TASK13_TASK12_REVIEW_COORDINATE=replace-with-absolute-coordinate
export TASK13_RETAINED_FOUNDATION_COORDINATE=replace-with-absolute-coordinate
export TASK13_RETAINED_PRE_SUPPORT_COORDINATE=replace-with-absolute-coordinate
export TASK13_RETAINED_TEMPLATE_COORDINATE=replace-with-absolute-coordinate
export TASK13_FENCE_TEMPLATE_COORDINATE=replace-with-absolute-coordinate
export TASK13_SUPPORT_TEMPLATE_COORDINATE=replace-with-absolute-coordinate
export TASK13_SUPPORT_INPUT_COORDINATE=replace-with-absolute-coordinate
export TASK13_BOOTSTRAP_COORDINATE=replace-with-absolute-coordinate
export TASK13_TRANSPORT_GATES_COORDINATES=replace-with-absolute-coordinate-array
export TASK13_ACCEPTED_BASELINE_COORDINATE=replace-with-absolute-coordinate
export TASK13_PROMPT_PACK_COORDINATE=replace-with-absolute-coordinate
export TASK13_TRAINING_CONFIG_COORDINATE=replace-with-absolute-coordinate
export TASK13_GPU_APPROVAL_COORDINATE=replace-with-absolute-coordinate
export TASK13_SUPPORT_APPROVAL_COORDINATE=replace-with-absolute-coordinate
export TASK13_RESIDUAL_APPROVAL_COORDINATE=replace-with-absolute-coordinate
export TASK13_PRODUCTION_DESCRIPTOR_COORDINATE=replace-with-absolute-coordinate

test ! -e "$TASK13_BASE_REQUEST"
test ! -e "$TASK13_PREQUAL_COORDINATES"
test ! -e "$TASK13_PREQUAL_REQUEST"
test ! -e "$TASK13_PREQUAL_REVIEWED"
python3 \
  aws/glm52-gpu/scripts/build_glm52_task13_prequalification_sources.py \
  --activation-id "$TASK13_ACTIVATION_ID" \
  --collector-version-arn "$TASK13_COLLECTOR_VERSION_ARN" \
  --task11-request "$TASK13_TASK11_REQUEST" \
  --task11-boundary "$TASK13_TASK11_BOUNDARY" \
  --staged-infrastructure-evidence "$TASK13_STAGED_INFRASTRUCTURE_EVIDENCE" \
  --retained-stack-id "$TASK13_RETAINED_STACK_ID" \
  --fence-change-set-name "$TASK13_FENCE_CHANGE_SET_NAME" \
  --support-change-set-name "$TASK13_SUPPORT_CHANGE_SET_NAME" \
  --monitor-descriptor-path "$TASK13_MONITOR_DESCRIPTOR" \
  --coordinate-source "$TASK13_TASK11_REVIEW_COORDINATE" \
  --coordinate-source "$TASK13_TASK12_REVIEW_COORDINATE" \
  --coordinate-source "$TASK13_RETAINED_FOUNDATION_COORDINATE" \
  --coordinate-source "$TASK13_RETAINED_PRE_SUPPORT_COORDINATE" \
  --coordinate-source "$TASK13_RETAINED_TEMPLATE_COORDINATE" \
  --coordinate-source "$TASK13_FENCE_TEMPLATE_COORDINATE" \
  --coordinate-source "$TASK13_SUPPORT_TEMPLATE_COORDINATE" \
  --coordinate-source "$TASK13_SUPPORT_INPUT_COORDINATE" \
  --coordinate-source "$TASK13_BOOTSTRAP_COORDINATE" \
  --coordinate-source "$TASK13_FRESH_COORDINATES" \
  --coordinate-source "$TASK13_TRANSPORT_GATES_COORDINATES" \
  --coordinate-source "$TASK13_ACCEPTED_BASELINE_COORDINATE" \
  --coordinate-source "$TASK13_PROMPT_PACK_COORDINATE" \
  --coordinate-source "$TASK13_TRAINING_CONFIG_COORDINATE" \
  --coordinate-source "$TASK13_GPU_APPROVAL_COORDINATE" \
  --coordinate-source "$TASK13_SUPPORT_APPROVAL_COORDINATE" \
  --coordinate-source "$TASK13_RESIDUAL_APPROVAL_COORDINATE" \
  --coordinate-source "$TASK13_PRODUCTION_DESCRIPTOR_COORDINATE" \
  --base-request-output "$TASK13_BASE_REQUEST" \
  --coordinates-output "$TASK13_PREQUAL_COORDINATES" \
  --request-output "$TASK13_PREQUAL_REQUEST" \
  --reviewed-artifacts-output "$TASK13_PREQUAL_REVIEWED"

test ! -e "$TASK13_PREQUAL_PACKAGE"
python3 \
  aws/glm52-gpu/scripts/build_glm52_task13_campaign_package.py \
  --request "$TASK13_PREQUAL_REQUEST" \
  --output "$TASK13_PREQUAL_PACKAGE"
```

## 7. Mandatory no-launch validation

This is the only runner invocation that must not receive authority or journal
arguments. It constructs inert objects for `services` and `journal`; it does
not invoke the production coordinator or AWS:

```bash
python3 \
  aws/glm52-gpu/scripts/run_glm52_task13_campaign.py \
  --package "$TASK13_PREQUAL_PACKAGE" \
  --reviewed-artifacts "$TASK13_PREQUAL_REVIEWED" \
  --stage validate
```

Require exact result fields with:

```text
status = VALIDATED_NO_EXTERNAL_EXECUTION
stage = validate
package_identity_sha256 = the package canonical identity
reviewed_artifacts_identity_sha256 = the canonical reviewed-list identity
canonical_identity_sha256 = the canonical result identity
```

Any `--authority`, `--fence-journal`, `--support-journal`, or
`--campaign-journal` argument makes validation fail closed. A successful
validation does not authorize or prove an external stage.

## 8. Stage-scoped controller authority

The issuer pins the exact local owner-approval bytes and the executable
coordinator bytes. Compute their identities from the current files; do not
copy a hash from this document:

```bash
export TASK13_OWNER_APPROVAL="$TASK13_ROOT/docs/superpowers/approvals/2026-07-28-glm52-campaign-owner-approvals.md"
export TASK13_COORDINATOR="$TASK13_ROOT/aws/glm52-gpu/scripts/glm52_task13_production_coordinator.py"
export TASK13_OWNER_APPROVAL_SHA256
TASK13_OWNER_APPROVAL_SHA256=$(shasum -a 256 "$TASK13_OWNER_APPROVAL" | awk '{print $1}')
export TASK13_COORDINATOR_SHA256
TASK13_COORDINATOR_SHA256=$(shasum -a 256 "$TASK13_COORDINATOR" | awk '{print $1}')
test -x "$TASK13_COORDINATOR"
```

For each stage below, issue a new private authority immediately before that
one invocation:

```bash
export TASK13_STAGE=replace-with-one-exact-stage
export TASK13_STAGE_PACKAGE=replace-with-absolute-package-path
export TASK13_STAGE_REVIEWED=replace-with-absolute-reviewed-path
export TASK13_STAGE_AUTHORITY=replace-with-new-absolute-authority-path

test ! -e "$TASK13_STAGE_AUTHORITY"
python3 \
  aws/glm52-gpu/scripts/issue_glm52_task13_controller_authority.py \
  --package "$TASK13_STAGE_PACKAGE" \
  --reviewed-artifacts "$TASK13_STAGE_REVIEWED" \
  --stage "$TASK13_STAGE" \
  --ttl-seconds 900 \
  --expected-owner-approval-sha256 \
  "$TASK13_OWNER_APPROVAL_SHA256" \
  --expected-coordinator-sha256 \
  "$TASK13_COORDINATOR_SHA256" \
  --full-run-work "$FULL_RUN_WORK" \
  --signing-private-key "$TASK13_SIGNING_PRIVATE_KEY" \
  --output "$TASK13_STAGE_AUTHORITY"

python3 \
  aws/glm52-gpu/scripts/run_glm52_task13_campaign.py \
  --package "$TASK13_STAGE_PACKAGE" \
  --reviewed-artifacts "$TASK13_STAGE_REVIEWED" \
  --stage "$TASK13_STAGE" \
  --authority "$TASK13_STAGE_AUTHORITY" \
  --full-run-work "$FULL_RUN_WORK" \
  --signing-private-key "$TASK13_SIGNING_PRIVATE_KEY" \
  --fence-journal "$TASK13_FENCE_JOURNAL" \
  --support-journal "$TASK13_SUPPORT_JOURNAL" \
  --campaign-journal "$TASK13_CAMPAIGN_JOURNAL"
```

The TTL must be between 60 and 900 seconds. Use 900 seconds here. The issuer
sets exactly one stage grant:

| Stage | Exact positive grant |
|---|---|
| `deploy-disabled` | change-set execution |
| `collect-first-five` | change-set execution |
| `collect-remaining` | change-set execution |
| `finalize` | change-set execution |
| `qualification-cache-seed` | qualification-cache seed |
| `h100-qualification` | H100 qualification |
| `launch` | production launch |
| `monitor` | production launch, used only for guarded inspection |
| `terminal` | production launch, used only for guarded terminal inspection |

Support-plane and residual-liability approvals are true in every issued
authority. A prequalification package cannot receive `launch`, `monitor`, or
`terminal` authority.

## 9. Deploy disabled and collect the rehearsal gate

Use the prequalification package and reviewed-artifact list for all six
preproduction stages across sections 9 through 11.

### 9.1 Deploy disabled

This is a live CloudFormation/support-plane mutation. Obtain explicit
live-change authorization, rerun the credential guard, then:

```bash
export TASK13_STAGE=deploy-disabled
export TASK13_STAGE_PACKAGE="$TASK13_PREQUAL_PACKAGE"
export TASK13_STAGE_REVIEWED="$TASK13_PREQUAL_REVIEWED"
export TASK13_STAGE_AUTHORITY="$TASK13_WORK/authorities/deploy-disabled.json"
```

Run the two issuer/runner commands from section 8. Require
`status=STAGE_COMMITTED` and `stage=deploy-disabled`.

This stage performs the migration bootstrap, creates/inspects/executes only
the package-pinned disabled change sets, materializes precreate and
postcreate Task 12 orphan authority, proves exact stack/template/resource
readback, runs the negative IAM probes, and proves zero active P5 instances.
It does not authorize a cache, qualification, or production worker.

### 9.2 First five rehearsal measurements

```bash
export TASK13_STAGE=collect-first-five
export TASK13_STAGE_PACKAGE="$TASK13_PREQUAL_PACKAGE"
export TASK13_STAGE_REVIEWED="$TASK13_PREQUAL_REVIEWED"
export TASK13_STAGE_AUTHORITY="$TASK13_WORK/authorities/collect-first-five.json"
```

Run section 8. This stage requires `deploy-disabled` to be durably committed
and collects measurement IDs 1 through 5.

### 9.3 Remaining fifteen rehearsal measurements

```bash
export TASK13_STAGE=collect-remaining
export TASK13_STAGE_PACKAGE="$TASK13_PREQUAL_PACKAGE"
export TASK13_STAGE_REVIEWED="$TASK13_PREQUAL_REVIEWED"
export TASK13_STAGE_AUTHORITY="$TASK13_WORK/authorities/collect-remaining.json"
```

Run section 8. This stage requires `collect-first-five` and collects
measurement IDs 6 through 20.

### 9.4 Finalize the rehearsal gate

```bash
export TASK13_STAGE=finalize
export TASK13_STAGE_PACKAGE="$TASK13_PREQUAL_PACKAGE"
export TASK13_STAGE_REVIEWED="$TASK13_PREQUAL_REVIEWED"
export TASK13_STAGE_AUTHORITY="$TASK13_WORK/authorities/finalize.json"
export TASK13_FINALIZE_INVOKE_METADATA="$TASK13_WORK/results/finalize-invoke-metadata.json"
export TASK13_FINALIZE_PAYLOAD="$TASK13_WORK/results/finalize-payload.json"
export TASK13_GATE_READBACK="$TASK13_WORK/results/finalize-gate-readback.json"
test ! -e "$TASK13_FINALIZE_INVOKE_METADATA"
test ! -e "$TASK13_FINALIZE_PAYLOAD"
test ! -e "$TASK13_GATE_READBACK"
```

Run only the authority-issuer command from section 8, then use this exact
runner invocation instead of the generic runner command:

```bash
python3 \
  aws/glm52-gpu/scripts/run_glm52_task13_campaign.py \
  --package "$TASK13_STAGE_PACKAGE" \
  --reviewed-artifacts "$TASK13_STAGE_REVIEWED" \
  --stage "$TASK13_STAGE" \
  --authority "$TASK13_STAGE_AUTHORITY" \
  --full-run-work "$FULL_RUN_WORK" \
  --signing-private-key "$TASK13_SIGNING_PRIVATE_KEY" \
  --fence-journal "$TASK13_FENCE_JOURNAL" \
  --support-journal "$TASK13_SUPPORT_JOURNAL" \
  --campaign-journal "$TASK13_CAMPAIGN_JOURNAL" \
  --finalize-invoke-metadata-output "$TASK13_FINALIZE_INVOKE_METADATA" \
  --finalize-payload-output "$TASK13_FINALIZE_PAYLOAD" \
  --gate-readback-output "$TASK13_GATE_READBACK"
```

This stage rereads all 20 immutable measurements, requires five distinct cold
environments, invokes the pinned finalizer exactly once, and accepts only a
closure-budget gate that binds the inspected measurement set. Require
`status=STAGE_COMMITTED` before using the three create-only capture files.

### 9.5 Build and publish activation-scoped CLEAN evidence

Build CLEAN evidence only after the disabled finalizer is committed. The
builder binds the exact 21-coordinate predecessor package/list, activation,
unchanged archive coordinate and manifest bytes, exact finalizer result, and
immutable gate readback:

```bash
export TASK13_CLEAN_REHEARSAL_EVIDENCE="$TASK13_WORK/results/clean-rehearsal-evidence.json"
export TASK13_CLEAN_REHEARSAL_COORDINATE="$TASK13_WORK/coordinates/clean-rehearsal.json"
test ! -e "$TASK13_CLEAN_REHEARSAL_EVIDENCE"
python3 \
  aws/glm52-gpu/scripts/build_glm52_task13_clean_rehearsal.py \
  --predecessor-package "$TASK13_PREQUAL_PACKAGE" \
  --predecessor-reviewed-artifacts "$TASK13_PREQUAL_REVIEWED" \
  --repository-archive-coordinate "$TASK13_ARCHIVE_COORDINATE" \
  --repository-archive-manifest "$TASK13_ARCHIVE_MANIFEST" \
  --finalize-invoke-metadata "$TASK13_FINALIZE_INVOKE_METADATA" \
  --finalize-payload "$TASK13_FINALIZE_PAYLOAD" \
  --gate-readback "$TASK13_GATE_READBACK" \
  --output "$TASK13_CLEAN_REHEARSAL_EVIDENCE"
```

Independently review the canonical evidence bytes and set both exact hashes:

```bash
export TASK13_CLEAN_FILE_SHA256=replace-with-reviewed-file-sha256
export TASK13_CLEAN_BODY_SHA256=replace-with-reviewed-body-sha256
```

After explicit publication authorization and the credential guard, publish
only through the dynamic reviewed publisher. The publisher derives
`task13/gates/clean-rehearsal/$TASK13_ACTIVATION_ID/<self-hash>.json` from the
validated evidence; there is no fixed CLEAN key or copy/adoption shim.

```bash
test ! -e "$TASK13_CLEAN_REHEARSAL_COORDINATE"
python3 \
  aws/glm52-gpu/scripts/publish_glm52_task13_reviewed_artifact.py \
  --artifact-kind CLEAN_REHEARSAL \
  --source "$TASK13_CLEAN_REHEARSAL_EVIDENCE" \
  --expected-file-sha256 "$TASK13_CLEAN_FILE_SHA256" \
  --expected-body-sha256 "$TASK13_CLEAN_BODY_SHA256" \
  --bucket "$TASK13_BUCKET" \
  --coordinate-output "$TASK13_CLEAN_REHEARSAL_COORDINATE"
```

## 10. Qualification-cache seed

This stage can start one on-demand `p5.48xlarge` and is not a rehearsal.
Obtain the explicit paid-H100 authorization, rerun the credential guard, and
confirm that `finalize` is committed in the same campaign journal.

```bash
export TASK13_STAGE=qualification-cache-seed
export TASK13_STAGE_PACKAGE="$TASK13_PREQUAL_PACKAGE"
export TASK13_STAGE_REVIEWED="$TASK13_PREQUAL_REVIEWED"
export TASK13_STAGE_AUTHORITY="$TASK13_WORK/authorities/qualification-cache-seed.json"
```

Run section 8. Require `status=STAGE_COMMITTED` and verify the result binds:

- exactly one required teacher row;
- `QUALIFICATION_CACHE_SEED_READY.json`;
- `TEACHER_CACHE_READY.json`;
- the qualification cache prefix and manifest identity;
- zero active instances after teardown;
- no production effects.

If the runner reports a possibly-sent mutation, follow section 14. Do not
invoke `submit_sky_campaign.py` directly and do not create a second cache-seed
submission.

## 11. Real H100 qualification

Obtain explicit paid-H100 authorization again and rerun the credential guard:

```bash
export TASK13_STAGE=h100-qualification
export TASK13_STAGE_PACKAGE="$TASK13_PREQUAL_PACKAGE"
export TASK13_STAGE_REVIEWED="$TASK13_PREQUAL_REVIEWED"
export TASK13_STAGE_AUTHORITY="$TASK13_WORK/authorities/h100-qualification.json"
```

Run the authority-issuer command from section 8, but use the single captured
runner invocation below instead of the generic runner command. Require
`status=STAGE_COMMITTED` and exact proof of:

- source `SOURCE_NODE_READY.json`;
- `QUALIFICATION_TERMINATION_REQUESTED.json`;
- a positively terminated source instance;
- a distinct replacement instance;
- exactly two adapter training steps;
- peak device memory strictly below 70 GiB;
- `H100_RESUME_READY.json`;
- zero active instances and no production effect after qualification.

Capture the canonical H100 stage stdout to a new file when section 8 runs that
stage. The result contains the authenticated `h100_resume_ready` marker, and
the manifest builder below accepts it only when its exact summary identity is
the COMMITTED `stage:h100-qualification` journal result:

```bash
export TASK13_H100_STAGE_RESULT="$TASK13_WORK/results/h100-qualification.json"
export TASK13_H100_STAGE_RESULT_TEMP
if test -e "$TASK13_H100_STAGE_RESULT"
then
  echo "refusing to overwrite H100 stage result" >&2
  exit 1
fi
TASK13_H100_STAGE_RESULT_TEMP=$(mktemp \
  "$TASK13_WORK/results/.h100-qualification.json.XXXXXX")
chmod 600 "$TASK13_H100_STAGE_RESULT_TEMP"
if python3 \
  aws/glm52-gpu/scripts/run_glm52_task13_campaign.py \
  --package "$TASK13_PREQUAL_PACKAGE" \
  --reviewed-artifacts "$TASK13_PREQUAL_REVIEWED" \
  --stage h100-qualification \
  --authority "$TASK13_STAGE_AUTHORITY" \
  --full-run-work "$FULL_RUN_WORK" \
  --signing-private-key "$TASK13_SIGNING_PRIVATE_KEY" \
  --fence-journal "$TASK13_FENCE_JOURNAL" \
  --support-journal "$TASK13_SUPPORT_JOURNAL" \
  --campaign-journal "$TASK13_CAMPAIGN_JOURNAL" \
  > "$TASK13_H100_STAGE_RESULT_TEMP"
then
  if ln "$TASK13_H100_STAGE_RESULT_TEMP" "$TASK13_H100_STAGE_RESULT"
  then
    rm -f "$TASK13_H100_STAGE_RESULT_TEMP"
  else
    TASK13_H100_STATUS=$?
    rm -f "$TASK13_H100_STAGE_RESULT_TEMP"
    exit "$TASK13_H100_STATUS"
  fi
else
  TASK13_H100_STATUS=$?
  rm -f "$TASK13_H100_STAGE_RESULT_TEMP"
  exit "$TASK13_H100_STATUS"
fi
unset TASK13_H100_STAGE_RESULT_TEMP
```

Do not synthesize a local H100 coordinate from the key or from an unversioned
S3 query.

## 12. Build the production successor

The Task 10 authority materializer requires a canonical manifest with these
live inputs:

```text
task11_boundary
approval_coordinates
task10_task_inputs
h100_resume_ready
task10_worker_descriptor
runner_relative_path
coordinator_relative_path
runner_file_sha256
coordinator_file_sha256
workflow_inventory
reviewed_artifacts_without_task10_authority
```

The materializer exact-reads every S3 source, validates the Task 11 boundary
and all of its inputs, validates the H100 marker against the campaign,
archive, and qualification-cache identities, pins the runner and coordinator
bytes, proves the Task 10 state-machine version is active, and conditionally
publishes one fixed authority object. Its reviewed-list output is an
intermediate 22-row list: the 21-row predecessor plus only
`TASK10_PRODUCTION_AUTHORITY`. It is not final production authority.

After explicit authority-materialization authorization and the credential
guard:

```bash
export TASK13_TASK10_INPUTS_COORDINATE=replace-with-absolute-coordinate
export TASK13_TASK10_WORKER_COORDINATE=replace-with-absolute-coordinate
export TASK13_TASK10_MANIFEST="$TASK13_WORK/inputs/task10-authority-materialization.json"
export TASK13_TASK10_JOURNAL="$TASK13_WORK/journals/task10-authority-materialization.jsonl"
export TASK13_TASK10_COORDINATE="$TASK13_WORK/coordinates/task10-production-authority.json"
export TASK13_AUTHORITY_JOINED_REVIEWED="$TASK13_WORK/inputs/authority-joined-reviewed.json"
export TASK13_PRODUCTION_REVIEWED="$TASK13_WORK/inputs/production-reviewed-25.json"
export TASK13_PUBLIC_TASK10_DIR="$TASK13_WORK/results/public-task10"
mkdir -m 700 "$TASK13_PUBLIC_TASK10_DIR"

test ! -e "$TASK13_TASK10_MANIFEST"
python3 \
  aws/glm52-gpu/scripts/build_glm52_task10_materialization_manifest.py \
  --prequalification-package "$TASK13_PREQUAL_PACKAGE" \
  --prequalification-reviewed-artifacts "$TASK13_PREQUAL_REVIEWED" \
  --task10-task-inputs "$TASK13_TASK10_INPUTS_COORDINATE" \
  --task10-worker-descriptor "$TASK13_TASK10_WORKER_COORDINATE" \
  --h100-stage-result "$TASK13_H100_STAGE_RESULT" \
  --campaign-journal "$TASK13_CAMPAIGN_JOURNAL" \
  --output "$TASK13_TASK10_MANIFEST"

test ! -e "$TASK13_TASK10_COORDINATE"
test ! -e "$TASK13_AUTHORITY_JOINED_REVIEWED"
python3 \
  aws/glm52-gpu/scripts/materialize_glm52_task10_authority.py \
  --manifest "$TASK13_TASK10_MANIFEST" \
  --journal "$TASK13_TASK10_JOURNAL" \
  --output-coordinate "$TASK13_TASK10_COORDINATE" \
  --output-reviewed-artifacts "$TASK13_AUTHORITY_JOINED_REVIEWED" \
  --public-output-dir "$TASK13_PUBLIC_TASK10_DIR"
```

The manifest builder validates the 21-row prequalification package/list, the
canonical H100 result, the full journal chains, the COMMITTED H100 summary
identity, fixed workflow inventory, and current runner/coordinator bytes.

Create the final list from the exact 21-row predecessor and the four exact
production additions. The repository-owned joiner validates every coordinate,
preserves every predecessor byte, sorts canonically, and writes one
create-only 25-row file:

```bash
test ! -e "$TASK13_PRODUCTION_REVIEWED"
python3 \
  aws/glm52-gpu/scripts/build_glm52_production_reviewed_artifacts.py \
  --prequalification-reviewed-artifacts "$TASK13_PREQUAL_REVIEWED" \
  --clean-rehearsal-coordinate "$TASK13_CLEAN_REHEARSAL_COORDINATE" \
  --task10-production-authority "$TASK13_TASK10_COORDINATE" \
  --task10-worker-descriptor "$TASK13_TASK10_WORKER_COORDINATE" \
  --task10-task-inputs "$TASK13_TASK10_INPUTS_COORDINATE" \
  --output "$TASK13_PRODUCTION_REVIEWED"
```

The production request contains those exact 25 rows: zero predecessor
changes and exactly four additions. Both successor builders require the same
mandatory CLEAN evidence:

```bash
export TASK13_PRODUCTION_REQUEST="$TASK13_WORK/inputs/production-request.json"
export TASK13_PRODUCTION_PACKAGE="$TASK13_WORK/packages/production-package.json"
test ! -e "$TASK13_PRODUCTION_REQUEST"
python3 \
  aws/glm52-gpu/scripts/build_glm52_task13_production_successor_request.py \
  --prequalification-request "$TASK13_PREQUAL_REQUEST" \
  --prequalification-package "$TASK13_PREQUAL_PACKAGE" \
  --prequalification-reviewed-artifacts "$TASK13_PREQUAL_REVIEWED" \
  --production-reviewed-artifacts "$TASK13_PRODUCTION_REVIEWED" \
  --clean-rehearsal-evidence "$TASK13_CLEAN_REHEARSAL_EVIDENCE" \
  --output "$TASK13_PRODUCTION_REQUEST"

test ! -e "$TASK13_PRODUCTION_PACKAGE"
python3 \
  aws/glm52-gpu/scripts/build_glm52_task13_campaign_package.py \
  --request "$TASK13_PRODUCTION_REQUEST" \
  --predecessor-package "$TASK13_PREQUAL_PACKAGE" \
  --predecessor-reviewed-artifacts "$TASK13_PREQUAL_REVIEWED" \
  --clean-rehearsal-evidence "$TASK13_CLEAN_REHEARSAL_EVIDENCE" \
  --output "$TASK13_PRODUCTION_PACKAGE"
```

Repeat the pure no-launch validation before any production authority:

```bash
python3 \
  aws/glm52-gpu/scripts/run_glm52_task13_campaign.py \
  --package "$TASK13_PRODUCTION_PACKAGE" \
  --reviewed-artifacts "$TASK13_PRODUCTION_REVIEWED" \
  --stage validate
```

Require `VALIDATED_NO_EXTERNAL_EXECUTION`.

## 13. Production launch

Before launch, independently confirm:

- the production package passed pure validation;
- all six preproduction stages are committed in the unchanged campaign
  journal;
- the exact reviewed immutable inputs still read back by `VersionId`;
- zero active P5 instances;
- more than 3600 approved GPU seconds remain;
- current credentials have at least 3600 seconds remaining;
- no existing `launch:task10-workflow` journal operation belongs to another
  package or request.

Then obtain explicit production-launch authorization and issue the fresh
authority:

```bash
export TASK13_STAGE=launch
export TASK13_STAGE_PACKAGE="$TASK13_PRODUCTION_PACKAGE"
export TASK13_STAGE_REVIEWED="$TASK13_PRODUCTION_REVIEWED"
export TASK13_STAGE_AUTHORITY="$TASK13_WORK/authorities/launch.json"
```

Run section 8 exactly once. The runner:

1. exact-reads the 12 production immutable inputs;
2. rereads the committed cache and H100 results;
3. proves zero active P5 instances;
4. materializes the sole-sender authority;
5. materializes one attempt-01 launch authority;
6. starts the versioned Task 10 production workflow at most once;
7. immediately reconciles the workflow result;
8. verifies the immutable `LAUNCH_OUTCOME.json` coordinate;
9. commits the launch stage only after a positive accepted result.

The closed capacity plan is one on-demand `p5.48xlarge`, ordered
`us-west-2a` through `us-west-2f`, stop on first success, and no Spot or
Capacity Block fallback.

### Optional Task 10 CLI bridge

`build_glm52_task13_launch_bridge.py` pins the production package, reviewed
list, launch authority, H100 coordinate, route-binding identity, runner,
coordinator, and the same three journals. It is required only when the public
Task 10 `submit_sky_campaign.py production ...` entrypoint is used. The
Task 13 runner above already owns the guarded production route and must not be
invoked in parallel with that public entrypoint.

The materializer writes one cross-bound local manifest plus the canonical
authority, task inputs, worker descriptor, H100 coordinate, and route identity
with mode `0600`. Build the bridge from that one manifest:

```bash
export TASK13_PUBLIC_TASK10_INPUTS="$TASK13_PUBLIC_TASK10_DIR/public-task10-inputs.json"
export TASK13_LAUNCH_BRIDGE="$TASK13_WORK/inputs/task13-launch-bridge.json"

test ! -e "$TASK13_LAUNCH_BRIDGE"
python3 \
  aws/glm52-gpu/scripts/build_glm52_task13_launch_bridge.py \
  --package "$TASK13_PRODUCTION_PACKAGE" \
  --reviewed-artifacts "$TASK13_PRODUCTION_REVIEWED" \
  --controller-authority "$TASK13_STAGE_AUTHORITY" \
  --public-task10-inputs "$TASK13_PUBLIC_TASK10_INPUTS" \
  --fence-journal "$TASK13_FENCE_JOURNAL" \
  --support-journal "$TASK13_SUPPORT_JOURNAL" \
  --campaign-journal "$TASK13_CAMPAIGN_JOURNAL" \
  --output "$TASK13_LAUNCH_BRIDGE"
```

If the public route is explicitly selected instead of the Task 13 runner,
invoke it with the emitted local files; never run both routes:

```bash
export TASK13_PUBLIC_HANDOFF="$TASK13_WORK/results/public-task10-handoff.json"
test ! -e "$TASK13_PUBLIC_HANDOFF"
python3 \
  aws/glm52-gpu/scripts/submit_sky_campaign.py \
  production start \
  --profile keep-gpu \
  --descriptor "$TASK13_PUBLIC_TASK10_DIR/task10-worker-descriptor.json" \
  --production-authority \
  "$TASK13_PUBLIC_TASK10_DIR/task10-production-authority.json" \
  --production-task-inputs \
  "$TASK13_PUBLIC_TASK10_DIR/task10-task-inputs.json" \
  --task13-launch-bridge "$TASK13_LAUNCH_BRIDGE" \
  --output-handoff "$TASK13_PUBLIC_HANDOFF"
```

## 14. Reconcile without replay

The Task 13 runner journals each mutation as:

```text
PREPARED -> POSSIBLY_SENT -> COMMITTED
```

The one-shot retained-foundation operation is outside the stage journals. If
its CloudFormation update commits but final local evidence emission fails, do
not rerun the apply route. Use
`apply_glm52_task13_retained_foundation.py --recover-change-set-evidence`
with the exact canonical `retained-change-set-evidence.json` from that
operation and a dedicated evidence directory whose exact recovery-readback
output path does not exist. Recovery refuses unless
CloudTrail contains one successful `ExecuteChangeSet` management event bound
to the archived change-set and stack ARNs. CloudTrail delivery can lag by
approximately 15 minutes, so an immediate refusal is safe to retry. Standard
event-history lookup retains only 90 days; after that window recovery refuses
permanently rather than inferring historical execution from current stack
state.

If any external stage exits nonzero after `POSSIBLY_SENT`, the only permitted
recovery is:

1. preserve the exact package, reviewed list, coordinator executable, and all
   three journals;
2. do not run the underlying AWS, Lambda, SkyPilot, or Step Functions command;
3. rerun the credential guard;
4. issue a new authority for the same stage to a new output path;
5. rerun the same stage with the same package, reviewed list, and journals.

For example, an ambiguous launch is reconciled with:

```bash
export TASK13_STAGE=launch
export TASK13_STAGE_PACKAGE="$TASK13_PRODUCTION_PACKAGE"
export TASK13_STAGE_REVIEWED="$TASK13_PRODUCTION_REVIEWED"
export TASK13_STAGE_AUTHORITY="$TASK13_WORK/authorities/launch-reconcile-001.json"
```

Then run section 8. The existing `POSSIBLY_SENT` journal row forces the
coordinator's `reconcile` action; it does not call the launch execute path
again. A committed stage is reread and must have the same result identity.

Do not invoke `submit_sky_campaign.py production reconcile` as a second,
parallel recovery route. The Task 13 stage replay is the journal-authoritative
reconcile path.

## 15. Monitor

`monitor` requires the production package, a fresh `monitor` authority, and a
committed launch stage. It is read-only but remains behind the same
production-authority and coordinator boundary.

For one observation:

```bash
export TASK13_STAGE=monitor
export TASK13_STAGE_PACKAGE="$TASK13_PRODUCTION_PACKAGE"
export TASK13_STAGE_REVIEWED="$TASK13_PRODUCTION_REVIEWED"
export TASK13_STAGE_AUTHORITY="$TASK13_WORK/authorities/monitor-001.json"
```

Run section 8. The only accepted statuses are `STARTING`, `RUNNING`,
`SUCCEEDED`, or `FAILED`, with a list of syntactically valid instance IDs.

Do not put monitoring in an unbounded shell loop. For another observation,
issue another `monitor` authority with a new file name. On `FAILED`, stop and
preserve evidence. `SUCCEEDED` is necessary but not sufficient for terminal
closure.

## 16. Terminal verification

The terminal stage requires all of the following generation-`00000001`
versioned objects:

```text
campaigns/glm52-sky-20260724/submissions/production/generations/00000001/terminal/TASK13_TERMINAL_PROOF.json
campaigns/glm52-sky-20260724/submissions/production/generations/00000001/terminal/CAMPAIGN_DRAINED.json
campaigns/glm52-sky-20260724/submissions/production/generations/00000001/terminal/TERMINAL_VERIFIED.json
```

It also requires Sky status `SUCCEEDED`, zero billable P5 instances, an exact
allocation ledger whose seconds and dollars reconcile to the fixed approval,
and all retained-cost categories.

The repository-owned terminal publisher exact-reads the singular immutable
`CAMPAIGN_DRAINED.json`, builds and conditionally publishes
`TERMINAL_VERIFIED.json`, exact-reads it, then builds and conditionally
publishes `TASK13_TERMINAL_PROOF.json`. An ambiguous put is resolved through
complete version inventory and byte-identical readback, never a second put.

Its terminal-state input must be canonical JSON plus LF with the exact legacy
terminal-verification schema, authenticated `DRAINED` phase, and an allowed
outcome of `completed`, `training_deferred`, or `resumable_deadline`. Its
settlement input must contain exactly:

```text
allocations
total_gpu_seconds
total_gpu_cost_usd
remaining_gpu_seconds
remaining_gpu_usd
retained_costs
```

Allocation totals must sum to the fixed 86,400-second and `$1,320.96`
approval. Retained costs must include controller, storage, checksum, network,
Lambda, logs, workflow, queue, notification, root volume, and incident tail.

After Sky reports `SUCCEEDED`, the exact P5 inventory is empty,
`CAMPAIGN_DRAINED.json` exists at the key above, and the two canonical local
inputs have been independently reviewed:

```bash
export TASK13_TERMINAL_STATE="$TASK13_WORK/inputs/terminal-state.json"
export TASK13_TERMINAL_SETTLEMENT="$TASK13_WORK/inputs/terminal-settlement.json"
export TASK13_TERMINAL_PUBLICATION="$TASK13_WORK/results/terminal-publication.json"
export TASK13_ACTIVATION_ID
TASK13_ACTIVATION_ID=$(python3 - "$TASK13_PRODUCTION_PACKAGE" <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_bytes())
activation_id = value.get("activation_id")
if not isinstance(activation_id, str) or not activation_id:
    raise SystemExit("production package activation_id is absent")
print(activation_id)
PY
)
export TASK13_VERIFIED_AT
TASK13_VERIFIED_AT=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

test ! -e "$TASK13_TERMINAL_PUBLICATION"
python3 \
  aws/glm52-gpu/scripts/publish_glm52_task13_terminal_evidence.py \
  --profile keep-gpu \
  --activation-id "$TASK13_ACTIVATION_ID" \
  --terminal-state "$TASK13_TERMINAL_STATE" \
  --settlement "$TASK13_TERMINAL_SETTLEMENT" \
  --verified-at "$TASK13_VERIFIED_AT" \
  > "$TASK13_TERMINAL_PUBLICATION"
```

Do not point the publisher at older runtime terminal paths and do not
hand-create any of the generation terminal objects.

After all three objects have exact versioned evidence:

```bash
export TASK13_STAGE=terminal
export TASK13_STAGE_PACKAGE="$TASK13_PRODUCTION_PACKAGE"
export TASK13_STAGE_REVIEWED="$TASK13_PRODUCTION_REVIEWED"
export TASK13_STAGE_AUTHORITY="$TASK13_WORK/authorities/terminal.json"
```

Run section 8. Require `status=STAGE_COMMITTED`, `stage=terminal`,
`sky_state=SUCCEEDED`, `billable_instance_count=0`, the two exact terminal
marker coordinates, reconciled GPU totals, remaining approval, and retained
costs.

Terminal completion does not authorize automatic model promotion,
publication, registry upload, or accepted-baseline replacement.

## 17. Honest remaining-gap register

| Gap | Current repository evidence | Execution effect | Required closure |
|---|---|---|---|
| Final transport-gate reseal after concurrent source changes | Gate JSON and publisher exist, but the final two-pass reseal must occur only after source stabilization | Transport publication/package input is not final | Rerun the independent two-pass gate observation, validate zero source drift, then publish once |
| Live artifact publication | No live VersionIds are recorded in this documentation task | Neither package can be claimed live-ready | Execute each authorized publisher once and retain exact readback evidence |
| Live terminal source evidence | The terminal publisher exists, but this documentation task has no live `CAMPAIGN_DRAINED.json`, terminal-state file, or settlement file | Terminal publication and the `terminal` stage must remain blocked | Produce the three sources from authenticated runtime, ledger, cost, and zero-billable-instance evidence, then run the one-shot publisher |
| Live AWS/deployed truth | This runbook performed no AWS calls or deployment | No stack, rehearsal, H100, launch, monitor, or terminal claim is currently proven | Execute only after explicit live authorization and retain the three journals plus immutable S3/CloudFormation/Step Functions evidence |

## 18. Stop conditions

Stop immediately and preserve every local and remote evidence record if any
of these occurs:

- account, region, profile, role, credential-expiry, approval, or executable
  identity drift;
- a required path is missing, relative, a symlink, or already exists where an
  exclusive output is required;
- a reviewed artifact lacks an exact VersionId, checksum, file hash, or body
  hash;
- package phase, predecessor, artifact count, fixed key, or semantic gate
  drift;
- any journal is not canonical append-only `0600` JSONL or journals alias;
- any stage prerequisite is not durably committed;
- a mutation is `POSSIBLY_SENT` and exact reconciliation is not yet positive;
- more than one active P5 instance, a foreign instance type, Spot, Capacity
  Block, or a raw launch path;
- cache/H100 identity drift, source/replacement identity equality, peak memory
  at or above 70 GiB, or unexpected production effects during qualification;
- one hour or less of approved GPU time remains before production;
- terminal markers, allocation costs, remaining approval, retained costs, Sky
  terminality, or zero-billable-P5 proof are incomplete;
- any proposal to retry by changing the run ID, activation, package, request,
  ClientToken, availability-zone order, cloud, region, budget, or validator.

The next action after any stop is evidence review. It is never an ad hoc
launch or a fabricated closure record.
