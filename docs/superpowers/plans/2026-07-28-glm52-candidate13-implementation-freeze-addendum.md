# GLM-5.2 Candidate-Thirteen Implementation Freeze Addendum

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this addendum
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the literal schema ambiguities discovered while converting the
candidate-thirteen design into the executable production-enforcement plane.

**Architecture:** This addendum is subordinate to
`docs/superpowers/plans/2026-07-27-glm52-full-campaign-run-handoff.md` and
changes no campaign scope, cost ceiling, instance shape, launch authority, or
live rollout order. It supplies exact names where the candidate uses prose or
contains a stale review remnant, choosing the most restrictive interpretation
that makes the handoff executable.

**Tech Stack:** Python 3.9-compatible standard-library enforcement package,
canonical JSON/SHA-256 records, DynamoDB conditional transactions, S3
conditional writes, CloudFormation, Step Functions Standard workflows, and
pinned SkyPilot `0.13.0`.

## Global Constraints

- Account is exactly `246813579024`; region is exactly `us-west-2`.
- Production uses at most one on-demand `p5.48xlarge`; Spot, Capacity Blocks,
  other regions, and other worker shapes are forbidden.
- The GPU envelope remains 24 cumulative hours and `$1,320.96`.
- The residual GPU reserve is exactly 900 seconds and `$13.76`.
- The 300-GiB root-volume tail is separate and at most `$0.01`; it is never
  folded into the GPU-reserve field.
- Every schema-defined absence is exact JSON `null`, never omission.
- Every closed record rejects missing fields, unknown fields, type coercion,
  booleans in integer fields, non-finite numbers, and canonical hash drift.
- No pure validator claims AWS terminality, transport success, or authority
  that only a live adapter can prove.

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

---

## Frozen ambiguity resolutions

### 1. Canonical self-hash rule

For every record containing `canonical_body_sha256`, its canonical identity is
the lowercase SHA-256 of the canonical UTF-8 JSON object with only
`canonical_body_sha256` omitted. No other field is omitted. This applies
uniformly to `ROLLOVER`, `OPERATOR_DISPOSITION`,
`SNAPSHOT_CLEANUP_TRANSITION`, `POST_TERMINAL_ALLOCATION`,
`WORKER_LAUNCH_LIABILITY_SETTLEMENT`, and `TERMINAL_V2`.

### 2. Current CONTROL closure grammar

The current normative grammar is:

```text
OPEN -> RECOVERY_SEALING -> RECOVERY_COMPLETE
RECOVERY_COMPLETE -> TEARDOWN_SEALING -> TEARDOWN_SEALED
```

The historical review-table text `OPEN -> TEARDOWN_SEALING` is stale and is
not executable.

### 3. Retained action confinement fields

The common retained action envelope from candidate thirteen gains these exact
fields so its body authenticates the ordinal/resource named by its sort key:

```text
generation
generation_text
allocation_ordinal
allocation_ordinal_text
worker_launch_identity_sha256
worker_launch_liability_identity_sha256
```

For non-worker domains, the allocation fields and both worker identity fields
are exact JSON `null`. For `WORKER_LAUNCH_LIABILITY_ACTION`, generation is the
bound production generation, both allocation fields are non-null and mutually
consistent, and both worker identities are non-null exact SHA-256 values.

### 4. Retained action-kind enums

The exact retained action-kind sets are:

```text
RECOVERY_ACTION:
  REQUEST_CANCEL
  JOB_CANCEL
  WORKER_DRAIN
  TERMINAL_V2_PUBLISH

FINALIZATION_ACTION:
  SUPPORT_DELETE
  SNAPSHOT_DISPOSITION
  H1G_DRAINED_PUBLISH

SNAPSHOT_CLEANUP_ACTION:
  SNAPSHOT_DELETE

WORKER_LAUNCH_LIABILITY_ACTION:
  SAME_TOKEN_COMPLETE
  TERMINATE_LATE_INSTANCE
  POST_TERMINAL_ALLOCATION_DISCOVER
  POST_TERMINAL_ALLOCATION_OPEN
  POST_TERMINAL_ALLOCATION_CLOSE
  LIABILITY_SETTLE
```

Creating a `POST_TERMINAL_ALLOCATION` in `DISCOVERED` consumes
`POST_TERMINAL_ALLOCATION_DISCOVER`. The later
`DISCOVERED -> ALLOCATION_OPEN` and
`INSTANCE_TERMINAL -> ALLOCATION_CLOSED` edges consume `_OPEN` and `_CLOSE`
respectively. No action kind is overloaded.

### 5. Ownership/nullability categories

Fields in each closed record are divided into immutable identity, owner,
progress evidence, and terminal evidence:

- `DORMANT`, `ARMED`, `START_OWNED`, `PREPARED_NOT_SENT`, and
  `UNOWNED_NOT_ACTIONABLE` use the literal initial-state nullability stated by
  candidate thirteen.
- Owner fields are all null or all non-null. Partial owner identity is
  invalid.
- A transition never clears a non-null evidence field.
- Evidence not yet reached in the state grammar is exact JSON `null`.
- Evidence required at or before the current state is non-null.
- Terminal states have no live owner except the candidate’s explicit
  termination-only liability states.
- Owner takeover changes only owner identity, owner attempt, owner hard
  expiry, revision, and `updated_at`; it preserves state and all evidence.
- Every successful edge increments `revision` by exactly one.

Record-specific tests must enumerate the required non-null evidence at every
edge. The generic validator may enforce the categories, but it may not accept
an omitted state-specific matrix.

### 6. Exact `TERMINAL_V2` schema

`TERMINAL_V2` contains exactly:

```text
schema_version
record_type
account_id
region
run_id
campaign_identity_sha256
activation_id
activation_ordinal
generation
generation_text
action_key
action_identity_sha256
handoff
binding
final_sky_state
final_ec2_states
request_cardinality
worker_cardinality
allocations
allocations_array_sha256
worker_launch_evidence
worker_launch_evidence_array_sha256
worker_launch_liabilities
worker_launch_liabilities_array_sha256
spend_ledger_head_identity
remaining_approved_gpu_seconds
remaining_approved_gpu_usd
final_heartbeat_identity
checkpoint_identity
cache_identity
training_identity
evaluation_identity
drain_identity
terminal_observation_window
request_evidence
request_evidence_array_sha256
post_terminal_quiescence_evidence
prior_terminal_v1_identity
outcome
operator_disposition_required
writer_function_version_arn
writer_dispatch_identity_sha256
writer_invocation_nonce_sha256
created_at
canonical_body_sha256
```

Additional rules:

- `schema_version` is `2`;
- `record_type` is `glm52_production_terminal_v2`;
- `generation_text` is the exact eight-digit rendering of `generation`;
- `remaining_approved_gpu_seconds` is an integer, not a boolean;
- `remaining_approved_gpu_usd` is a two-decimal canonical decimal string;
- `final_ec2_states` is instance-ID sorted;
- every array hash is recomputed from canonical array bytes;
- `request_cardinality`, `worker_cardinality`, outcome enums, handoff/binding
  nullability, allocation counts, request-evidence construction, and
  operator-disposition requirement follow the complete candidate-thirteen
  matrices;
- late workers never mutate this record.

### 7. Terminal marker identities

`final_heartbeat_identity`, `checkpoint_identity`, `cache_identity`,
`training_identity`, `evaluation_identity`, and `drain_identity` are each an
exact object containing:

```text
key
version_id
body_sha256
canonical_identity_sha256
```

or exact JSON `null`. DRAINED outcomes require non-null `drain_identity`;
`DRAINED_COMPLETED` requires non-null training and evaluation identities;
`DRAINED_TRAINING_DEFERRED` requires non-null cache/checkpoint evidence and
null training/evaluation identities; `DRAINED_RESUMABLE_DEADLINE` requires a
non-null checkpoint identity. No-job and no-launch outcomes use null marker
identities except the outcome-specific prior terminal/checkpoint evidence
required by candidate thirteen.

### 8. Residual price correction

`$13.76` is authoritative because:

```text
$55.04/hour * 900 seconds / 3600 seconds/hour = $13.76
```

Every `$13.77` occurrence in historical candidate review/rollout prose is a
stale aggregate/rounding remnant. Validators and mutants must reject
`gpu_reserve_usd="13.77"`. The separately approved root-volume tail remains
`root_volume_tail_usd_max="0.01"`.

### 9. DynamoDB transaction preimage and client-request token

`transaction_bytes_sha256` is the SHA-256 of a canonical transaction preimage,
not the impossible hash of final self-containing DynamoDB wire bytes. The
preimage is canonical UTF-8 JSON containing exactly:

```text
schema_version = 1
domain
operation_identity_sha256
ordered_logical_write_plan
```

The ordered logical write plan contains closed logical keys, conditions,
before records, and after records in the exact request order. It contains no
raw owner nonce and no derived client-request token. This removes the
candidate's circular dependency between the final `ROLLOVER` body,
`transaction_bytes_sha256`, and
`transaction_client_request_token_sha256`.

For this preimage only, record normalization is exact and uniform:
`canonical_body_sha256`, `transaction_bytes_sha256`, and every
`*_transaction_client_request_token_sha256` field are omitted. The rollover
transaction has one additional necessary derived-field normalization. In the
four new control Put records, `rollover_identity_sha256` is represented in
the preimage by the exact JSON string
`"__DERIVED_ROLLOVER_CANONICAL_BODY_SHA256__"`. After the preimage hash and
client token are derived, the final ROLLOVER record is materialized and
self-hashed; that exact final `canonical_body_sha256` is then installed into
all four control records. The four frozen control schemas do not contain a
`canonical_body_sha256` field, so there is no control self-hash to recompute;
their complete exact field sets are validated after installation. No other
field is omitted, substituted, or changed. This is the sole executable
resolution of the otherwise circular requirement that the transaction
preimage carry the control records while those records carry the final
ROLLOVER identity.

The liability-settlement transaction uses the same rule for its one derived
cross-record identity. In the LIABILITY Update after-record,
`settlement_identity_sha256` is represented in the preimage by the exact JSON
string `"__DERIVED_SETTLEMENT_CANONICAL_BODY_SHA256__"`. After the preimage
hash and client token are derived, the final SETTLEMENT Put record is
materialized and self-hashed; that exact `canonical_body_sha256` is installed
as the terminal LIABILITY after-record's
`settlement_identity_sha256`. The LIABILITY schema has no self-hash field, so
its complete exact field set is validated after installation. The live and
durable-adoption paths must enforce the same final binding.

For later rollover, the ACTIVATION_INDEX predecessor identities follow the
existing exact-object schema. `prior_activation_terminal_v2_identity` is the
exact three-field mapping `{"key": prior_terminal_v2_key, "version_id":
prior_terminal_v2_version_id, "body_sha256":
prior_terminal_v2_body_sha256}`. `prior_h1g_drained_identity` uses the
identical three-field object projection over the corresponding
`prior_h1g_drained_*` fields.
`prior_spend_ledger_head_identity` is copied exactly from ROLLOVER, and
`snapshot_cleanup_lineage_sha256` equals ROLLOVER's exact lineage hash. A
first-activation index has an absent prestate and null predecessor identities.

The exact `TransactWriteItems.ClientRequestToken` is:

```text
"h1g-" + first_32_lowercase_hex(
  SHA256(
    UTF8("glm52-ddb-crt-v1") || NUL ||
    UTF8(domain) || NUL ||
    UTF8(operation_identity_sha256) || NUL ||
    UTF8(transaction_bytes_sha256) || NUL ||
    raw_owner_nonce
  )
)
```

It is exactly 36 ASCII characters. Records store only
`SHA256(ClientRequestToken ASCII)` and the transaction-preimage hash. The raw
32-byte nonce is never serialized, logged, stored, returned, or included in
exception text.

The AWS ten-minute client-request-token idempotency window is transport
evidence only. A live invocation issues each transaction at most once and
never retries it. Every nominal success and every ambiguous error takes the
same mandatory coherent-readback path. A restart never inherits ownership
from the token, and a later reconciler adopts only exact durable records under
the separately closed duplicate-adoption rules. Because
`TransactWriteItems` does not return successful updated items, all candidate
language that says a transaction “returns” its new item means one no-retry
write followed by the required consistent or transactional readback.

### 10. Guarded production CLI vocabulary

The production operator surface is:

```text
aws/glm52-gpu/scripts/submit_sky_campaign.sh --production validate-only ...
aws/glm52-gpu/scripts/submit_sky_campaign.sh --production start ...
aws/glm52-gpu/scripts/submit_sky_campaign.sh --production reconcile ...
```

The shell maps those invocations to the Python parser mode `production` with
actions `validate-only`, `start`, and `reconcile`. `start` starts only the
exact published production Standard-workflow version for the already
allocated activation/epoch; it never invokes the public SkyPilot
`jobs.launch()` client and never submits directly from the operator process.
`reconcile` observes the same execution identity after an ambiguous start and
cannot allocate a new epoch or issue a second start.

Production input requires the exact immutable
`campaigns/glm52-sky-20260724/qualification/H100_RESUME_READY.json` key,
opaque VersionId, exact file/body hashes, and the already validated marker
identity. The route must exact-read that sole version and bind it into the
production submission intent before the workflow can consume the production
action. A literal marker name in a comment, a current-version read without an
opaque VersionId, or aliasing production to qualification/cache-seed does not
satisfy this contract.

### 11. S3 package, metadata, reconciliation, and marker coordinates

`glm52_enforcement.__init__` contains only version metadata. Privileged code
imports explicit submodules; the package initializer never imports approvals,
records, transitions, AWS adapters, MLX, NumPy, or `mlx_vq.quality`.

The production bucket name is authenticated from the closed campaign
descriptor and then treated as an immutable activation input. It is not a
new hard-coded schema constant. Every S3 list, get, head, and put call passes
`ExpectedBucketOwner="246813579024"`.

Every immutable JSON create uses this exact closed, lower-case S3 user-metadata
map:

```text
glm52-account-id
glm52-activation-id
glm52-body-sha256
glm52-candidate-identity-sha256
glm52-file-sha256
glm52-generation-text
glm52-record-kind
glm52-region
glm52-run-id
```

The values are exact strings; `generation_text` is eight decimal digits.
Metadata is compared as a complete map after S3's lower-case normalization.
Missing, extra, non-string, or substituted metadata is a mismatch.

The already canonical H.1c coordinates remain authoritative:

```text
campaign descriptor:
  the exact campaign_descriptor_key carried and authenticated by the
  descriptor record

GPU spend approval:
  campaigns/{run_id}/authorities/
  GPU_SPEND_APPROVAL-{approval_file_sha256}.json
```

The five activation source families use the existing accepted key helpers and
the order frozen by candidate thirteen. A direct exact `200` and an ambiguous
create that reconciles to exactly one sole current non-delete version with
matching raw bytes, checksum, content type, metadata, and key both yield the
service-assigned object identity needed by the next source. Reconciliation
does not reconstruct direct-response authority: its provenance remains
`all-version-reconciliation`, it authorizes no repeated put, and it may feed
the next candidate only after the exact predecessor identity is durably
recorded by the same activation owner. Zero candidates, history, siblings,
delete markers, multiple versions, or any mismatch abort the source sequence.

The remaining marker coordinates are:

```text
SKY_POST_HANDOFF:
  campaigns/{run_id}/submissions/production/generations/
  {generation_text}/handoff/SKY_POST_HANDOFF.json

BOOTSTRAP_READY:
  campaigns/{run_id}/submissions/production/generations/
  {generation_text}/allocations/{allocation_ordinal_text}/
  BOOTSTRAP_READY.json

WORKER_GRACEFUL_STOP:
  campaigns/{run_id}/submissions/production/generations/
  {generation_text}/allocations/{allocation_ordinal_text}/
  WORKER_GRACEFUL_STOP.json

CAMPAIGN_DRAINED:
  campaigns/{run_id}/submissions/production/generations/
  {generation_text}/terminal/CAMPAIGN_DRAINED.json

SUPPORT_PLANE_FINALIZED:
  campaigns/{run_id}/submissions/production/activations/
  {activation_id}/finalization/SUPPORT_PLANE_FINALIZED.json

H1G_DRAINED:
  campaigns/{run_id}/submissions/production/activations/
  {activation_id}/finalization/H1G_DRAINED.json
```

Each physical path is one line after concatenation. `generation_text` and
`allocation_ordinal_text` are eight digits; allocation ordinals are positive.
`activation_id` is the exact closed ledger identity and must pass the same
ASCII-safe path-segment rules as every H.1f key. Generic marker names at the
campaign root are compatibility inputs only and are not H.1g production
authority.

The Task 4 conditional-create effect adapter initially admits only
`authority_domain="ACTIVATION"` with `operation_kind="S3_CREATE"` and exact
ACTIVATION_INDEX/CONTROL/ACTION coherent readback. Recovery, finalization,
snapshot-cleanup, and liability publications are rejected at that boundary
until their later tasks add and independently review the corresponding
domain-specific control/action schemas and transitions; a caller may never
relabel activation records as another authority domain.

### 12. Three-stack bootstrap and lifecycle names

The exact stack names are:

```text
retained infrastructure: keep-glm52-gpu
production fence:        keep-glm52-h1g-fence
ephemeral support:       keep-glm52-h1g-support
```

The final fence-policy logical ID is
`H1gProductionFenceBucketPolicy`. The final fence stack is parameterless and
contains only that one `AWS::S3::BucketPolicy`.

CloudFormation requires every template to declare at least one resource, so
the candidate's literal zero-resource bootstrap container is not executable.
Each new container is instead created from a versioned parameterless template
containing exactly one `AWS::CloudFormation::WaitConditionHandle` named
`ContainerAnchor`, no outputs, and no other resource. The anchor receives no
signals and has no `AWS::CloudFormation::WaitCondition`, so stack creation
does not wait. It is a bootstrap identity only:

- the fence import change set retains `ContainerAnchor` while importing the
  policy, because import change sets cannot mix unrelated resource mutation;
- the immediately following reviewed update removes only `ContainerAnchor`;
- the disabled-support update replaces its anchor with the exact support
  inventory;
- no runtime, workflow, role, policy, secret, network, launch, or S3 authority
  is attached to either anchor.

Member-account lifecycle controls enumerate the two exact anchor-removal
transitions in addition to the finite fence/support transitions. Both new
stacks have termination protection enabled at create time.

The support lifecycle's exact deletion sequence is:

```text
FINALIZATION_CONTROL reaches SUPPORT_FINALIZED
-> exact retained lifecycle role calls
   UpdateTerminationProtection(EnableTerminationProtection=false)
   on only the service-assigned keep-glm52-h1g-support stack ID
-> coherent DescribeStacks readback proves false
-> the same retained lifecycle execution calls DeleteStack once,
   passing only the exact support deletion service role
-> deletion is reconciled to exact stack absence
```

No other principal may disable termination protection. The retained and fence
stacks remain protected. An ambiguous termination-protection response is
reconcile-only and never causes a second mutation until exact readback proves
the current state.

Every new stack uses this exact tag map:

```text
Project=KEEP
Campaign=GLM-5.2
RunId=glm52-sky-20260724
Environment=production
ManagedBy=CloudFormation
Authority=H1g
```

Service-assigned stack IDs, current live bucket name/policy, import identifier
key, retained deployment RoleId, member-account identity, and credential expiry
are live readback facts and must never be filled from a default or guessed.
`AWS::S3::BucketPolicy` supports CloudFormation import but not drift
detection; the migration therefore requires direct canonical
`GetBucketPolicy` equality before, during, and after ownership transfer.

## Dependency-ordered implementation tasks

- [x] **Task 1:** Authenticated support-plane and residual-liability approval
  builders/validators.
- [x] **Task 2:** Exact import-light records, key grammar, state matrices,
  canonical identities, transitions, and owner takeover.
- [x] **Task 3:** DynamoDB nonce/epoch/transaction/readback adapter.
- [x] **Task 4:** S3 conditional-create and fresh H.1f adapter.
- [x] **Task 5:** Sequential source publishers and batch successor.
- [x] **Task 6:** Three-stack finite CloudFormation/fence/support ownership.
- [x] **Task 7:** Support network, secrets, KMS, and cost ceilings.
- [x] **Task 8:** Live H.1d and spend authority.
- [x] **Task 9:** Sky admission, intent-only provisioner, same-token completer,
  and liability custody.
- [x] **Task 10:** Mount-free task, worker units, and guarded production CLI.
- [ ] **Task 11:** Decision workflow and measured closure.
- [ ] **Task 12:** Correlation, terminal-v2, retained recovery/finalization,
  snapshot cleanup, and orphan audit.
- [ ] **Task 13:** Complete `T01`–`T25`, 22-mutant, disabled-stack rehearsal,
  archive, and independent review gates.

Each task begins with a named failing test, records literal RED and GREEN
evidence, and is independently reviewed before the next task.
