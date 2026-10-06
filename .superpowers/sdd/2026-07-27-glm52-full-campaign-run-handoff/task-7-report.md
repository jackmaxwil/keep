# Task 7 Report — ephemeral support plane, secrets, KMS, and ceilings

Date: 2026-07-28

## Status

`DONE`

Task 7 is implemented as an import-light, Python 3.9-compatible pure contract
and deterministic builder. It makes no AWS, network, deployment, billing, or
live-secret claim. No Task 6 file was modified.

## Owned files

- `src/glm52_enforcement/support_plane.py`
- `src/glm52_enforcement/support_custom_resources.py`
- `aws/glm52-gpu/scripts/build_h1g_support_plane.py`
- `aws/glm52-gpu/cfn/h1g/support-input-contract-v1.json`
- `aws/glm52-gpu/cfn/h1g/support-template-contract-v1.json`
- `aws/glm52-gpu/cfn/h1g/support-spend-envelope-v1.json`
- `aws/glm52-gpu/cfn/h1g/support-contract-manifest-v1.json`
- `tests/test_glm52_enforcement_support_plane.py`
- `tests/test_glm52_enforcement_support_custom_resources.py`
- `.superpowers/sdd/2026-07-27-glm52-full-campaign-run-handoff/task-7-report.md`

There was no integration edit to
`src/glm52_enforcement/cloudformation_stacks.py`. The generated support
inventory remains consumable by the Task 6 reviewed-inventory boundary, while
Task 7 owns its stricter resource, topology, secret, KMS, lifecycle, and spend
validation.

## Implemented contracts

### Inputs, identity, generation, and ownership

- A closed `SupportInputs` type authenticates every environment-specific
  identity. It rejects missing/unknown fields, guessed aliases, type
  coercion, digest drift, unsafe IDs, malformed stack ARNs, unbound AMI/VPC/
  subnet/KMS/ledger/bucket/code identities, duplicate relay ports, and
  noncanonical timestamps.
- Canonical input identity is the SHA-256 of the exact canonical JSON
  projection.
- The builder accepts only canonical ASCII JSON ending in one LF, performs no
  AWS or network access, refuses an existing output directory, and writes
  byte-identical support template, retained augmentation contract,
  support-spend descriptor, and manifest artifacts.
- The manifest binds the input, template, retained augmentation, spend,
  resource ownership, retained-only import direction, and live KMS-grant
  evidence state. Every support resource has the sole owner
  `keep-glm52-h1g-support`.
- Checked-in H.1g assets are intentionally non-live contracts. They contain no
  invented AMI, VPC, subnet, or retained KMS ARN. An authenticated bound build
  is required to create a concrete support template.

### Network, host, storage, and runtime inventory

- Exact fixed support CIDRs are `10.20.101.0/24`,
  `10.20.102.0/24`, and `10.20.103.0/24`. Validation proves retained-VPC
  containment, pairwise nonoverlap, no overlap with authenticated retained
  subnet/secondary CIDRs, primary/alternate AZ mapping, and fixed host-address
  membership.
- The template owns three route tables and associations, one host-only default
  NAT route, one NAT/EIP, support S3/DynamoDB gateway endpoints attached to
  all three support route tables, and one private-DNS Secrets Manager
  interface endpoint in exactly the two isolated Lambda subnets. No KMS or
  second interface endpoint is legal.
- Four client security groups, four exact client egress rules, and four exact
  host ingress rules bind mutually distinct client identities to mutually
  distinct fixed ports. The combined host has no public IP and no port 22.
- Exactly one private `c6a.xlarge` host is bound to an exact authenticated AMI,
  fixed address, IMDSv2-required metadata, exact user-data and boot digests,
  encrypted retained-key root shape, and one encrypted 50-GiB gp3 data
  volume. The data volume has both snapshot policies. No snapshot restore,
  direct snapshot creation, alternate instance launch, or snapshot-recovery
  grant authority is emitted.
- Six support runtime function families and the custom-resource handler have
  pinned versions, reserved concurrency one, bounded time/memory, 14-day log
  groups, alarms, and an asynchronous deadline DLQ. Decision, deadline, and
  custom-resource functions are non-VPC. Attestation, admission, binding, and
  cancellation are attached only to both isolated subnets and their one
  client security group.
- One versioned Standard state machine and an invoke-only exact-version role
  are rendered. Its Task 7 definition fails closed until the later Task 11
  workflow implementation.

### Eight secrets, TLS, and custom-resource callback protocol

- Exactly eight physical Secrets Manager resources and eight resource policies
  are generated. Each secret is retained-key encrypted and carries one unique
  activation-and-purpose-specific immutable VersionStage.
- Raw token, bootstrap, four client TLS, combined-host TLS, and CA issuance
  families are distinct. The four client roles may read raw token plus only
  their own TLS family; the host reads only bootstrap and host TLS; the
  handler owns only TLS/CA material writes; the decision role reads no secret.
  Readers are stage-bound and explicitly denied mutation and all other
  families. Rotation and a ninth secret are rejected.
- `Custom::H1gSkyBootstrap` and `Custom::H1gTlsBundle` depend on the exact
  handler version, permission, handler/KMS-access role, and all material they
  must reconcile. The host depends on both completed custom resources.
  Reverse deletion therefore keeps these resources alive through Delete
  callback observation.
- Stable `PhysicalResourceId` uses only stack ID, logical ID, resource kind,
  and activation. Request type/ID and canonical event bytes affect only the
  operation token.
- The model covers Create, identity-preserving Update, partial writes,
  identical delivery, same-ID/different-event rejection, failed-Create
  material rollback, Delete material tombstones, immutable callback caching,
  callback attempt, transport result, and separate CloudFormation
  observation.
- The ResponseURL validator requires HTTPS, the exact commercial-partition
  CloudFormation response-bucket hostname family, the expected regional S3
  SigV4 scope, and exact StackId/RequestId/LogicalResourceId object identity.
  Alternate host, account, region, logical ID, request ID, scheme, redirects,
  and caller-selected destination fail closed.
- Canonical SUCCESS and worst-case FAILED callback bodies are at most 4,096
  bytes. The callback Data schema admits only compact issuance/version/hash
  identities and rejects private-material fields or bytes.

### Retained KMS grants, lifecycle, and spend

- Direct H.1g grants require exact request/response hashes, grant ID/name,
  grantee, retiring principal, bounded operations, and encryption context.
  Service-created grants require baseline/diff, ListGrants, CloudTrail,
  originating resource/service, context, and settling evidence.
- Unknown, duplicate, missing, overbroad, unattributed, or snapshot-recovery
  grants fail. Completion requires exact byte-for-byte equality with the
  retained grant baseline.
- Lifecycle evaluation has literal edges at 68 hours, 71 hours, and 72 hours.
  A 72-hour survivor writes `SUPPORT_DELETE_DEADLINE_MISSED`, pages, keeps
  host stopped and egress disabled, performs one bounded read-only stack
  reconciliation, and emits no direct child deletion.
- Egress evaluation sums all four NAT counters, warns at 5 GiB, drains and
  disables at 6 GiB, records authority excess above 10 GiB, and fails closed
  on missing, delayed, or `INSUFFICIENT_DATA` metrics after bootstrap grace.
- Every frozen invocation, concurrency, execution, history-event,
  finalization, runtime-observation, log, watcher, transition/read, custom
  resource, cancellation, snapshot, resource-cardinality, byte, and retention
  ceiling has one closed validator field.
- The dated regional price card contains all exact support and separately
  retained price terms. Extended prices are recomputed. A support estimate
  above `$25.00` is rejected. Retained S3/ledger/KMS, liability watcher,
  snapshot cleanup, and snapshot retention remain separate; the GPU residual
  reserve and worker root-volume tail are explicitly excluded.

## Literal RED to GREEN evidence

All production behavior began behind a named failing test or a named failing
cluster. The literal observed transitions were:

1. Exact typed support inputs/canonical identity:
   `ModuleNotFoundError: No module named 'glm52_enforcement.support_plane'`;
   `1 failed in 0.03s` -> `1 passed in 0.02s`.
2. Sole-stack ownership/resource cardinality:
   `ImportError: cannot import name 'SECRET_LOGICAL_IDS'`;
   part of `5 failed, 1 deselected in 0.05s` ->
   `6 passed in 0.03s`.
3. CIDR containment/nonoverlap/AZ mapping:
   `ImportError: cannot import name 'validate_support_network_inputs'`;
   same five-test RED -> six-test GREEN.
4. Host-only NAT route and isolated-route denial:
   `ImportError: cannot import name 'validate_support_template'`;
   same five-test RED -> six-test GREEN.
5. Endpoint count/association/private DNS/two-subnet/no-KMS contract:
   absent validator in the same endpoint/NAT RED -> six-test GREEN.
6. Four mutually unusable relay/client paths:
   `ImportError: cannot import name 'CLIENT_PATHS'`;
   same five-test RED -> six-test GREEN.
7. `$25.00` gate and separately itemized retained costs:
   `ImportError: cannot import name 'build_support_spend_descriptor'`;
   same five-test RED; first GREEN exposed literal
   `KeyError: 'absence_expected_hours'`; corrected to
   `6 passed in 0.03s`.
8. Exact eight-secret/stage/reader/no-rotation contract:
   `ImportError: cannot import name 'secret_inventory'`;
   part of checkpoint-2 `7 failed, 7 passed` -> combined
   `14 passed in 0.05s`.
9. Retained-key grant provenance/baseline restoration:
   `ImportError: cannot import name 'validate_kms_grant_inventory'`;
   same checkpoint-2 RED -> combined GREEN.
10. Stable custom-resource identity/request-bound operation:
    `ModuleNotFoundError: No module named
    'glm52_enforcement.support_custom_resources'`;
    part of `5 failed in 0.04s` -> custom-resource GREEN.
11. Foreign/malformed ResponseURL denial:
    same missing-module RED; the first semantic run caught an encoded-region
    mutant that had not actually changed bytes. The mutant was corrected to
    `%2F`, after which all foreign URL cases were killed.
12. Nonsecret 4,096-byte SUCCESS/FAILED callback budget:
    same missing-module RED -> byte-budget GREEN.
13. Replay/partial-write/transport/observation separation:
    same missing-module RED -> state separation GREEN.
14. Failed-Create rollback and Delete reconciliation:
    `ImportError: cannot import name 'complete_custom_resource_delete'`;
    `1 failed in 0.07s` -> `5 passed in 0.09s`.
15. Reverse deletion graph:
    same checkpoint-2 missing-module RED -> dependency mutant GREEN.
16. Runtime version/VPC/log/alarm/DLQ inventory:
    `ImportError: cannot import name 'RUNTIME_FUNCTIONS'`;
    part of `5 failed, 1 passed, 9 deselected in 0.72s` ->
    `6 passed, 9 deselected in 0.79s`.
17. 68/71/72-hour transitions/no child deletion:
    `ImportError: cannot import name 'SupportLifecycleState'`;
    same checkpoint-3 RED -> GREEN.
18. 5/6/10-GiB and missing-metric behavior:
    `ImportError: cannot import name 'evaluate_support_egress'`;
    same checkpoint-3 RED -> GREEN.
19. Every usage/log/event/runtime/cancel/snapshot ceiling:
    `ImportError: cannot import name 'SUPPORT_USAGE_CEILINGS'`;
    same checkpoint-3 RED -> GREEN.
20. Deterministic canonical no-overwrite builder:
    Python could not open the missing
    `aws/glm52-gpu/scripts/build_h1g_support_plane.py`;
    same checkpoint-3 RED -> GREEN.
21. Checked-in non-live generated contracts:
    `ImportError: cannot import name 'build_support_contract_artifacts'`;
    `1 failed in 0.11s` -> `1 passed in 0.04s`.

The combined-host/root/data-volume assertions first went GREEN under the
earlier resource/cardinality implementation and were then preserved by the
named host/storage mutant test. The Python 3.9/forbidden-import assertion also
passed on its first focused run; its independent import guard rejects any
attempted blocked import at runtime. These two are additional regression
proofs rather than fabricated RED transcripts.

## Verification

Focused Task 7:

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_support_plane.py \
    tests/test_glm52_enforcement_support_custom_resources.py
21 passed in 0.47s
```

Tasks 1–7 enforcement aggregate:

```text
$ .venv/bin/python -m pytest -q tests/test_glm52_enforcement_*.py
519 passed, 2 warnings in 11.76s
```

The two warnings are the pre-existing accepted SWIG deprecation warnings from
source-authority fixture parity.

Frozen current CloudFormation regression:

```text
$ .venv/bin/python -m pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 3.66s
```

Python 3.9 compilation/import:

```text
$ PYTHONPYCACHEPREFIX=/tmp/glm52-task7-pycache-019faa1b \
    /usr/bin/python3 -m py_compile \
    src/glm52_enforcement/support_plane.py \
    src/glm52_enforcement/support_custom_resources.py \
    aws/glm52-gpu/scripts/build_h1g_support_plane.py
PASS

$ PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 <blocked-import proof>
3.9.6 task7-python39-import-light-ok
```

Generated JSON, diff, and whitespace:

```text
all aws/glm52-gpu/cfn/h1g/*.json canonical parse: 7 artifacts
git diff --check over Task 7-owned tracked paths: PASS
owned-file trailing-whitespace audit: PASS
```

The focused builder test executed two independent bound builds, proved four
identical canonical artifacts and every manifest file hash/byte count, then
proved a second write to the existing destination failed without changing any
artifact.

## Final SHA-256

```text
0a1b3a013ae1a0e02b754c911ab8cd832f4793e120ec34edc337e664cf51074a  src/glm52_enforcement/support_plane.py
a244e3d729f468cc549d62872edf3673a41d5aa23760da87248c9c1197d4874a  src/glm52_enforcement/support_custom_resources.py
90d909b7f3b575d0c41732b0e43f7a756694291bff12df037ec28937836def6f  aws/glm52-gpu/scripts/build_h1g_support_plane.py
b84d14a2ea43e9d2da13e0c06b4c53289e2c8c55ccf9d3ec52544edeb4057a40  tests/test_glm52_enforcement_support_plane.py
d368ba79baf23b75f682540d64257717b720d9747bdab62b132d30d7ddda8e52  tests/test_glm52_enforcement_support_custom_resources.py
00ec7d4fff0805a54decb896e2c958b7cbc9840305341fc7fda7b76ca1f84866  aws/glm52-gpu/cfn/h1g/support-input-contract-v1.json
7944bb11fba8ffb0ab8a90c7b05a150d7842e1e4412c2b6a90b7493fc7b3e96d  aws/glm52-gpu/cfn/h1g/support-template-contract-v1.json
1fc1183c65349145cf8f05dca7f2f940cea396c72727d50c582e4b460bc9235e  aws/glm52-gpu/cfn/h1g/support-spend-envelope-v1.json
3cc0d0a08e4a2d9f96cd8a16c4b3d0ab5d1bcbc5fc69a6abbf654010119881f2  aws/glm52-gpu/cfn/h1g/support-contract-manifest-v1.json
```

## Self-review findings and limitations

- No Task 6 or unrelated file was edited. No existing test was weakened,
  skipped, marked xfail, or satisfied by marker-only source text.
- A self-review caught two test-quality errors: an explicit IAM Deny was being
  mistaken for an Allow, and a foreign-region URL mutant had not changed its
  percent-encoded credential scope. Both tests now assert the real behavior.
- A later self-review added explicit failed-Create private-material rollback
  and exact Delete tombstone reconciliation before completion.
- The checked-in artifacts are non-live contracts by design. A deployable
  bound support template requires authenticated exact environment inputs and a
  dated regional price card. No default AMI, VPC, subnet/AZ, fixed host
  address, retained KMS identity, or live price was invented.
- The template points to authenticated immutable code/layer coordinates; this
  task does not build cryptography binaries, perform certificate issuance, or
  claim a live Lambda invocation. The pure state model proves the callback
  contract only.
- The state machine is deliberately fail-closed until Task 11 supplies its
  reviewed decision graph. Task 7 does not claim launch custody, worker
  production CLI, spend-ledger reinspection, terminal-v2, or later teardown
  workflow behavior.
- CloudFormation resource acceptance, ResponseURL hostname/context behavior,
  IAM condition-key enforcement, service-created KMS grants, actual
  `ListSecretVersionIds`, real prices, and deletion ordering still require the
  later disabled-stack/live evidence gates. Pure validators do not promote
  those to deployed truth.
- No AWS call, network call, deployment, live secret, random secret value,
  billable resource, Git mutation, commit, or push occurred.

## Independent review fix round 1 of 5

Status: `DONE`

All eight review findings were reproduced with named tests and closed without
changing Task 6 or any unrelated file.

### Finding closures

1. Each of the four isolated readers now has one explicit TCP/443
   security-group egress edge to the Secrets Manager endpoint and one matching
   endpoint ingress edge. Endpoint and identity policies contain only exact
   caller, secret, and immutable-stage tuples, plus a closed cross-run deny.
2. The two custom resources now declare exact two-secret and six-secret target
   families. Material writes record exact family, VersionId, VersionStage,
   material digest, and request/response identities; all eight writes and
   `ListSecretVersionIds` observations are required before Create SUCCESS.
   The event parser requires the exact emitted five-property schema and
   resource-kind-specific target set. Each target must have only SecretId and
   VersionStage, with either its matching rendered logical Ref or its exact
   account/region/run/activation/purpose Secrets Manager ARN.
3. Create, identity-preserving Update, and Delete have distinct fail-closed
   state transitions. Update performs no writes, Delete requires exact
   per-family tombstones and reconciliation, callback Data is derived only from
   recorded versions, and callback attempt/HTTP 200/CloudFormation observation
   remain separate facts. Callback issuance IDs and reasons use closed
   grammars.
4. Retained KMS validation now requires exact equality between baseline plus
   authenticated direct/service-created projections and the active grant
   inventory. Exact grantee, retiring principal, operations, encryption
   context, revocability, request/response hashes, ListGrants, and CloudTrail
   provenance are mandatory; unknown, missing, duplicate, overbroad,
   unattributed, unrevocable, and snapshot-recovery grants fail.
5. Price-card identity now authenticates the canonical unsigned card against
   the exact input identity. Every term has a closed unit, positive fixed
   quantity and rate, mechanically recomputed estimate, and a recomputed
   support total at or below `$25.00`.
6. Host input and callback scanners now reject PEM private keys, credential
   assignments, AWS access-key patterns, JWT-shaped values, private-material
   fields, and private-material bytes. First boot receives only exact
   SecretId, VersionId, and VersionStage coordinates.
7. The support template and retained augmentation now contain executable
   68/71/72-hour schedules, exact lifecycle/deletion roles, host stop and
   security-group egress disable authority, stack deletion through one passed
   CloudFormation deletion role, DynamoDB reconciliation, operator paging,
   fail-closed 5/6/10-GiB alarms, and a zero-retry versioned invocation path.
   Lifecycle roles have no direct child-deletion or retained-stack mutation
   authority.
8. Security, lifecycle, retained-augmentation, runtime, and secret-reader
   validators now inspect the supplied graph structurally. A dedicated test
   replaces `_build_support_template` with a failing sentinel and proves the
   runtime and reader validators do not regenerate or trust the builder.

### Fix-round RED to GREEN evidence

```text
endpoint/reader-path cluster:
3 failed, 21 passed -> 24 passed

custom-resource write/observation/lifecycle cluster:
7 failed, 22 passed -> 29 passed

KMS, price-card, private-material, and template binding cluster:
32 focused tests passed after the named RED assertions were implemented

lifecycle authority and independent-graph cluster:
2 failed, 32 passed
-> intermediate validator fan-out exposed 15 failed, 19 passed
-> type-aware IAM Resource inspection left 2 failed, 32 passed
-> exact relay-path and rotation-resource checks produced 34 passed

independent-validator self-review:
1 failed in 0.09s
-> 3 selected tests passed in 0.04s
-> 35 focused tests passed in 0.26s

emitted SecretTargets event self-review:
1 failed in 0.05s
-> 1 passed in 0.03s
-> 36 focused tests passed in 0.30s
```

### Fix-round verification

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_support_plane.py \
    tests/test_glm52_enforcement_support_custom_resources.py
36 passed in 0.30s

$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py
593 passed, 2 warnings in 4.39s

$ .venv/bin/python -m pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.53s

$ PYTHONPYCACHEPREFIX=/tmp/glm52-task7-pycache-fix1 \
    /usr/bin/python3 -m py_compile \
    src/glm52_enforcement/support_plane.py \
    src/glm52_enforcement/support_custom_resources.py \
    aws/glm52-gpu/scripts/build_h1g_support_plane.py
PASS

$ PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
    /usr/bin/python3 <blocked-import proof>
3.9.6 task7-python39-import-light-ok

regenerated four checked-in Task 7 contracts: byte-identical
all aws/glm52-gpu/cfn/h1g/*.json canonical parse: 7 artifacts
git diff --check over Task 7-owned tracked paths: PASS
owned-file trailing-whitespace audit: PASS
```

The two aggregate warnings are the same accepted SWIG deprecation warnings
from source-authority fixture parity.

### Fix-round final SHA-256

```text
d48c575bc364ad9560397aab195f4436470e65478b6a8c19bb5592af0d70a0ed  src/glm52_enforcement/support_plane.py
13c49db154784173fdf083ad95e87fe2df635d22b4b1870822b27bc52ca61165  src/glm52_enforcement/support_custom_resources.py
2a81d2c217113d7fb4a87560bdd8e517f7b2c2e14a4ba502e7d08370116f6bb0  aws/glm52-gpu/scripts/build_h1g_support_plane.py
f54947addd14d385cf0a41c5f3d499cf873de7397558d0a3e903cf4cf690719f  tests/test_glm52_enforcement_support_plane.py
d628d622dbe15f4873098664993c9feebae90d12fbb94e19647b6bc50134344b  tests/test_glm52_enforcement_support_custom_resources.py
00ec7d4fff0805a54decb896e2c958b7cbc9840305341fc7fda7b76ca1f84866  aws/glm52-gpu/cfn/h1g/support-input-contract-v1.json
7944bb11fba8ffb0ab8a90c7b05a150d7842e1e4412c2b6a90b7493fc7b3e96d  aws/glm52-gpu/cfn/h1g/support-template-contract-v1.json
1fc1183c65349145cf8f05dca7f2f940cea396c72727d50c582e4b460bc9235e  aws/glm52-gpu/cfn/h1g/support-spend-envelope-v1.json
3cc0d0a08e4a2d9f96cd8a16c4b3d0ab5d1bcbc5fc69a6abbf654010119881f2  aws/glm52-gpu/cfn/h1g/support-contract-manifest-v1.json
```

### Fix-round limitations

- These remain deterministic pure-contract and fixture proofs, not live AWS,
  IAM, CloudFormation, Secrets Manager, KMS, price, billing, or deletion
  evidence.
- The support state machine remains intentionally disabled until Task 11.
- No AWS call, network call, deployment, secret generation, Git mutation,
  commit, or push occurred in this fix round.

## Reviewer fix round 2

### Closed findings

- Secrets Manager endpoint authority now emits separate exact statements for
  staged `GetSecretValue` and unconditioned exact-secret
  `ListSecretVersionIds`; the stage condition can no longer make list
  reconciliation unsatisfiable. Host user data rejects PEM certificates,
  certificate requests, public keys, SSH public keys, and public-trust
  assignments as well as private material.
- KMS validation is activation-bound. It requires one exact direct grant with
  the TLS handler grantee, deletion-role retiring principal, exact operations,
  exact run/activation context, and exact request/response identities. Service
  grants require exact originating service, logical resource, resource
  identity, name, operations, context, evidence identities, and retiring
  principal before exact retained-baseline restoration is accepted.
- Support-stack deletion has one owner. The support deadline role can only
  read the support stack and cannot update protection, delete the stack, or
  pass the deletion role. The retained lifecycle role alone can submit
  `UpdateTerminationProtection(false)`, require a `DescribeStacks` readback,
  submit one `DeleteStack` with the exact CloudFormation service role, and
  reconcile ambiguous outcomes without resubmitting either mutation.
- The authenticated support deletion inventory binds unique exact resource
  ARNs, IDs, names, action-to-resource allowlists, and the two exact custom
  resource Delete callbacks. The CloudFormation deletion service role has 40
  independently validated actions with no bare wildcard resource and exact
  snapshot/KMS conditions.
- NAT authority is activation-cumulative rather than one-minute-local. A
  replay-safe pure accumulator accepts consecutive 60-second windows, sums all
  four NAT counters once, rejects gaps and altered replays, and caps accepted
  windows at the 72-hour campaign bound. A retained one-minute schedule
  persists that ledger and publishes
  `GLM52/H1g/ActivationCumulativeNatProcessedBytes`; the 5/6/10-GiB alarms read
  that cumulative metric.
- Bootstrap grace is derived from the authenticated activation start and fixed
  at 900 seconds. Missing metrics never authorize work during grace and drain
  after the exact deadline. Alarm-level missing data is non-breaching because
  the retained evaluator owns the activation-bound missing-metric decision.

### Fix-round-2 RED to GREEN

```text
endpoint/list-condition and public-trust RED:
2 failed
-> 4 selected tests passed

activation-bound KMS RED:
named foreign/unbound-grant test failed on the missing exact input binding
-> 4 selected KMS/manifest tests passed

retained deletion RED:
2 failed
-> 2 selected tests passed

cumulative NAT/bootstrap grace RED:
3 failed, 31 deselected
-> 3 passed, 31 deselected

full focused Task 7:
44 passed in 0.34s
```

### Fix-round-2 verification

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_support_plane.py \
    tests/test_glm52_enforcement_support_custom_resources.py
44 passed in 0.34s

$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py
601 passed, 2 warnings in 4.67s

$ .venv/bin/python -m pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.48s

$ PYTHONPYCACHEPREFIX=/tmp/glm52-task7-pycache-fix2 \
    /usr/bin/python3 -m py_compile \
    src/glm52_enforcement/support_plane.py \
    src/glm52_enforcement/support_custom_resources.py \
    aws/glm52-gpu/scripts/build_h1g_support_plane.py
PASS under Python 3.9.6

$ PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
    /usr/bin/python3 <blocked-import proof>
3.9.6 task7-python39-import-light-ok

regenerated four checked-in Task 7 contracts: byte-identical
all aws/glm52-gpu/cfn/h1g/*.json canonical parse: 7 artifacts
git diff --check over Task 7-owned tracked paths: PASS
owned-file trailing-whitespace audit: PASS
```

The two aggregate warnings remain the accepted SWIG deprecation warnings from
source-authority fixture parity.

### Fix-round-2 final SHA-256

```text
32c5b0eb9e5f50f83e48aa4df933992d85fa13782292b5f7bc3fa7f211f07e2e  src/glm52_enforcement/support_plane.py
13c49db154784173fdf083ad95e87fe2df635d22b4b1870822b27bc52ca61165  src/glm52_enforcement/support_custom_resources.py
2a81d2c217113d7fb4a87560bdd8e517f7b2c2e14a4ba502e7d08370116f6bb0  aws/glm52-gpu/scripts/build_h1g_support_plane.py
06b8602ccad4d148f205efed79e0424d645bef080a26626283f61404c15db4fb  tests/test_glm52_enforcement_support_plane.py
d628d622dbe15f4873098664993c9feebae90d12fbb94e19647b6bc50134344b  tests/test_glm52_enforcement_support_custom_resources.py
d84dab97e683ab043ea20cb90dcc5d462ec37cafae054c64ca831b2e3a668cc3  aws/glm52-gpu/cfn/h1g/support-input-contract-v1.json
7944bb11fba8ffb0ab8a90c7b05a150d7842e1e4412c2b6a90b7493fc7b3e96d  aws/glm52-gpu/cfn/h1g/support-template-contract-v1.json
1fc1183c65349145cf8f05dca7f2f940cea396c72727d50c582e4b460bc9235e  aws/glm52-gpu/cfn/h1g/support-spend-envelope-v1.json
98706ca3fb6d932f95309956567aa54abca96e8ca212036b219e9097617852a8  aws/glm52-gpu/cfn/h1g/support-contract-manifest-v1.json
```

### Fix-round-2 limitations

- These remain deterministic pure-contract and fixture proofs, not live AWS,
  IAM, CloudFormation, Secrets Manager, KMS, metrics, billing, deletion, or
  public-trust evidence.
- The support state machine remains intentionally disabled until Task 11.
- No AWS call, network call, deployment, secret generation, Git mutation,
  commit, or push occurred in fix round 2.

## Reviewer fix round 3

Status: `DONE`

### Closed findings

1. Service-created KMS grants are no longer accepted by ID grammar alone.
   Volume, deletion-snapshot, and secret grants must equal the exact physical
   identity in the authenticated CloudFormation resource readback for the
   stated logical resource. Well-formed foreign `vol-*`, `snap-*`, and secret
   ARNs fail.
2. Public-trust scanning now rejects PEM PKCS7, CMS, and PKCS #7 signed-data
   containers plus RSA, ECDSA, Ed25519, security-key, and future algorithm
   OpenSSH certificate-key forms ending in
   `-cert-v01@openssh.com`.
3. The deletion inventory now contains a canonical exact
   `ListStackResources`/Describe resource readback and a separately hashed
   request/response evidence record. Resource type, logical ID, physical ID,
   ARN, source API, callback handler version, action family, resource grammar,
   and inventory membership are independently checked. The readback record
   also binds canonical hashes of the complete action/resources, ARN, ID, and
   name projection. The authenticated input and retained augmentation bind
   both readback hashes. A foreign retained role or a substituted callback
   version fails even after an attacker recomputes every caller-controlled
   hash.
4. The CloudFormation deletion role contains the complete mutation and
   describe/read surface. Mutation statements remain exact physical-resource
   allowlists. AWS APIs that do not support resource-level authorization are
   isolated to twelve enumerated read-only Describe actions with one
   `Resource: "*"` each and an exact `aws:RequestedRegion=us-west-2`
   condition; wildcard mutation or an unlisted wildcard read fails. The
   authenticated readback/evidence boundary, not the IAM wildcard, determines
   which resources may be acted upon.
5. The existing support deletion adapter in `fence_executor.py` now uses the
   frozen `RUN#glm52-sky-20260724`,
   `ACTIVATION#...#FINALIZATION_CONTROL`, and
   `ACTIVATION#...#FINALIZATION_ACTION#SUPPORT_DELETE#00000001` records.
   It authenticates the closed control/action owner, nonce, action kind,
   attempt, transition revisions, candidate, request, and inventory. A newly
   committed `CONSUMED` action is the durable
   `UpdateTerminationProtection(false)` intent. The exact false readback is
   hashed and persisted as the monotonic `COMPLETED` delete intent before the
   one exact-role `DeleteStack`. Only a live transaction commit may submit a
   mutation; durable adoption and ambiguity are read-only reconciliation.
   Crash-after-update and crash-after-delete tests prove neither call is
   resubmitted after process restart.
6. NAT observations are bound to an authenticated `NatGatewayId` that must
   equal the support-stack resource readback. The accumulator state and
   observation hash bind that ID, the frozen `RUN#...` partition key, the
   frozen finalization-control sort key, and a canonical Task 7 support-counter
   contract. No fictional `NAT_ACCUMULATOR#...` predecessor key or predecessor
   record schema is claimed; Task 7 declares its additive
   `ACTIVATION#...#TASK7_SUPPORT_COUNTER` sort key explicitly and binds it to
   the frozen control authority. The exact four-query `AWS/NATGateway`
   GetMetricData contract and cumulative metric dimensions include the exact
   NAT gateway ID, and foreign valid NAT IDs fail.
7. The minute schedule has an exact name, group, target input, 72-hour
   `EndDate`, and `ActionAfterCompletion: DELETE`. The retained lifecycle role
   can Get/Update/Delete only that exact schedule. Its pure executable
   retirement evaluator implements disable-once, readback, delete-once, and
   reconcile-only semantics. A retained Lambda-error alarm and an
   activation/NAT-bound observation-age alarm route to the exact retained
   function version with zero retry. Missing/delayed observations are
   `TreatMissingData: breaching`; after the immutable bootstrap grace the
   evaluator drains and disables work. Graph mutants for a foreign NAT
   dimension, missing schedule end, and fail-open delayed-metric handling are
   rejected.

### Fix-round-3 RED to GREEN

```text
public-trust and KMS identity RED:
2 failed, 33 deselected
-> focused public-trust/KMS family GREEN

authenticated deletion inventory/IAM boundary:
resource-readback schema RED
-> 3 passed, 34 deselected

durable deletion restart RED:
1 failed, 17 deselected
-> full fence-executor file 18 passed

NAT identity/counter/schedule/metric graph RED:
4 failed, 34 deselected
-> 4 passed, 34 deselected
-> support-plane production checks 37 passed, 1 generated-artifact check
   deselected

generated contracts refreshed:
full Task 7 focused suite 66 passed
```

### Fix-round-3 verification

```text
$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_support_plane.py \
    tests/test_glm52_enforcement_support_custom_resources.py \
    tests/test_glm52_enforcement_fence_executor.py
66 passed in 0.72s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py
606 passed, 2 warnings in 4.46s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.47s

$ PYTHONPYCACHEPREFIX=/tmp/glm52-task7-pycache-fix3 \
    /usr/bin/python3 -m py_compile \
    src/glm52_enforcement/support_plane.py \
    src/glm52_enforcement/support_custom_resources.py \
    src/glm52_enforcement/fence_executor.py \
    aws/glm52-gpu/scripts/build_h1g_support_plane.py
PASS under Python 3.9.6

$ PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 <import proof>
3.9.6 task7-python39-import-light-ok

regenerated four checked-in non-live Task 7 contracts
all aws/glm52-gpu/cfn/h1g/*.json canonical parse: 7 artifacts
owned-file trailing-whitespace audit: PASS
```

The two aggregate warnings remain the accepted SWIG deprecation warnings from
source-authority fixture parity.

### Fix-round-3 final SHA-256

```text
47f16eaeb5383f2229f2ee9c877490a490a0c5463ae57789f588b3d55cf8f517  src/glm52_enforcement/support_plane.py
f92d6cb0378adc19f4afdafa7025f195a831af81a1cf442c2ce71eafc80c8f24  src/glm52_enforcement/fence_executor.py
1bed6cfcaac08b7f834e78f4a3184ee8f174850f736fdf48d76cba0d1dd419a6  tests/test_glm52_enforcement_support_plane.py
12bdd6fc32d76fe9392b772e448abf5c9fd488cbd87595f3f5dcaeabc997e9c5  tests/test_glm52_enforcement_fence_executor.py
acde410e7c86e0e62a6bd8f8fc405dc31ce2632ae63ab13d4aa2a11b47c0c40f  aws/glm52-gpu/cfn/h1g/support-input-contract-v1.json
7944bb11fba8ffb0ab8a90c7b05a150d7842e1e4412c2b6a90b7493fc7b3e96d  aws/glm52-gpu/cfn/h1g/support-template-contract-v1.json
1fc1183c65349145cf8f05dca7f2f940cea396c72727d50c582e4b460bc9235e  aws/glm52-gpu/cfn/h1g/support-spend-envelope-v1.json
6d64df5b5d664522f37ddbb81d04dedd8d6f148e5cffdf971aa0fdfc9bd16af2  aws/glm52-gpu/cfn/h1g/support-contract-manifest-v1.json
```

### Fix-round-3 limitations

- These are deterministic contract, graph, restart-fake, and fixture proofs;
  they are not live AWS, IAM, CloudFormation, KMS, Scheduler, CloudWatch,
  deletion, metric, price, billing, or deployed evidence.
- The retained schedule fallback contract is mechanically bounded, but actual
  Scheduler `UpdateSchedule` request/readback behavior remains a live evidence
  gate.
- No AWS call, network call, deployment, live secret, Git mutation, commit, or
  push occurred in fix round 3.

## Reviewer fix round 4

Status: `DONE`

### Closed findings

1. The support-stack deletion path now distinguishes a durable
   definitely-not-sent pre-submit state from a possibly-sent state for both
   `UpdateTerminationProtection(false)` and `DeleteStack`. The frozen outbox
   progression is
   `UPDATE_PRE_SUBMIT -> UPDATE_POSSIBLY_SENT ->
   UPDATE_RECONCILED_FALSE -> DELETE_PRE_SUBMIT ->
   DELETE_POSSIBLY_SENT`, followed by exactly one of `DELETE_DIRECT` or
   `DELETE_AMBIGUOUS`. An adopted pre-submit owner gets one safe send; an
   adopted possibly-sent owner can only reconcile and never resubmit.
   Explicit restart tests cover all four adoption boundaries, process loss
   after each external side effect, and exact one-call cardinality.
2. The finalization action remains `CONSUMED` through the false-protection
   readback and delete pre-submit/possibly-sent states. A direct successful
   `DeleteStack` response alone binds `DELETE_DIRECT` and `COMPLETED`.
   Ambiguous transport or readback after possible submission binds
   `DELETE_AMBIGUOUS` and `AMBIGUOUS`. No state before the `DeleteStack`
   submission can complete the action.
3. Support construction is now explicitly two phase. `SupportBuildInputs`,
   `build_support_precreate_plane`, and the checked-in CLI build only the
   logical template; their exact field surface excludes `nat_gateway_id`,
   `support_stack_id`, and `support_deletion_inventory`. Runtime template
   coordinates use `Ref: AWS::StackId` and `Ref: NatGateway`. The separate
   injected `materialize_support_postcreate` and
   `build_support_postcreate_plane` path accepts service readback, not caller
   physical IDs.
4. Postcreate materialization requires the exact 113-resource template set.
   It paginates `ListStackResources`, calls `DescribeStackResource` for every
   logical resource, hashes each full response, and requires exact
   summary/detail stack, type, physical ID, and status agreement. It derives
   the combined-host root volume through `DescribeInstances` and the two
   Secrets Manager endpoint ENIs through `DescribeVpcEndpoints`. Every one of
   the 113 resources is classified into a frozen deletion action family and
   appears exactly once in the reviewed deletion authority. Named physical
   substitution, callback substitution, missing/extra/duplicate rows, an
   unclassified resource, and a postcreate snapshot fail.
5. The forensic snapshot is no longer represented as precreate or postcreate
   physical truth. Postcreate binds only `CAPTURE_AT_DELETION`, its
   authenticated source volume/KMS/tag contract, and `snapshot_id=None`.
6. NAT accounting no longer mutates or shares the finalization ledger. The
   retained augmentation contains `H1gTask7SupportCounterTable` with exact
   `CounterId =
   ACTIVATION#{activation_id}#NAT_PROCESSED_BYTES`, plus a dedicated NAT
   accumulator role, function, and version. Only that role gets
   `GetItem`/`UpdateItem` on the counter table with the exact leading key.
   The NAT function environment has no finalization control key or GLM-5.2
   ledger partition key. NAT schedule/metric authority is absent from the
   lifecycle role.
7. The generated precreate input contract now says caller physical IDs are
   rejected, authenticated postcreate materialization is required, and
   snapshot binding occurs at deletion. Deterministic regeneration changed
   only that input contract and its manifest; the template contract and spend
   envelope remained byte-identical.

### Fix-round-4 RED to GREEN

```text
durable outbox vocabulary RED:
missing SupportDeletionOutboxState import
-> enum contract GREEN

adopted UPDATE_PRE_SUBMIT RED:
zero update submissions where one safe send was required
-> pre-submit safe-send node GREEN

authenticated full inventory RED:
reviewed resource constructor rejected source-evidence fields
-> exact 113-row authority node GREEN

logical precreate RED:
missing SupportBuildInputs/build_support_precreate_plane imports
-> precreate field and logical-template nodes GREEN

postcreate materializer RED:
missing materializer import
-> 113 DescribeStackResource calls and derived-resource node GREEN

foreign named physical ID RED:
expected rejection did not occur
-> foreign S3 physical-ID substitution rejected

direct postcreate builder RED:
missing build_support_postcreate_plane import
-> retained augmentation binds authenticated inventory

dedicated NAT store RED:
missing H1gTask7SupportCounterTable
-> dedicated table/role/environment/routing node GREEN

final explicit restart-boundary audit:
fence executor 24 passed
```

### Fix-round-4 verification

```text
$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_support_plane.py \
    tests/test_glm52_enforcement_support_custom_resources.py \
    tests/test_glm52_enforcement_fence_executor.py
76 passed in 1.03s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py
616 passed, 2 warnings in 5.12s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.61s

$ PYTHONPYCACHEPREFIX=<fresh temp> \
    /Users/jack.mazac/Applications/Xcode-26.4.1.app/Contents/Developer/usr/bin/python3 \
    -m compileall -q -f src/glm52_enforcement
PASS under Python 3.9.6

$ PYTHONPATH=src PYTHONPYCACHEPREFIX=<same temp> \
    /Users/jack.mazac/Applications/Xcode-26.4.1.app/Contents/Developer/usr/bin/python3 \
    <recursive import-light proof>
python39-import-light-ok modules=16

$ /Users/jack.mazac/.cache/uv/archive-v0/0ZblHzvk4Jeduz1-/bin/ruff \
    check --select E4,E7,E9,F <Task-7-owned Python files>
All checks passed!

fresh temporary regeneration equals all four checked-in Task 7 contracts
all aws/glm52-gpu/cfn/h1g/*.json canonical parse: 7 artifacts
git diff --check: PASS
```

The two aggregate warnings remain the accepted SWIG deprecation warnings from
source-authority fixture parity.

### Fix-round-4 final SHA-256

```text
a3b5a568549e09609c5a5d4ab9a2c03db26f3e46ae8a1202fe91f2e2ff7e02fb  src/glm52_enforcement/support_plane.py
09b5f9dca389b462efb68e2640e576ab4703b19de178e47b0a7829e9729e2a2b  src/glm52_enforcement/fence_executor.py
75e064a0aceb542d099ff974e3cdb849136152dc602395cf6d78d921418bea18  aws/glm52-gpu/scripts/build_h1g_support_plane.py
703779416644678d4da1ed1396cc2231dde3e9108bbcec64c73bdc7c4ce81ca6  tests/test_glm52_enforcement_support_plane.py
ba2084e64e83969889a38c2024a178ebe4c062f4bedde5a669796de6e94372f9  tests/test_glm52_enforcement_fence_executor.py
96702fa553e8373c678046439fc36d6b58a60f987940c34e97b8055b891bc9c3  aws/glm52-gpu/cfn/h1g/support-input-contract-v1.json
7944bb11fba8ffb0ab8a90c7b05a150d7842e1e4412c2b6a90b7493fc7b3e96d  aws/glm52-gpu/cfn/h1g/support-template-contract-v1.json
1fc1183c65349145cf8f05dca7f2f940cea396c72727d50c582e4b460bc9235e  aws/glm52-gpu/cfn/h1g/support-spend-envelope-v1.json
3bae04cd884c372ca4e0764c79bbdbe2ffbbad2f89ac43f032521c4702e72f61  aws/glm52-gpu/cfn/h1g/support-contract-manifest-v1.json
```

### Fix-round-4 limitations

- These remain local contract, injected-service, graph, canonical-artifact,
  and restart-fake proofs. They are not live AWS, IAM, CloudFormation, EC2,
  KMS, Scheduler, CloudWatch, snapshot, deletion, metric, billing, or
  deployed evidence.
- No live postcreate inventory exists until an explicitly authorized
  deployment returns service-assigned identities and the injected
  materializer reads all 113 resources.
- The forensic snapshot ID intentionally remains absent until the separately
  authorized deletion-time capture.
- No AWS call, network call, deployment, resource launch, Git mutation,
  commit, or push occurred in fix round 4.

## Fix round 5: durable service ownership and non-forgeable postcreate route

### Outcome

Fix round 5 removes the two rejected process-local ownership designs from the
production surface.

1. Support-stack teardown is now owned by one exact retained Standard
   Workflow version:
   `keep-glm52-h1g-retained-lifecycle:<positive-version>`.
   `UpdateTerminationProtection` and `DeleteStack` are AWS SDK Task
   integrations in that definition. Neither mutation Task has `Retry`, and
   neither mutation is in a workflow cycle. Every ambiguous edge proceeds
   only to read-only `DescribeStacks` reconciliation. The execution succeeds
   only after the exact `does not exist` absence result.
2. Python can only authenticate live `SUPPORT_FINALIZED` readback, describe
   and start/adopt the deterministic exact-version execution, page and
   validate execution history, and reconcile exact stack plus reviewed
   113-resource absence. The local CloudFormation mutation adapter and durable
   outbox types were removed.
3. The retained augmentation now contains the exact Standard state machine,
   its retained role, and its immutable version. Only the state-machine role
   owns `UpdateTerminationProtection`, `DeleteStack`, and the exact
   CloudFormation service-role pass. The Lambda runtime role cannot call
   either mutation.
4. The workflow has a conservative one-hour, five-second-readback history
   ceiling of 7,984 events, below the frozen 12,000-event ceiling.
5. Postcreate materialization has no public caller-constructible
   materialization dataclass or direct downstream builder. One
   `coordinate_support_postcreate` boundary validates the canonical
   precreate template, exact stack ID, authenticated raw CloudFormation pages,
   all 113 `DescribeStackResource` responses exactly once, EC2 root-volume and
   endpoint-ENI readbacks, account/region/name allowlists, and callback/version
   bindings before the private value is immediately consumed.
6. `aws/glm52-gpu/scripts/materialize_h1g_support_plane.py` is the executable
   postcreate entrypoint. It requires `keep-gpu`, account `246813579024`,
   `us-west-2`, an exact support stack ARN, empty `AWS_PAGER`, bounded
   timeouts, and `total_max_attempts=1`. It emits only fresh canonical
   no-overwrite retained augmentation, deletion authority, and manifest
   artifacts.
7. The rejected Task 7 DynamoDB counter table and second NAT
   role/function/version family were removed. The one-minute Scheduler target
   now invokes the frozen retained lifecycle version. Its import-light core
   reads the four `AWS/NATGateway` byte counters over consecutive 60-second
   windows from activation start, bounded to 72 hours and 17,280 datapoints,
   and publishes the one activation-cumulative metric. AWS client construction
   lives in the executable Lambda shim outside the import-light package.

### Named RED to GREEN evidence

```text
test_support_delete_mutations_are_owned_by_one_exact_standard_execution
RED: missing Standard Workflow adapter/definition
GREEN: exact deterministic execution survives process loss after StartExecution

test_process_local_support_delete_adapter_is_not_production_reachable
RED: SupportDeletionAdapter still selected the process-local CFN caller
GREEN: the public name is the exact Standard Workflow adapter

test_postcreate_authority_has_no_public_forgeable_materialization
RED: public materialization dataclass and direct builder were present
GREEN: only the single coordinator can mint and consume private authority

test_retained_nat_observation_reuses_the_frozen_lifecycle_family
RED: second table/role/function/version family was rendered
GREEN: minute observation targets the existing retained lifecycle version

test_postcreate_cli_guards_identity_and_uses_only_injected_read_runner
GREEN after the real CLI was introduced; the injected runner proves STS-first
guarding, read-only client transfer, canonical no-overwrite output, and foreign
profile/account rejection without AWS access
```

The removed round-4 assertions were not skipped or deselected. Their still
valid adversarial properties are covered by stronger collected tests:

- direct/restart/pre-submit/possibly-sent mutation boundaries:
  exact Standard execution adoption, no process-local adapter, no mutation
  cycle, and exactly one scheduled event for each mutation;
- ambiguous update/delete outcomes: Catch edges are read-only and cannot
  return to either mutation;
- retained/fence targeting and unstable support prestates: support-only
  authority plus `UPDATE_COMPLETE` and termination-protection-true Choice
  predicates;
- syntactic finalization: no live finalization readback means zero
  `DescribeExecution` or `StartExecution` effect;
- residual replacement/presence: terminal reconciliation rejects any change
  to the reviewed 113-resource absence projection;
- NAT missing/delayed/cutoff/role isolation: existing fail-closed egress and
  schedule-retirement tests plus the new single-family runtime tests;
- foreign S3/account/callback identities and the full 113-resource inventory:
  the opaque coordinator and postcreate CLI tests.

### Fix-round-5 verification

```text
$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_support_plane.py \
    tests/test_glm52_enforcement_support_custom_resources.py \
    tests/test_glm52_enforcement_fence_executor.py
76 passed in 0.53s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py
616 passed, 2 warnings in 4.45s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.54s

$ PYTHONPYCACHEPREFIX=<fresh temp> \
    /Users/jack.mazac/Applications/Xcode-26.4.1.app/Contents/Developer/usr/bin/python3 \
    -m compileall -q -f src/glm52_enforcement
PASS under Python 3.9.6

$ PYTHONPYCACHEPREFIX=<fresh temp> <same Python 3.9> -m py_compile \
    aws/glm52-gpu/scripts/build_h1g_support_plane.py \
    aws/glm52-gpu/scripts/materialize_h1g_support_plane.py \
    aws/glm52-gpu/scripts/retained_support_lifecycle_handler.py
PASS

$ PYTHONPATH=src PYTHONPYCACHEPREFIX=<fresh temp> <Python 3.9 import guard>
python39-import-light-ok modules=16

$ ruff check --select E4,E7,E9,F <Task-7-owned Python files>
All checks passed!

fresh temporary regeneration equals all four checked-in Task 7 contracts
all aws/glm52-gpu/cfn/h1g/*.json canonical parse: 7 artifacts
materialize_h1g_support_plane.py mode: executable
git diff --check: PASS
```

The two aggregate warnings remain the accepted SWIG deprecation warnings from
source-authority fixture parity.

### Fix-round-5 final SHA-256

```text
c1a5afc60769fc246f6a52fffa04773e4d21f45e7b618bf514bc84847e6b9099  src/glm52_enforcement/support_plane.py
94faeef339016df9717c17924d2731418470e336cc380694083c10421a726f53  src/glm52_enforcement/fence_executor.py
daae58a98503e06516dba2fd9d208de4a0bc28799af4c667d7379adb12959c82  src/glm52_enforcement/retained_support_lifecycle_handler.py
3da80fac3418b6786f2a6ca98927d4ad9bad552edf9fbb5f410534babb48b00e  aws/glm52-gpu/scripts/materialize_h1g_support_plane.py
e1d2b3cef46562bcd819f31ddabcdb0170e4ac9eda8e6fa2af4b9aefe3c109d0  aws/glm52-gpu/scripts/retained_support_lifecycle_handler.py
87655d876039fa4b092a38ecd3d06e2480e83c1af0445b5139710e1e18c8cfdd  tests/test_glm52_enforcement_support_plane.py
1b14fbd3ebc6ff5b2fd0aeefe33ecd242b5bb521038952f4839d3844bd260e72  tests/test_glm52_enforcement_fence_executor.py
```

### Fix-round-5 limitations

- These are local graph, injected-service, canonical-artifact, history-fake,
  and Python compatibility proofs. They are not live AWS, IAM,
  CloudFormation, Step Functions, EC2, Scheduler, CloudWatch, billing,
  deletion, or deployed evidence.
- The executable postcreate route was not invoked against AWS. No live
  postcreate inventory or deletion authority exists until a separately
  authorized deployment supplies the exact service-assigned stack ID.
- The Standard Workflow exactly-once contract is proven structurally here;
  no live execution history exists in this task.
- No AWS call, network call, deployment, resource launch, Git mutation,
  commit, or push occurred in fix round 5.

## Controller closure: final independent-review findings

The final independent review rejected fix round 5 on two concrete production
contracts. Both were repaired directly before Task 7 closure.

1. Postcreate authority could still be forged from module-accessible private
   mint machinery. The mint capability, constructible materialization class,
   and downstream materialization consumer are now absent. The sole builder
   accepts only the canonical precreate inputs/template, exact stack ID, and
   service clients; it materializes and consumes all live truth within one
   call. A regression assertion verifies that no public or private mint
   capability/class/consumer remains and that the builder has no
   `materialization` parameter.
2. One datapoint in a 60-minute NAT query could authorize work, and missing
   post-grace data did not itself drain the host. The retained runtime now
   requires exact consecutive 60-second timestamps from activation start.
   Partial coverage is `DELAYED`; empty coverage is `MISSING`. After the
   frozen grace deadline, either state calls exact `DescribeInstances` and
   `DescribeSecurityGroups`, stops the authenticated combined host if
   necessary, and revokes its exact tagged egress permission if present.
   The retained role grants only the two required EC2 reads and exact
   stop/revoke mutations. A retained 16/16, missing-is-breaching alarm closes
   the independent observation hole without adding another runtime family.

### Controller-closure verification

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_support_plane.py \
    tests/test_glm52_enforcement_support_custom_resources.py \
    tests/test_glm52_enforcement_fence_executor.py
78 passed in 0.56s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py
618 passed, 2 warnings in 4.71s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.51s

Python 3.9 compileall/py_compile: PASS
Ruff E4/E7/E9/F: PASS
fresh temporary regeneration equals all four checked-in Task 7 contracts
git diff --check: PASS
```

The two aggregate warnings remain the accepted SWIG deprecation warnings from
source-authority fixture parity.

### Controller-closure SHA-256

```text
ae603b4b369dfcb714f762a00518a92764847045c07b1d9d442611369837c8f4  src/glm52_enforcement/support_plane.py
eba6075569d0052440769f70e9ad7900545a10ebc014f40496fcdc48368f088e  src/glm52_enforcement/retained_support_lifecycle_handler.py
d99f8c087eef6627e562b1cbe7d2533849440d2ad96f322c2d260d9c22becad5  aws/glm52-gpu/scripts/retained_support_lifecycle_handler.py
f5623d6a4017d4af061a1960433e4164244e6b0305503e3e888096c8cb130523  tests/test_glm52_enforcement_support_plane.py
```

### Controller-closure timing correction

The closure re-review confirmed the two prior attacks were closed and found
one Important production timing defect: Scheduler delivery and Lambda startup
can occur off the activation-aligned minute, while the response validator
requires a whole number of 60-second windows. The retained runtime now derives
the last completed activation-aligned minute from the wall clock, caps it at
the frozen 72-hour limit, and uses that cutoff for both the CloudWatch request
and response-coverage proof. The actual wall clock remains the authority for
the post-grace drain decision. A post-grace invocation at `01:00:05Z` now
queries through `01:00:00Z`, classifies empty data as `MISSING`, and performs
the exact host stop plus egress revoke.

```text
focused Task 7: 79 passed in 0.58s
Tasks 1-7 enforcement aggregate: 619 passed, 2 warnings in 4.71s
frozen CloudFormation: 26 passed in 1.52s
git diff --check: PASS

041d45831458dff6d6d20573f5890af28dde7be9ff7d7657fcc221fa62e4a4c2  src/glm52_enforcement/retained_support_lifecycle_handler.py
052240729f0d8bcc469866f0399b5747bb9e33e135b56b537094852cbe2e8d23  tests/test_glm52_enforcement_support_plane.py
```

## Independent closure review

Task 7 is approved with no Critical or Important findings at immutable target
`bb6370aab920fc1887985bbeb8e18f113d266ea8`. The reviewer independently
confirmed off-minute and 72-hour-plus-latency query alignment, wall-clock
grace enforcement, partial/missing classification, exact host/egress drain,
and the direct 72-hour request cap. Focused independent verification was
79 passed.
