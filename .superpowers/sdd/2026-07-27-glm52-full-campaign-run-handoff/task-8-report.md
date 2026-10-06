# Task 8 Report — live H.1d and authenticated spend authority

Date: 2026-07-29

## Status

`DONE`

Task 8 implements the fresh, fail-closed H.1d authority walk, authenticated
append-only GPU-spend reconstruction and reservation contract, and an
executable production read adapter. The ordinary H.1d route is read-only. The
reserve method is separate and explicit.

No predecessor file was modified. No Git mutation, AWS call, network call,
deployment, launch, reserve write, spend, or billing effect occurred.

## Owned files

- `src/glm52_enforcement/live_authority.py`
- `src/glm52_enforcement/spend_authority.py`
- `aws/glm52-gpu/scripts/inspect_h1d_live_authority.py`
- `aws/glm52-gpu/scripts/materialize_h1d_trusted_sources.py`
- `tests/test_glm52_enforcement_live_authority.py`
- `tests/test_glm52_enforcement_spend_authority.py`
- `tests/test_glm52_h1d_trusted_source_materializer.py`
- `.superpowers/sdd/2026-07-27-glm52-full-campaign-run-handoff/task-8-report.md`

## Implemented contracts

### Fresh H.1d live authority

- Closed typed expected-state, request, response-page, Task 6/7 source
  authentication, Task 9 Sky/relay probe, and result records have canonical
  SHA-256 identities.
- One coordinator exact-reads STS plus the required CloudFormation, Lambda,
  IAM, EventBridge, Scheduler, SQS, SNS, EC2, S3, DynamoDB, SSM, CloudWatch,
  and Logs families. It requires the exact closed family sequence, complete
  monotonically indexed pagination, unique tokens/pages/resources, typed
  direct response metadata, and exact equality with authenticated expected
  items.
- The expected-state authority exact-reads canonical Task 6 migration
  manifest/template bytes and Task 7 postcreate manifest/inventory bytes by
  S3 VersionId and file SHA-256 before service inspection. It verifies
  self-hashes, cross-bindings, Task 6 template hashes, the Task 7 support
  template hash, every expected family identity, and the combined-host
  immutable EC2 shape. Direct production-class regressions prove that
  caller-rehashed EC2 and IAM substitutions are rejected.
- Alarm/metric, DLQ, confirmed SNS, exact SSM host, protected bucket, no-P5,
  no-foreign-instance, must-start, execution-deadline, no-open-spend, exact
  Task 9 probe, and seven-second phase checks fail closed. Necessary
  observations such as alarm `OK`, approximate queue zero, and SSM `Online`
  never bypass exact-state equality or the other authorities.
- The in-memory result binds caller, authenticated expected state, every page,
  every family, spend, Task 9 probe, and measured phase interval. Serialized
  output is explicitly `NON_AUTHORITATIVE_EVIDENCE`; there is no stored-result
  input route.

### Authenticated GPU spend

- Exact GPU approval and descriptor validators bind the accepted Kon approval
  source without importing model/ML code.
- The inspector exact-reads the descriptor, approval, `LATEST`, signed
  snapshot, immutable ledger records, and optional held reserve by VersionId;
  fully paginates namespaces; rejects delete markers, duplicate keys,
  unexpected versions, repeated tokens, forks, reordering, count/tip/
  predecessor drift, self-hash mutation, stale/future/foreign records, and
  allocation/EC2 disagreement.
- Decimal-only aggregate-second arithmetic enforces exactly 86,400 GPU
  seconds and `$1,320.96`, with used, open, reserved, remaining, and refundable
  balances reported without double counting. Open or unresolved allocations
  are charged and nonrefundable.
- The separate reserve operation requires exactly 900 seconds and `$13.76`,
  binds the separately approved 300-GiB worker-root tail capped at `$0.01`,
  authenticates exact residual-liability approval bytes, persists intent
  before one conditional write, uses no second submission, and permits one
  exact readback only for ambiguous transport. Exact duplicate adoption is
  read-only; stale head, insufficient balance, mismatch, definite rejection,
  second reserve, or absent authenticated settlement fails closed. Task 8
  provides no refund mint.

### Executable read adapter

- `inspect_h1d_live_authority.py` is executable and requires exact profile
  `keep-gpu` and region `us-west-2`.
- STS is the first AWS operation and every foreign account stops before
  reading the expected-state file, manifests, or service families.
- The production adapter uses a closed read-only in-process AWS SDK inventory
  with one-attempt botocore configuration, bounded concurrent family fanout,
  deterministic service-native pagination, exact S3 VersionIds, canonical
  input, and canonical no-overwrite output. The AWS CLI implementation remains
  available as an explicit debug boundary with five-second process timeouts,
  `AWS_PAGER=""`, `AWS_MAX_ATTEMPTS=1`, and
  `AWS_RETRY_MODE=standard`. Cost Explorer, Budgets, anomaly detection, and
  mutating verbs are absent from both boundaries.
- The default Task 9 adapter deliberately fails closed. Task 9 must inject the
  real exact mTLS/attestation/admission/Sky-identity probe before this route can
  produce a successful live authority result.

## Literal RED to GREEN

Initial named-test collection, before either enforcement module existed:

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py
ERROR tests/test_glm52_enforcement_live_authority.py
ModuleNotFoundError: No module named 'glm52_enforcement.live_authority'
ERROR tests/test_glm52_enforcement_spend_authority.py
ModuleNotFoundError: No module named 'glm52_enforcement.spend_authority'
2 errors during collection
```

The first executable-adapter checkpoint had four named CLI tests RED because
`inspect_h1d_live_authority.py` did not exist. They became GREEN through the
real command runner/reader and not through a marker or suppressed assertion.

Independent review then exposed a production-authentication RED: the real
`AwsExpectedStateAuthority` authenticated source coordinates but did not
derive and compare the caller-supplied expected resources. The production
class now derives identities and EC2 immutable shape from the exact Task 6/7
bytes. Direct real-class EC2 and IAM rehash probes are GREEN. A subsequent
test-fixture correction produced the literal intermediate result
`2 failed, 75 passed`; the corrected exact SQS physical identity then yielded
`77 passed`, and the real three-page CLI pagination regression brought the
final focused result to `78 passed`.

Final focused GREEN:

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py
78 passed in 0.36s
```

## Verification

Tasks 1–8 enforcement aggregate including migration:

```text
$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py
697 passed, 2 warnings in 16.42s
```

The warnings are the accepted SWIG deprecation warnings in the existing
source-authority fixture parity test.

Frozen CloudFormation regression:

```text
$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 3.99s
```

Python 3.9.6:

```text
$ <Xcode Python 3.9.6> -m py_compile \
    src/glm52_enforcement/live_authority.py \
    src/glm52_enforcement/spend_authority.py \
    aws/glm52-gpu/scripts/inspect_h1d_live_authority.py
PASS

$ PYTHONPATH=src <Xcode Python 3.9.6> <blocked-import guard>
Python 3.9 blocked-import proof: PASS
```

The guard rejects `boto3`, `botocore`, `mlx`, `mlx_vq`, `numpy`, `torch`, and
`transformers` while importing both package modules and the executable
adapter.

Static and whitespace gates:

```text
$ uvx --offline ruff check --select E4,E7,E9,F <Task-8-owned Python files>
All checks passed!

$ git diff --check
PASS

inspect_h1d_live_authority.py mode: executable
```

## SHA-256

```text
44390d069119b938c19b4cc878a1fa9f84adb6016c9e90b778e5c1127935ffcb  src/glm52_enforcement/live_authority.py
62a7d522f01f3743a5fcc791e0c4a2ce0f11cc2fe78e63c2e493739e0f217a48  src/glm52_enforcement/spend_authority.py
97dd81419b79791c4b6cd0cebf8f150731d17391fda77e9a49e9388c82d7545e  aws/glm52-gpu/scripts/inspect_h1d_live_authority.py
be072c2413321e2943d47b2d2f72eed4a3e679cd03d94c9a901280597695c3ab  tests/test_glm52_enforcement_live_authority.py
5c0a0c74f72033c8c25b0475640a0ff8f3202916cea102cf39d01a0a9a34b13b  tests/test_glm52_enforcement_spend_authority.py
```

## Self-review and limitations

- The production expected-state substitution gap identified by independent
  review is closed in the real `AwsExpectedStateAuthority`; it is not covered
  only by the injected fake.
- The AWS adapter is a genuine executable read route, but its service
  projections are supplied by authenticated expected-state input and remain
  fail-closed if a required Task 6/7 physical identity is absent.
- The focused suite uses injected service/process boundaries. It proves exact
  commands, parsing, pagination, retry absence, canonical identities,
  arithmetic, and mutation handling without making live AWS or deployed-truth
  claims.
- No live AWS, STS, S3, EC2, DynamoDB, CloudFormation, Sky, mTLS, reserve,
  spend, launch, deployment, billing, or campaign-running claim is made by
  this task.
- Task 9 still owns the exact live Sky/relay probe and reservation consumer;
  Task 11 owns the full 58-second authority suffix and 600-second measured
  closure.

## Independent-review fix round 1

Status: `DONE`

The first independent review rejected Task 8 with six Critical and two
Important production findings. All eight were reproduced with real-class or
real-adapter adversarial tests before production changes.

### Literal fix-round RED

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py \
    -k '<15 independent-review adversarial cases>'
15 failed, 78 deselected in 0.21s
```

The 15 failures were:

1. rehashed IAM inline-policy, Lambda code, and EC2 network identities;
2. a caller-restricted EC2 inventory plan;
3. three Sky-probe request, activation, and stale-observation replays;
4. execution-deadline expiry during the phase;
5. IAM detail-operation pagination truncation;
6. an absent S3 response `VersionId`;
7. cached STS outside the measured phase;
8. a production spend service without held-reserve enumeration;
9. the invented EC2 `TerminatedAt` field;
10. an authenticated closed allocation aged out of EC2;
11. a caller-forged, self-hashed `SpendAuthorityResult` passed to reserve.

Two self-review regressions were then introduced RED-first:

```text
Sky token expires before phase completion:
1 failed, 65 deselected in 0.06s

HELD reserve points to an authenticated historical ledger node:
1 failed, 27 deselected in 0.06s
```

### Fixes

- The real `AwsExpectedStateAuthority` now directly derives IAM inline-policy,
  Lambda code, EC2 network, EC2 host/storage, and every physical family
  identity from exact-version, exact-hash Task 6/7 sources. The complete live
  spec/config identity is source-bound. A production EC2 read is mandatory
  and its inventory arguments are closed to the exact campaign-tag scope;
  caller-added instance filters cannot hide tagged P5 or foreign campaign
  instances.
- `reserve_gpu_liability` no longer accepts a caller-supplied current result.
  It requires the typed spend request and services and reruns the complete
  authenticated descriptor/approval/ledger/snapshot/EC2/reserve inspection
  inside the reserve boundary before any writer read or write.
- The executable production spend service always installs
  `AwsCliReserveReader`. It paginates the fixed run reserve namespace,
  exact-reads every record version, requires canonical bytes, and fails closed
  if the boundary is absent. A HELD reserve remains charged when its
  predecessor is any authenticated node in the current append-only chain, not
  only the latest tip.
- Every detail operation now declares its own pagination paths and token
  argument. All pages are fetched with monotonic, nonrepeating tokens and all
  page response identities are bound before projection. Hidden later IAM
  policies, targets, or versions cannot be silently dropped.
- EC2 spend reconstruction no longer reads nonexistent `TerminatedAt`.
  Every open allocation must still exist live and match exactly. Closed
  instances may age out of `DescribeInstances`; the authenticated immutable
  allocation-ended ledger record remains the exact closure authority, while
  any still-visible closed instance must match its immutable identity,
  launch, type, lifecycle, AZ, tags, and terminal state.
- A Sky probe now binds the canonical exact probe request, account, region,
  run, and activation. Its observation must occur within the current measured
  phase, and its token must remain valid through phase completion.
- The executable keeps the initial stop-before-input STS account guard and
  performs a second fresh STS read inside the measured H.1d coordinator.
  Completion rechecks both the seven-second ceiling and signed execution
  deadline.
- Exact S3 object reads now require the response `VersionId` to be present and
  byte-for-byte equal to the requested version. Missing metadata is no longer
  accepted.

### Fix-round verification

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py
94 passed in 0.20s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py
713 passed, 2 warnings in 5.11s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.55s

Python 3.9 compile and blocked-import proof: PASS
offline Ruff E4/E7/E9/F: PASS
git diff --check: PASS
inspect_h1d_live_authority.py executable mode: PASS
```

The two aggregate warnings remain the accepted SWIG deprecation warnings in
the existing source-authority parity test.

### Fix-round SHA-256

```text
bc8596e09bcaf65824c39246c899dd856f5fe3b307954db34807e2c2d87c0ca8  src/glm52_enforcement/live_authority.py
9c351bf723429c4fef30774a022b00964bdce8c3ddc5b21c5e046b864188f9db  src/glm52_enforcement/spend_authority.py
20dff624c65e37b88ee765259212648c5e7f92de938e307e6016e6eb47fb2032  aws/glm52-gpu/scripts/inspect_h1d_live_authority.py
7c34d8cce7b22ebb38dbdfbd77cc5c5484fbf4a37d5ccf31814d989ad28e7ea5  tests/test_glm52_enforcement_live_authority.py
d7478a0307afa4ada6eba169d143a225a3d31ec2660c8d029566ff5ef2dd69b3  tests/test_glm52_enforcement_spend_authority.py
```

### Fix-round limitations

- All fix-round service and process boundaries remain injected. No live AWS,
  network, reserve, spend, launch, deployment, billing, or campaign-running
  claim is made.
- Task 9 must still supply the real exact mTLS Sky/relay probe and reserve
  writer. The default probe remains deliberately unavailable.
- Task 11 still owns the full 58-second authority suffix and measured
  600-second closure.

## Independent-review fix round 2

Status: `DONE`

The second review found that the executable omitted account and region from
the exact Sky-probe request, and that a caller could replace and rehash the
template bundle together with its S3 coordinate. Final review also required
complete all-family detail-command closure, canonical detail transforms,
reachable exact-duplicate reserve adoption, and canonical AWS timestamp
normalization.

### Literal fix-round-2 RED

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py \
    -k '<10 fix-round-2 adversarial cases>'
10 failed, 92 deselected in 0.20s
```

Those failures covered four independent-source coordinate/version/digest
mutants, a rehashed IAM inventory restriction, missing IAM detail commands,
missing canonical detail-page hashing, a `+00:00` EC2 launch timestamp, the
incomplete executable Sky request, and a freshly enumerated exact HELD
reserve that could not be adopted without a second debit.

### Fixes

- The executable now reads the four Task 6/7 source coordinates from the
  independently addressed, strongly consistent
  `RUN#glm52-sky-20260724 | H1D_TRUSTED_SOURCE` record in the fixed
  `keep-glm52-h1g-ledger-v1` table. The record must be `SEALED`, has an exact
  attribute schema, stores canonical JSON, and binds its source set by
  canonical SHA-256. Caller-envelope coordinates must equal that source set
  before any source artifact is accepted.
- Every live family now has a closed inventory response/pagination contract.
  Every non-inventory command must appear once, in order, with an
  identity-bound argument and its exact pagination shape. Inventory
  restrictions, missing commands, unprojected detail responses, incomplete
  expected fields, or a wrong transform fail closed. Every expected
  `*_sha256` is derived with `canonical_sha256`; multi-detail projections bind
  all contributing responses.
- The real CLI reader applies `identity` or `canonical_sha256` transforms
  after completely paginating each detail operation. The focused IAM
  regression proves that two policy pages produce the canonical hash of the
  complete `["A", "B"]` sequence.
- The executable Sky-probe request now contains exact account, region, run,
  and activation fields, matching the core request digest contract.
- An exact duplicate reserve is adoptable only when the fresh held-reserve
  enumeration already charges exactly 900 seconds and `$13.76` and the
  writer's exact read matches the entire record. Adoption performs no intent
  write, no submit, and no second balance debit. An unenumerated, foreign, or
  unresolved held liability remains fail-closed.
- AWS EC2 `LaunchTime` is parsed as a timezone-aware instant and normalized to
  canonical UTC `Z` before comparison with the authenticated ledger.

### Fix-round-2 verification

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py
103 passed in 0.19s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py
722 passed, 2 warnings in 5.49s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 2.54s

Python 3.9.6 compile and blocked-import proof: PASS
offline Ruff E4/E7/E9/F: PASS
git diff --check: PASS
inspect_h1d_live_authority.py executable mode: PASS
```

The two aggregate warnings remain the accepted SWIG deprecation warnings in
the existing source-authority parity test.

### Fix-round-2 SHA-256

```text
bc8596e09bcaf65824c39246c899dd856f5fe3b307954db34807e2c2d87c0ca8  src/glm52_enforcement/live_authority.py
45bae66c683d0a0cf2e5ae11b7d24f4fa39f2cace419ec86d9ee5f9492a6cc46  src/glm52_enforcement/spend_authority.py
15dd9e4dde557294464dc4a83f246c4aa8f7345928e529856ca8e9359aae902f  aws/glm52-gpu/scripts/inspect_h1d_live_authority.py
51dc348cc5770bcad27f6cbfcbdf2224ba793ec45925a5933b8e27d9c80b6812  tests/test_glm52_enforcement_live_authority.py
c13f744239110f21ccb586052e7d98cc87736f5f43ffa1736247e5baaa34b68b  tests/test_glm52_enforcement_spend_authority.py
```

### Fix-round-2 limitations and integration boundary

- The fixed trusted-source record is a required production input. This task
  defines and tests its exact read contract but does not create it because
  Task 8 is read-only. The authorized deployment/control-plane phase must
  materialize the SEALED record before invoking this adapter.
- All service, object-store, DynamoDB, and process boundaries remain injected
  in tests. No live AWS, network, reserve, spend, launch, deployment, billing,
  or campaign-running claim is made.
- Task 9 must still inject the live exact Sky/relay probe and reserve writer.
  Task 11 still owns the full 58-second authority suffix and measured
  600-second closure.

## Independent-review fix round 3

Status: `DONE`

The third review found four remaining production-contract gaps: the caller
could still supply generic detail-command arguments instead of the executable
owning an operation-specific AWS query graph; an omitted terminal AWS
pagination token was rejected; the trusted-source read happened before the
measured H.1d phase; and the spend observation was not bound to the current
phase. It also required an executable production materializer for the
independent SEALED trusted-source record.

### Literal fix-round-3 RED

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py \
    tests/test_glm52_h1d_trusted_source_materializer.py \
    -k '<9 fix-round-3 adversarial cases>'
9 failed, 73 deselected in 0.57s
```

The failures covered a stale spend observation, the production trusted-source
record schema, IAM and EC2 dependent-identifier query graphs, an omitted
terminal inventory token, a trusted-source read before the measured core, and
the three missing materializer behaviors.

The stale-spend fixture was then rebuilt with a canonical matching identity so
the named regression proved the intended defect rather than failing at an
earlier identity gate:

```text
1 failed, 78 deselected
Failed: DID NOT RAISE
```

Local AWS CLI input-skeleton validation subsequently exposed a nonexistent
`cloudformation describe-termination-protection` operation. A named RED test
captured the defect:

```text
1 failed, 79 deselected
```

The invalid operation was removed; termination protection is now read from
the `EnableTerminationProtection` field returned by `DescribeStacks`.

### Fixes

- The authenticated CLI plan is now schema version 2 and contains projections
  only. The executable owns every family operation, command order, inventory
  restriction, pagination contract, dependent identifier, and exact detail
  argument. A caller can no longer substitute generic
  `--resource-identity` arguments and rehash the plan.
- IAM detail reads are derived from live role, inline-policy, attached-policy,
  managed-policy, and default-version identifiers. EC2 volume, image, instance
  user-data, and launch details are likewise derived from live inventory.
  CloudFormation, Lambda, EventBridge, Scheduler, SQS, SNS, S3, DynamoDB, and
  CloudWatch use code-owned operation-specific arguments.
- Every code-owned inventory and detail command is fully paginated. An absent
  terminal token is accepted as terminal, while a malformed or missing parent
  path still fails closed. Canonical projections cover every contributing
  response and require `canonical_sha256` for digest fields.
- The independent trusted-source read now occurs inside
  `Glm52LiveAuthorityCore.authenticate()`, and therefore inside its measured
  seven-second H.1d phase. The production reader performs one exact,
  strongly-consistent read of the fixed
  `RUN#glm52-sky-20260724 | H1D_TRUSTED_SOURCE` record.
- `materialize_h1d_trusted_sources.py` provides the production SEALED-record
  creation path. It verifies STS account and region, accepts one canonical
  source input with its supplied SHA-256, computes the record self-hash,
  performs one conditional absent-only `PutItem`, then performs one exact
  strongly-consistent readback. It fixes attempts at one and exposes no hidden
  retry path.
- Spend authority now rejects an observation whose timestamp is before the
  current H.1d phase start or after its end.

### Fix-round-3 verification

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
111 passed in 0.17s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
730 passed, 2 warnings in 4.81s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.53s

Python 3.9.6 compile and blocked-import proof: PASS
offline Ruff E4/E7/E9/F: PASS
git diff --check: PASS
both production scripts executable mode: PASS
AWS CLI v2 closed-operation input syntax validation: PASS
```

The AWS CLI check used local `--generate-cli-skeleton input` validation for
every closed operation and representative exact required arguments; it made
no AWS service call. The two aggregate warnings remain the accepted SWIG
deprecation warnings in the existing source-authority parity test.

### Fix-round-3 SHA-256

```text
9a030c7d64e1477bff37ba9f7fd953dff2d1cb7f09ad713a6e6a7dd66528fb2d  src/glm52_enforcement/live_authority.py
45bae66c683d0a0cf2e5ae11b7d24f4fa39f2cace419ec86d9ee5f9492a6cc46  src/glm52_enforcement/spend_authority.py
0ed02069328075fc7eb8b27c134456a70ebfc131bec0f8223b398f9265a23608  aws/glm52-gpu/scripts/inspect_h1d_live_authority.py
0bf3b0cb8cdf19482c53661ec5926f6aa280310577ec87913a088bd6813ca073  aws/glm52-gpu/scripts/materialize_h1d_trusted_sources.py
f1e65618959e510a23f81a9596bb934a7a85a21b3497e63c55cd2ba0ae7cd52b  tests/test_glm52_enforcement_live_authority.py
c13f744239110f21ccb586052e7d98cc87736f5f43ffa1736247e5baaa34b68b  tests/test_glm52_enforcement_spend_authority.py
d30e466ccc7da0d29fb919c64f856512d5701e28e8fe4ecb5b0f04f5d5182c7d  tests/test_glm52_h1d_trusted_source_materializer.py
```

### Fix-round-3 limitations and integration boundary

- The dedicated materializer now implements the required production creation
  path, but it was not invoked in this no-AWS task. The authorized control
  phase must execute it before the H.1d adapter performs its fixed read.
- All AWS, object-store, DynamoDB, and process boundaries remain injected in
  tests. No live AWS, network, reserve, spend, launch, deployment, billing, or
  campaign-running claim is made.
- Task 9 still owns the live exact Sky/relay probe and reserve writer. Task 11
  still owns the full 58-second authority suffix and measured 600-second
  closure.

## Independent-review fix round 4

Status: `DONE`

The fourth review found that the prior executable route still serialized AWS
CLI subprocesses, authenticated Lambda function inventory at unqualified ARNs,
used CLI rather than DynamoDB service pagination, did not preserve every
service RequestId, could miss altered or untagged P5 instances, and trusted a
caller-rehashable Task 8 spec claim. It also found that the independent
trusted-source materializer sealed only S3 coordinates rather than the
semantics and activation-scoped DynamoDB state needed by H.1d.

### Literal fix-round-4 RED

The first production-route batch was RED as nine named regressions:

```text
9 failed, 79 deselected in 0.39s
```

Those failures covered the missing default SDK route, bounded concurrent
reader, DynamoDB trusted-row exclusion, qualified Lambda version fanout,
universal debug `--no-paginate`, two exact S3 absence classes, the shared
seven-second preparation deadline, and S3 service-native `MaxKeys`.

The second batch was RED as four remaining request-ID and EC2-union
regressions:

```text
4 failed, 2 passed, 88 deselected
```

The semantic-source batch was RED as six regressions covering rehashed
CloudFormation, EventBridge, Scheduler, and S3 claims, retained-versus-
rehearsal bucket health, and activation-scoped DynamoDB identities:

```text
6 failed, 94 deselected
```

The materializer expansion was then RED because the sealed record lacked its
derived semantic source digest and exact current DynamoDB inventory:

```text
1 failed
```

Finally, a real DynamoDB response-shape regression was RED because the reader
attempted caller-style fields on typed DynamoDB items instead of projecting
the strongly consistent query rows itself.

### Fixes

- The ordinary production route now lazily constructs `AwsSdkCommandRunner`
  and `AwsSdkLiveReader`. The SDK boundary has an exact read-only operation and
  argument allowlist; one-second connect and two-second read timeouts; one
  total attempt; a fourteen-connection pool; and no import-time boto
  dependency. Missing `boto3`/`botocore` fails closed with an explicit error.
- The reader prepares all fourteen service-family walks with a bounded
  fourteen-worker executor under the core's single seven-second deadline. It
  cancels unfinished work on deadline or first failure and exposes only the
  completely prepared typed pages to the core.
- Every SDK response/page must carry a real `ResponseMetadata.RequestId`.
  Live pages authenticate the complete unique RequestId sequence. STS, exact
  S3 reads, S3 version listings, and EC2 spend listings preserve the service
  RequestId rather than substituting a local response hash. The CLI debug
  boundary retains a deterministic hash only when injected fixtures omit
  response metadata.
- Lambda discovery now enumerates every published numeric version, rejects
  `$LATEST` and malformed/duplicate version ARNs, and binds version-sensitive
  detail reads to the qualified ARN. EC2 discovery unions the campaign-run
  tag, exact `p5.48xlarge` instance type, and exact run-name tag queries and
  rejects inconsistent duplicate instance identities.
- DynamoDB uses `LastEvaluatedKey` and canonical
  `ExclusiveStartKey`, never CLI `NextToken`/`StartingToken`. The live family
  performs one fully paginated strongly consistent campaign-partition query,
  excludes exactly the fixed trusted-source metadata row, and code-projects
  each full typed item into its key, record type, full-item SHA-256, and
  consistent-read proof. It performs no per-row rescan.
- S3 spend enumeration uses service-native `MaxKeys=1000`. Lifecycle and
  replication absence is accepted only for the exact authenticated
  `NoSuchLifecycleConfiguration` and `ReplicationConfigurationNotFoundError`
  codes. Live health distinguishes the retained model/evidence bucket from
  the Task 7 support-rehearsal bucket and enforces each class's exact
  versioning/lifecycle/replication semantics.
- The materializer now exact-reads the four immutable S3 source objects by
  fixed campaign bucket, key, VersionId, and file SHA-256; validates canonical
  bytes, self-hashes, campaign identity, record types, Task 6/7 cross-hashes,
  support-template identity, activation, and retained bucket; and derives a
  semantic digest that deliberately excludes the caller's
  `h1d_specs_identity_sha256` claim.
- The materializer also fully paginates a strongly consistent query of the
  fixed campaign partition, excludes only its own trusted-source row, requires
  the current activation index/control/Sky action, and seals every full
  DynamoDB item identity into the trusted record. The production authority
  re-derives the semantic source digest from the exact S3 bytes and requires
  the expected DynamoDB set to equal that sealed inventory before live
  inspection.
- Debug CLI commands always add `--no-paginate`; service pagination is owned
  explicitly by the reader. S3 list-object-versions does not combine mutually
  exclusive CLI `--max-items` and `--no-paginate`.

### Fix-round-4 verification

```text
$ .venv/bin/pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
140 passed in 0.26s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
759 passed, 2 warnings in 5.62s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.63s

Python 3.9.6 compile: PASS
offline Ruff E4/E7/E9/F: PASS
git diff --check: PASS
both production scripts executable mode: PASS

boto3 1.43.51
botocore 1.43.58
```

The two aggregate warnings remain the accepted SWIG deprecation warnings in
the existing source-authority parity test. The boto dependencies were present
in the synchronized campaign runtime for this final verification; dependency
pinning is owned by the controller rather than Task 8.

### Fix-round-4 SHA-256

```text
8eb4d83ca18ba2c3147e28975ff61ef635de4dfcb39da79c0430760869dff151  src/glm52_enforcement/live_authority.py
45bae66c683d0a0cf2e5ae11b7d24f4fa39f2cace419ec86d9ee5f9492a6cc46  src/glm52_enforcement/spend_authority.py
220a964f298569718f10a7553cd11d5e8ad822e98be5977d140581aa7daef541  aws/glm52-gpu/scripts/inspect_h1d_live_authority.py
12416ff8e43815c202269f1200107308e0620fa41c20f3b36afe5d8cffa990f0  aws/glm52-gpu/scripts/materialize_h1d_trusted_sources.py
12b942de2141ca719ee1247ff935601d5aeec6538a1a0e46c04036f16918ef7b  tests/test_glm52_enforcement_live_authority.py
c13f744239110f21ccb586052e7d98cc87736f5f43ffa1736247e5baaa34b68b  tests/test_glm52_enforcement_spend_authority.py
75b47aab4e22953bc06bb9f8617ed42c577ef603537d60ffad895f2f22648c7f  tests/test_glm52_h1d_trusted_source_materializer.py
```

### Fix-round-4 limitations and integration boundary

- All AWS, S3, DynamoDB, EC2, and SDK boundaries remain injected in Task 8
  tests. No live AWS, reserve, spend, launch, deployment, billing, or
  campaign-running claim is made by this task.
- Task 9 still owns the live exact mTLS Sky/relay probe and reservation
  consumer. Task 11 still owns the full 58-second authority suffix and
  measured 600-second closure.

## Independent approval

Independent read-only re-review approved immutable commit `2021719e` with no
Critical or Important findings. The reviewer independently verified the real
materializer's empty and whitespace-only physical-resource mutants, exact
nonempty byte preservation, `20/20` materializer tests, `328/328` focused
Task 8 tests, `947/947` aggregate enforcement tests with the two accepted SWIG
warnings, and `26/26` frozen CloudFormation tests.

## Independent-review fix round 7

Status: `DONE`

The seventh review found three remaining production-response identity gaps:
EC2 accepted the root and first non-root EBS volume while ignoring additional
attachments; CloudWatch treated an `OK` alarm and `Complete` status as
sufficient even when metric evidence was empty or misaligned; and the Lambda,
IAM, and EC2 source adapters still compared CFN property/name shapes against
different AWS response/ARN shapes.

### Literal fix-round-7 RED

The EC2 and CloudWatch regressions first failed exactly three cases:

```text
3 failed, 252 deselected in 0.25s
```

The AWS-native semantic fixture then failed the three exact remaining identity
boundaries while four mutants/controls already behaved correctly:

```text
3 failed, 4 passed, 254 deselected in 0.18s
```

### Fixes

- EC2 now requires exactly two inventory block-device mappings and exactly two
  described volumes, a bijection between their IDs, exactly one attached
  attachment per volume for the inspected instance, and exactly one root plus
  one non-root data device. The canonical item binds the complete sorted
  root/data attachment contract, so a third mapping, volume, attachment, or
  device cannot be ignored.
- CloudWatch now requires exactly one metric-data result whose status is
  `Complete`, with nonempty, aligned timestamp/value arrays, unique timestamps,
  and finite numeric values. The canonical item records
  `ALIGNED_NONEMPTY`; alarm `OK` remains necessary but is not sufficient.
- Lambda configuration identity is no longer a hash of raw
  `GetFunction.Configuration`. The source authority authenticates the sorted
  code-owned field contract, maps CFN properties into the corresponding AWS
  response semantics, and hashes only that projection. Environment error
  detail, layer metadata, VPC ID, architecture, ephemeral-storage,
  package-type, SnapStart, logging, and other documented service-default noise
  are ignored unless they are source-owned. Source-owned Handler, Layers,
  MemorySize, Role, Runtime, Timeout, VPC, environment, tracing, KMS,
  description, and dead-letter semantics remain drift-sensitive.
- CFN `Fn::GetAtt` Lambda role references now resolve through the authenticated
  physical role name, path, account, and partition to the exact IAM role ARN.
  IAM role instance-profile membership and EC2 `IamInstanceProfile` references
  likewise derive full `arn:aws:iam::246813579024:instance-profile/...` values
  from CFN physical names instead of comparing raw names with AWS ARNs.
- Each source-owned `AWS::Lambda::Permission` is transformed into the
  semantically equivalent canonical AWS `GetPolicy` statement. The mapping
  binds the authenticated CFN physical permission ID as `Sid`, `Allow`,
  action, service/account principal, exact qualified function resource,
  SourceArn, and SourceAccount conditions. Live policy documents discard only
  the service top-level defaults and canonicalize every statement before
  hashing. Mutants cover Sid, action, principal, qualified resource,
  SourceArn, and SourceAccount.
- Lambda versions with no resource policy now accept only the authenticated
  AWS `ResourceNotFoundException` response as an empty policy. Both CLI and
  SDK adapters preserve the narrow absence classification; all other Lambda
  errors still fail closed.

### Fix-round-7 verification

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py
269 passed in 1.88s

$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
302 passed in 2.15s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
921 passed, 2 warnings in 7.24s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.62s

Python compile: PASS
offline Ruff E4/E7/E9/F: PASS
git diff --check: PASS
```

The two aggregate warnings remain the accepted SWIG deprecation warnings in
the existing source-authority parity test.

### Fix-round-7 SHA-256

```text
8eb4d83ca18ba2c3147e28975ff61ef635de4dfcb39da79c0430760869dff151  src/glm52_enforcement/live_authority.py
45bae66c683d0a0cf2e5ae11b7d24f4fa39f2cace419ec86d9ee5f9492a6cc46  src/glm52_enforcement/spend_authority.py
1c849d6226c6cdb85b25f06f32660235161c6e40da8b601112dc9fcfe87154b6  aws/glm52-gpu/scripts/inspect_h1d_live_authority.py
12416ff8e43815c202269f1200107308e0620fa41c20f3b36afe5d8cffa990f0  aws/glm52-gpu/scripts/materialize_h1d_trusted_sources.py
dfa94efe1fc3c9bfabb99418ec0172e1b22d446cf4df324906c77d4432633fa2  tests/test_glm52_enforcement_live_authority.py
c13f744239110f21ccb586052e7d98cc87736f5f43ffa1736247e5baaa34b68b  tests/test_glm52_enforcement_spend_authority.py
75b47aab4e22953bc06bb9f8617ed42c577ef603537d60ffad895f2f22648c7f  tests/test_glm52_h1d_trusted_source_materializer.py
```

### Fix-round-7 limitations and integration boundary

- All AWS, S3, DynamoDB, EC2, and SDK boundaries remain injected in Task 8
  tests. No live AWS, reserve, spend, launch, deployment, billing, or
  campaign-running claim is made by this task.
- Task 9 still owns the live exact mTLS Sky/relay probe and reservation
  consumer. Task 11 still owns the full 58-second authority suffix and
  measured 600-second closure.

## Independent-review fix round 6

Status: `DONE`

The sixth review found that the round-five expected-state derivation was
independent, but the live reader still used generic caller-shaped field paths.
Real AWS responses expose service-native names and nested response shapes, so
those projections could not produce the same canonical item contract. The
final review also found that a single support-plane prefix could hide
run-ID-only foreign EventBridge, Scheduler, SQS, CloudWatch, and Logs
resources.

### Literal fix-round-6 RED

The production-plan regression first proved that every authenticated family
still carried a schema-version-two generic projection:

```text
1 failed, 216 deselected in 0.10s
```

### Fixes

- Production expected specs now carry only the exact schema-version-three
  `<family>.aws_response_v1` normalizer identity. Generic schema-version-two
  projections remain available only to isolated debug-reader tests and are
  rejected by `AwsExpectedStateAuthority`.
- The production reader has explicit AWS-response normalizers for
  CloudFormation, Lambda, IAM, EventBridge, Scheduler, SQS, SNS, EC2, S3,
  DynamoDB, Systems Manager, CloudWatch, and Logs. Each normalizer consumes
  service-native inventory/detail shapes, parses JSON policy documents,
  canonicalizes pagination-complete collections, handles qualified Lambda
  versions, translates queue URLs to queue ARNs, and preserves only
  independently authenticated provenance fields that have no live AWS
  analogue.
- Source derivation and live normalization now share canonical semantic
  contracts for CloudFormation resource inventories, Lambda code/configuration,
  resolved network edges, EC2 user data, IAM policy collections, and S3
  expected absence. Lambda `CodeSha256` is decoded from AWS base64 to the
  source-bound hexadecimal digest. S3 `NoSuchBucketPolicy`,
  `NoSuchLifecycleConfiguration`, and
  `ReplicationConfigurationNotFoundError` are the only accepted absence
  classes.
- A full core-path regression now enters
  `inspect_h1d_live_authority`, authenticates the exact Task 6/7 source
  authority, runs the concurrent production SDK reader, traverses real
  inventory/detail/pagination code for every live family using realistic AWS
  response shapes, completes spend and Sky probe checks, and reaches a valid
  result. It does not intercept the request before the core.
- Twelve non-DynamoDB family normalizers each have a source-derived acceptance
  test and a family-specific live-drift mutant. DynamoDB retains its existing
  strongly consistent full-typed-item hash tests and exhaustive sealed-field
  mutants.
- EventBridge, Scheduler, SQS, CloudWatch, and Logs now union two server-side
  inventories: the code-owned support prefix and the exact run-ID-derived
  prefix. The union fully paginates each prefix, canonical-deduplicates
  identities, rejects inconsistent duplicates, and then applies the existing
  exact/campaign client scope. Five prefix-union tests and five run-ID-only
  foreign-resource tests prove those resources remain visible to exact
  inventory comparison.

### Fix-round-6 verification

```text
$ .venv/bin/pytest -q tests/test_glm52_enforcement_live_authority.py
252 passed in 1.76s

$ .venv/bin/pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
285 passed in 1.77s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
904 passed, 2 warnings in 7.08s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.55s

Python compile: PASS
offline Ruff E4/E7/E9/F: PASS
git diff --check: PASS
```

The two aggregate warnings remain the accepted SWIG deprecation warnings in
the existing source-authority parity test.

### Fix-round-6 SHA-256

```text
8eb4d83ca18ba2c3147e28975ff61ef635de4dfcb39da79c0430760869dff151  src/glm52_enforcement/live_authority.py
45bae66c683d0a0cf2e5ae11b7d24f4fa39f2cace419ec86d9ee5f9492a6cc46  src/glm52_enforcement/spend_authority.py
c241c46bef787f340dc27851c85e0c0e14eda3b150f71b5a8ea51dc0c28b3712  aws/glm52-gpu/scripts/inspect_h1d_live_authority.py
12416ff8e43815c202269f1200107308e0620fa41c20f3b36afe5d8cffa990f0  aws/glm52-gpu/scripts/materialize_h1d_trusted_sources.py
6d1002800647e7791ec7d975a6641d515d0948abe76aba86cbd6171797663238  tests/test_glm52_enforcement_live_authority.py
c13f744239110f21ccb586052e7d98cc87736f5f43ffa1736247e5baaa34b68b  tests/test_glm52_enforcement_spend_authority.py
75b47aab4e22953bc06bb9f8617ed42c577ef603537d60ffad895f2f22648c7f  tests/test_glm52_h1d_trusted_source_materializer.py
```

### Fix-round-6 limitations and integration boundary

- All AWS, S3, DynamoDB, EC2, and SDK boundaries remain injected in Task 8
  tests. No live AWS, reserve, spend, launch, deployment, billing, or
  campaign-running claim is made by this task.
- Task 9 still owns the live exact mTLS Sky/relay probe and reservation
  consumer. Task 11 still owns the full 58-second authority suffix and
  measured 600-second closure.

## Independent-review fix round 5

Status: `DONE`

The fifth review found that the exact source boundary still authenticated only
selected expected-state fields. Caller-rehashed IAM managed-policy versions,
Lambda configuration, CloudFormation parameters, SQS redrive policy, S3
policy, and Logs retention could therefore survive the authority check. It
also found caller-controlled deadline and credential-expiration claims,
non-quiescent timeout cancellation, and account-global inventory reads for
several services.

### Literal fix-round-5 RED

The six named semantic substitutions were first demonstrated as six accepted
mutants:

```text
6 failed, 196 deselected in 0.20s
```

The complete expected-item matrix then exercised all 89 fields across every
item-bearing live family:

```text
65 failed, 24 passed, 113 deselected in 1.39s
```

Fourteen additional regressions covered two caller-extended deadline shapes,
authenticated SDK credential expiration, quiescent cancellation, seven
account-global inventory scopes, and three service-side campaign filters:

```text
14 failed, 202 deselected in 0.42s
```

### Fixes

- `AwsExpectedStateAuthority` now independently derives the complete expected
  item tuple for every family from the exact Task 6 manifest/templates, exact
  Task 7 support template/postcreate inventory, fixed campaign rules, and the
  sealed activation-scoped DynamoDB inventory. It compares every field and
  item, then compares the complete operation, identity field, parameters, and
  code-owned projection plan. The caller-rehashable
  `h1d_specs_identity_sha256` field remains non-authoritative.
- The expected-state authority enforces the handoff's fresh twelve-hour
  `must_start_by` window and a fixed maximum twenty-four-hour execution
  deadline. A caller cannot extend either to manufacture continuing
  authority.
- The live identity adapter no longer accepts credential expiration from the
  serialized input envelope. It obtains the fixed expiration from the
  authenticated boto session credential provider used for the STS read, and
  static or otherwise non-expiring credentials fail closed.
- The SDK reader now shares a monotonic cooperative cancellation guard across
  all family workers and every SDK call/page/detail boundary. Timeout or first
  failure sets cancellation, cancels queued futures, and waits for all
  in-flight reads to quiesce before returning, so no post-return AWS calls can
  occur.
- CloudFormation, Lambda, IAM, EventBridge, Scheduler, SQS, and S3 inventories
  retain exact expected identities plus all `keep-glm52-h1g`/run-namespaced
  foreign resources while ignoring unrelated account resources. EventBridge,
  Scheduler, and SQS also use supported name-prefix arguments. Systems
  Manager uses the exact campaign-run tag filter; CloudWatch and Logs use
  fixed campaign prefixes. EC2 retains the prior three-query campaign/P5/name
  union and SNS retains the exact topic scope.
- The in-process SDK argument allowlist now carries the new prefix filters and
  translates the Systems Manager tag filter to the service's typed `Key` /
  `Values` request shape.

### Fix-round-5 verification

```text
$ .venv/bin/pytest -q tests/test_glm52_enforcement_live_authority.py
216 passed in 1.98s

$ .venv/bin/pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
249 passed in 1.96s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
868 passed, 2 warnings in 6.73s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.51s

Python compile: PASS
offline Ruff E4/E7/E9/F: PASS
git diff --check: PASS
```

The two aggregate warnings remain the accepted SWIG deprecation warnings in
the existing source-authority parity test.

### Fix-round-5 SHA-256

```text
8eb4d83ca18ba2c3147e28975ff61ef635de4dfcb39da79c0430760869dff151  src/glm52_enforcement/live_authority.py
45bae66c683d0a0cf2e5ae11b7d24f4fa39f2cace419ec86d9ee5f9492a6cc46  src/glm52_enforcement/spend_authority.py
7cfb87badd4fdbf8e2972b1cb8647587703db95c96c4d081a15a166adff9e74f  aws/glm52-gpu/scripts/inspect_h1d_live_authority.py
12416ff8e43815c202269f1200107308e0620fa41c20f3b36afe5d8cffa990f0  aws/glm52-gpu/scripts/materialize_h1d_trusted_sources.py
75fd908743f72117cf0fca45ff2d115bd564ba17f9ab1b784f95f4efe5589ef7  tests/test_glm52_enforcement_live_authority.py
c13f744239110f21ccb586052e7d98cc87736f5f43ffa1736247e5baaa34b68b  tests/test_glm52_enforcement_spend_authority.py
75b47aab4e22953bc06bb9f8617ed42c577ef603537d60ffad895f2f22648c7f  tests/test_glm52_h1d_trusted_source_materializer.py
```

### Fix-round-5 limitations and integration boundary

- All AWS, S3, DynamoDB, EC2, and SDK boundaries remain injected in Task 8
  tests. No live AWS, reserve, spend, launch, deployment, billing, or
  campaign-running claim is made by this task.
- Task 9 still owns the live exact mTLS Sky/relay probe and reservation
  consumer. Task 11 still owns the full 58-second authority suffix and
  measured 600-second closure.

## Independent-review fix round 8

Status: `DONE`

The eighth review found four final source-to-live identity gaps. The
CloudFormation contract still derived physical resources and deployment-role
identity from Task 7 rows or constructed names instead of independently
sealing all three deployed Task 6 stacks. Scheduler targets did not resolve an
IAM `Fn::GetAtt Role.Arn` into the exact account role ARN. IAM roles did not
bind their trust policy, path, permissions boundary, or tags. Lambda did not
bind the complete event-source mapping inventory.

### Literal fix-round-8 RED

The three semantic adapter regressions and their controls first produced:

```text
4 failed, 2 passed, 266 deselected in 0.21s
```

The three-stack materializer regressions first failed before the
CloudFormation reader existed:

```text
3 failed, 5 deselected in 0.04s
```

The production-native materializer boundary then had its own explicit RED:

```text
1 failed, 8 deselected in 0.05s
```

### Fixes

- The trusted-source materializer's production default is now an in-process
  AWS SDK runner with an exact operation/option allowlist, authenticated
  service RequestIds, fixed one- and two-second connection/read timeouts, and
  `total_max_attempts: 1`. The previous one-attempt CLI runner remains only as
  an injectable debug boundary.
- Before creating the `SEALED` record, the materializer reads the exact
  retained, fence, and support stack IDs from the Task 6 manifest. For every
  stack it performs exact `DescribeStacks`, `GetTemplate`, fully paginated
  `ListStackResources`, and `DescribeTerminationProtection` calls. It requires
  the exact stack ID/name, a complete status, an account-local IAM
  `RoleARN`, exact source-template semantics, a bijection over source logical
  IDs and resource types, nonempty physical IDs, and enabled termination
  protection. All three stacks must use the same observed deployment role.
- The `SEALED` trusted-source contract now carries exactly three sorted
  CloudFormation expected items with the real StackIds, status, actual
  deployment role ARN, source-equal template hash, parameter hash, canonical
  full physical-resource inventory hash, termination protection, and exact
  Task 6 manifest byte identity. The trusted reader validates that closed
  schema. Expected-state derivation accepts only those three sealed physical
  inventories and requires their IDs and manifest identity to match the exact
  Task 6 source.
- Scheduler source derivation resolves an IAM
  `Fn::GetAtt [Role, Arn]` through the authenticated CFN role resource, path,
  account, and partition. Source and live target hashes therefore carry the
  same exact account role ARN; raw intrinsic or role-name substitutions drift.
- IAM role identity now includes the canonical trust-policy hash, exact path,
  optional permissions-boundary ARN, and sorted exact role tags from the
  source template and native `GetRole` response. A rehashed foreign principal
  in `AssumeRolePolicyDocument` is rejected.
- Lambda expected state now derives every source-owned
  `AWS::Lambda::EventSourceMapping` associated with each published function
  identity. The authenticated contract binds the exact UUID set, sorted
  source-owned field contract, full canonical mapping hash, and default
  enabled semantics. Live normalization consumes the complete, fully
  paginated native mapping inventory; an extra foreign enabled mapping,
  missing mapping, UUID substitution, or source-field drift is rejected.
- Realistic production-path fixtures now contain the exact three-stack
  inventory, native Scheduler role ARN, complete IAM role response, and native
  Lambda event-source mapping. Dedicated mutants cover foreign/mismatched
  stack roles, missing stack resources, foreign IAM trust, Scheduler intrinsic
  leakage, and unexpected Lambda mappings.

### Fix-round-8 verification

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py
272 passed in 1.93s

$ .venv/bin/python -m pytest -q \
    tests/test_glm52_h1d_trusted_source_materializer.py
9 passed in 0.03s

$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
309 passed in 2.03s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
928 passed, 2 warnings in 6.91s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.56s

Python compile: PASS
offline Ruff E4/E7/E9/F: PASS
both production scripts executable mode: PASS
Git commands: NOT RUN (Task 8 no-Git fence)
```

The two aggregate warnings remain the accepted SWIG deprecation warnings in
the existing source-authority parity test.

### Fix-round-8 SHA-256

```text
8eb4d83ca18ba2c3147e28975ff61ef635de4dfcb39da79c0430760869dff151  src/glm52_enforcement/live_authority.py
45bae66c683d0a0cf2e5ae11b7d24f4fa39f2cace419ec86d9ee5f9492a6cc46  src/glm52_enforcement/spend_authority.py
69b3d6ef0ec212f47319b126bf9dcefa32a7c2ecf42fcf8e1f8d6cc8b4f23823  aws/glm52-gpu/scripts/inspect_h1d_live_authority.py
71a2319fd1cf20a2ecd97abcf7b71b631ed53899e4902a3841edd0d44bede8bb  aws/glm52-gpu/scripts/materialize_h1d_trusted_sources.py
418c079c560eadd82cd4dc72b2fd05bbdf2ae5b6d5b3e8111d4eaf5e51fcefa5  tests/test_glm52_enforcement_live_authority.py
c13f744239110f21ccb586052e7d98cc87736f5f43ffa1736247e5baaa34b68b  tests/test_glm52_enforcement_spend_authority.py
896a04c814247f4f3d6d758e7b8c28d693b256fb6b75f221b66284796dfbe56d  tests/test_glm52_h1d_trusted_source_materializer.py
```

### Fix-round-8 limitations and integration boundary

- All AWS, S3, DynamoDB, CloudFormation, and SDK responses remain injected in
  Task 8 tests. No live AWS, reserve, spend, launch, deployment, billing, or
  campaign-running claim is made by this task.
- Task 9 still owns the live exact mTLS Sky/relay probe and reservation
  consumer. Task 11 still owns the full 58-second authority suffix and
  measured 600-second closure.

## Independent-review fix round 9

Status: `DONE`

The ninth review found that the three-stack seal still trusted a shared
same-account `RoleARN` without proving its stable IAM `RoleId`, accepted any
stack status ending in `_COMPLETE`, did not compare deployed stack tags with
the authenticated Task 6 tag sets, and omitted each resource's
`ResourceStatus` from the canonical inventory identity.

### Literal fix-round-9 RED

The materializer RoleId, exact stack-state, tag, and resource-state
regressions first produced:

```text
12 failed, 6 deselected in 0.10s
```

The downstream trusted-reader and live-normalizer regressions then proved
that the new sealed role contract was not accepted and that tag/resource
status mutants were still ignored:

```text
3 failed, 271 deselected in 0.20s
```

### Fixes

- The production-native materializer allowlist now includes only the exact
  IAM `GetRole` operation and `RoleName` option required for this proof. After
  all three stack reads agree on one observed `RoleARN`, the materializer
  performs exactly one native `GetRole`. The SDK boundary authenticates its
  service `RequestId`, and the semantic layer requires response ARN,
  RoleName, path-derived ARN, and RoleId to be mutually consistent.
- The observed IAM `RoleId` must exactly equal the authenticated Task 6
  `retained_deployment_role_id`. The `SEALED` record carries the closed
  `cloudformation_deployment_role` object containing the actual RoleARN,
  stable RoleId, and IAM service RequestId. The trusted reader validates that
  object, requires every sealed stack to use its exact ARN, and returns it as
  part of the authenticated campaign-source contract.
- `AwsExpectedStateAuthority` independently revalidates the sealed RoleId
  against the exact Task 6 bytes, the account-local ARN and nonempty request
  identity, and the exact same RoleARN on all three expected stack items.
  Mutants replacing the shared ARN and RoleId with a same-account attacker
  role, or replacing only the observed ARN against authenticated `GetRole`
  truth, fail closed.
- Every retained, fence, and support stack must now be exactly
  `UPDATE_COMPLETE`; `UPDATE_ROLLBACK_COMPLETE` is not accepted merely because
  its spelling ends in `_COMPLETE`.
- Every deployed stack tag is canonicalized into a sorted unique key/value
  list and must exactly equal that stack's authenticated Task 6 tag mapping.
  Missing, extra, duplicated, malformed, or foreign tags fail before sealing.
  The canonical stack tags are persisted in every sealed expected item,
  source-revalidated downstream, and compared against each fresh native
  `DescribeStacks` response.
- Every listed CloudFormation resource must have one of the accepted complete
  current-resource states: `CREATE_COMPLETE`, `IMPORT_COMPLETE`, or
  `UPDATE_COMPLETE`. Missing, failed, rollback, or in-progress states fail
  before sealing. `resource_status` is now part of every canonical resource
  row and therefore part of `resources_sha256`; fresh live normalization uses
  the same full rows, so later status drift cannot be ignored.
- Tests cover the positive three-stack/one-IAM-call contract, native SDK
  `GetRole` option translation, split roles, shared attacker role and RoleId,
  observed-ARN substitution against IAM truth, missing resources,
  `UPDATE_ROLLBACK_COMPLETE`, foreign/missing tags, and failed, rollback,
  in-progress, or missing resource statuses. Downstream mutants cover a
  jointly substituted attacker ARN/RoleId, non-exact status, foreign tags,
  and fresh live tag/resource-status drift.

### Fix-round-9 verification

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py
277 passed in 2.13s

$ .venv/bin/python -m pytest -q \
    tests/test_glm52_h1d_trusted_source_materializer.py
18 passed in 0.05s

$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
323 passed in 2.05s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
942 passed, 2 warnings in 7.41s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.59s

Python compile: PASS
offline Ruff E4/E7/E9/F: PASS
both production scripts executable mode: PASS
Git commands: NOT RUN (Task 8 no-Git fence)
```

The two aggregate warnings remain the accepted SWIG deprecation warnings in
the existing source-authority parity test.

### Fix-round-9 SHA-256

```text
8eb4d83ca18ba2c3147e28975ff61ef635de4dfcb39da79c0430760869dff151  src/glm52_enforcement/live_authority.py
45bae66c683d0a0cf2e5ae11b7d24f4fa39f2cace419ec86d9ee5f9492a6cc46  src/glm52_enforcement/spend_authority.py
dce32d6f6d09ad2f91fb169367a570341569daeac62167d0f19332343575d751  aws/glm52-gpu/scripts/inspect_h1d_live_authority.py
81f98dd1a0877ef2815dca798422597e68832cb32f897212e30f21b16fc1c84f  aws/glm52-gpu/scripts/materialize_h1d_trusted_sources.py
344355a0227250781d94f972e3e6f69354a9aeb995e5a4ab29d91462aa3c4edc  tests/test_glm52_enforcement_live_authority.py
c13f744239110f21ccb586052e7d98cc87736f5f43ffa1736247e5baaa34b68b  tests/test_glm52_enforcement_spend_authority.py
514246e7a8747664440e20a910882db0a9e6355c0355f74f988b125d37b3c43d  tests/test_glm52_h1d_trusted_source_materializer.py
```

### Fix-round-9 limitations and integration boundary

- All AWS, S3, DynamoDB, CloudFormation, IAM, and SDK responses remain
  injected in Task 8 tests. No live AWS, reserve, spend, launch, deployment,
  billing, or campaign-running claim is made by this task.
- Task 9 still owns the live exact mTLS Sky/relay probe and reservation
  consumer. Task 11 still owns the full 58-second authority suffix and
  measured 600-second closure.

## Independent-review fix round 10

Status: `DONE`

The tenth review found an identity-lineage collision in the core result.
`inspect_h1d_live_authority` validated an `ExpectedStateAuthentication` and
then discarded it. The final `H1dLiveAuthorityResult` bound only the caller's
expected-spec identity, so two successful inspections with identical expected
items but different exact Task 6/7 bytes or direct service RequestId lineages
could produce the same final canonical identity.

### Literal fix-round-10 RED

The closed result, lineage-collision, deterministic-control, and malformed
identity regressions first produced:

```text
4 failed, 276 deselected in 0.21s
```

The failures showed that the result had no
`expected_state_authentication_identity_sha256` field, distinct authenticated
lineages collided, and a missing or substituted lineage identity could not be
represented and validated by the frozen result contract.

### Fixes

- The core now preserves the exact validated
  `ExpectedStateAuthentication` returned by the expected-state authority
  instead of discarding the validator's return value.
- `H1dLiveAuthorityResult` has the exact additive field
  `expected_state_authentication_identity_sha256`. It is populated only from
  the validated authentication object's canonical identity, never from the
  caller's request or expected-state envelope.
- The field participates in `_result_body` and therefore in the final H.1d
  canonical identity. That authentication identity already binds the exact
  expected-state identity, Task 6 manifest bytes, Task 6 template-bundle
  bytes, Task 7 postcreate manifest identity, Task 7 inventory identity, the
  complete unique direct-read RequestId tuple, and observation time.
- Result-body validation requires an exact SHA-256 for the new field before
  canonical comparison. Missing/malformed identity values fail directly; a
  valid-shaped substituted digest without the matching result rehash fails
  the existing canonical identity check.
- The regression executes the real core twice with one identical
  authentication lineage and once with a different Task 6/direct-RequestId
  lineage while keeping expected specs, live pages, caller, spend, probe, and
  phase times identical. The same lineage is byte-deterministic; the distinct
  lineage changes both the new field and the final H.1d identity.

### Fix-round-10 verification

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py
280 passed in 2.06s

$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
326 passed in 2.06s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
945 passed, 2 warnings in 7.27s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.56s

Python compile: PASS
offline Ruff E4/E7/E9/F: PASS
both production scripts executable mode: PASS
Git commands: NOT RUN (Task 8 no-Git fence)
```

The two aggregate warnings remain the accepted SWIG deprecation warnings in
the existing source-authority parity test.

### Fix-round-10 SHA-256

```text
9a1f4bc2f24cb9f8c244761901585f9c3e1604f26864641eb12ed488a92fe3dd  src/glm52_enforcement/live_authority.py
45bae66c683d0a0cf2e5ae11b7d24f4fa39f2cace419ec86d9ee5f9492a6cc46  src/glm52_enforcement/spend_authority.py
dce32d6f6d09ad2f91fb169367a570341569daeac62167d0f19332343575d751  aws/glm52-gpu/scripts/inspect_h1d_live_authority.py
81f98dd1a0877ef2815dca798422597e68832cb32f897212e30f21b16fc1c84f  aws/glm52-gpu/scripts/materialize_h1d_trusted_sources.py
11105ae77670050d9e7c9a3e2f41e0b84259e281a69e703b8697b5b1e384865c  tests/test_glm52_enforcement_live_authority.py
c13f744239110f21ccb586052e7d98cc87736f5f43ffa1736247e5baaa34b68b  tests/test_glm52_enforcement_spend_authority.py
514246e7a8747664440e20a910882db0a9e6355c0355f74f988b125d37b3c43d  tests/test_glm52_h1d_trusted_source_materializer.py
```

### Fix-round-10 limitations and integration boundary

- All AWS, S3, DynamoDB, CloudFormation, IAM, SDK, and direct RequestId
  lineages remain injected in Task 8 tests. No live AWS, reserve, spend,
  launch, deployment, billing, or campaign-running claim is made by this
  task.
- Task 9 still owns the live exact mTLS Sky/relay probe and reservation
  consumer. Task 11 still owns the full 58-second authority suffix and
  measured 600-second closure.

## Independent-review fix round 11

Status: `DONE`

The eleventh review found one narrow materializer gap: a complete
CloudFormation resource with an empty or whitespace-only
`PhysicalResourceId` passed the type check and entered the canonical sealed
resource hash.

### Literal fix-round-11 RED

Named empty-string and whitespace-only mutants using the real three-stack
materializer helper first produced:

```text
2 failed, 18 deselected in 0.05s
```

Both failures were the intended `DID NOT RAISE` result, proving that the
materializer accepted and sealed the invalid physical identities.

### Fix

- Every complete listed CloudFormation resource must now provide a string
  `PhysicalResourceId` with at least one non-whitespace character. Empty and
  whitespace-only identities fail before row normalization, hashing, IAM role
  authentication, or `SEALED` record creation.
- The value is validated with `strip()` but is not normalized or rewritten.
  The original exact nonempty physical identity remains in the canonical row
  and therefore in `resources_sha256`.
- The positive real-helper assertion independently recomputes the retained
  stack's resource hash using its exact `retained-physical` value and complete
  resource status. The two named mutants prove both empty forms reject.

### Fix-round-11 verification

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_h1d_trusted_source_materializer.py
20 passed in 0.25s

$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_enforcement_spend_authority.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
328 passed in 2.08s

$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py \
    tests/test_glm52_h1d_trusted_source_materializer.py
947 passed, 2 warnings in 7.34s

$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.59s

Python compile: PASS
offline Ruff E4/E7/E9/F: PASS
both production scripts executable mode: PASS
Git commands: NOT RUN (Task 8 no-Git fence)
```

The two aggregate warnings remain the accepted SWIG deprecation warnings in
the existing source-authority parity test.

### Fix-round-11 SHA-256

```text
9a1f4bc2f24cb9f8c244761901585f9c3e1604f26864641eb12ed488a92fe3dd  src/glm52_enforcement/live_authority.py
45bae66c683d0a0cf2e5ae11b7d24f4fa39f2cace419ec86d9ee5f9492a6cc46  src/glm52_enforcement/spend_authority.py
dce32d6f6d09ad2f91fb169367a570341569daeac62167d0f19332343575d751  aws/glm52-gpu/scripts/inspect_h1d_live_authority.py
843106f2921af44c87167c82d444ce7ef0bbcca6a6191c71c0504901b0c6d846  aws/glm52-gpu/scripts/materialize_h1d_trusted_sources.py
11105ae77670050d9e7c9a3e2f41e0b84259e281a69e703b8697b5b1e384865c  tests/test_glm52_enforcement_live_authority.py
c13f744239110f21ccb586052e7d98cc87736f5f43ffa1736247e5baaa34b68b  tests/test_glm52_enforcement_spend_authority.py
55956abedf74796259d5e1cdf71cdd095116f7b5210368b9eb38d178e4002431  tests/test_glm52_h1d_trusted_source_materializer.py
```

### Fix-round-11 limitations and integration boundary

- All CloudFormation and SDK responses remain injected in Task 8 tests. No
  live AWS, reserve, spend, launch, deployment, billing, or campaign-running
  claim is made by this task.
- Task 9 still owns the live exact mTLS Sky/relay probe and reservation
  consumer. Task 11 still owns the full 58-second authority suffix and
  measured 600-second closure.
