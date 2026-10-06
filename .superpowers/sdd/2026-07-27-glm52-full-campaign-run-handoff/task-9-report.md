# Task 9 Report — Sky admission, launch custody, and retained liability

Date: 2026-07-29

## Status

`DONE`

Task 9 implements the pinned Sky identity and loopback admission contracts,
repository-owned intent-only Sky provisioner route, durable prepared WAL,
deterministic launch parameters and ClientToken, the sole bounded same-token
sender, retained liability watcher and exact-instance termination custody,
post-terminal final view, atomic settlement coordinator, and an executable
production composition over the closed Task 3 DynamoDB adapter.

The fix round necessarily extended the Task 3 adapter with six narrow,
record-family-specific operations; no generic transaction escape hatch was
introduced. No Git mutation, deployment, Sky POST, launch, termination,
reserve/spend write, billing effect, or installed-Sky mutation occurred.

## Owned files

- `src/glm52_enforcement/sky_admission.py`
- `src/glm52_enforcement/launch_custody.py`
- `src/glm52_enforcement/launch_wal.py`
- `src/glm52_enforcement/task9_contract.py`
- `src/glm52_enforcement/dynamodb.py`
- `aws/glm52-gpu/skypilot/worker_launch_intent_patch.py`
- `aws/glm52-gpu/skypilot/skypilot-0.13.0-lock.txt`
- `aws/glm52-gpu/skypilot/server-config-v1.json`
- `aws/glm52-gpu/lambda/worker_launch_custody_handler.py`
- `aws/glm52-gpu/scripts/build_h1g_task9_contract.py`
- `aws/glm52-gpu/scripts/inspect_h1d_live_authority.py`
- `aws/glm52-gpu/cfn/h1g/task9-contract-v1.json`
- `tests/test_glm52_enforcement_sky_admission.py`
- `tests/test_glm52_enforcement_launch_custody.py`
- `tests/test_glm52_enforcement_dynamodb.py`
- `tests/test_glm52_enforcement_live_authority.py`
- `tests/test_glm52_task9_adapters.py`
- `tests/test_glm52_task9_contract.py`
- `.superpowers/sdd/2026-07-27-glm52-full-campaign-run-handoff/task-9-report.md`

## Implemented contracts

### Pinned Sky and admission

- The pinned identity binds SkyPilot `0.13.0`, installed wheel METADATA and
  RECORD, interpreter path and bytes, exact dependency lock, loopback server
  configuration, original and repository-patched provisioner source, jobs
  server source, and `jobs.launch retryable=False`.
- The Sky server is loopback-only at `127.0.0.1:46580`, uses one authenticated
  least-privileged `user` service account, and accepts the combined host as the
  sole effective controller with zero separate controllers and no inert
  controller resource pin.
- Attestation, launch admission, numeric binding, and retained cancellation
  have distinct ports, principals, client/server certificate identities, and
  mutually unusable paths. They expose no IMDS, shell, credentials, raw route,
  token selection, or operator command.
- Attestation is closed to its freshness nonce. Admission re-reads Task 8
  authority, current owner, reserve, and action state, performs at most one
  relay POST, and treats caller-asserted probe success as non-authoritative.
  Accepted, rejected, timeout, connection-loss, replay, and process-death
  paths have distinct fail-closed outcomes.

### Prepared launch and same-token completion

- The immutable launch shape is exactly one on-demand `p5.48xlarge` in
  `us-west-2` with approved AMI/subnet/security group/profile/tags, IMDSv2,
  and one encrypted delete-on-termination 300-GiB gp3 root at 3,000 IOPS and
  125 MiB/s. Spot, Capacity Block, launch template, alternate shape, and data
  disk are rejected.
- The deterministic 64-character lowercase ClientToken binds the immutable
  campaign, activation, generation, action, Sky request/job, allocation
  ordinal, and exact launch-parameter identity.
- The repository-owned Sky patch authenticates the original installed source
  before replacing the entrypoint. Its production route creates only a durable
  intent and contains no reachable EC2 create/run, nested retry,
  mount/upload, subprocess, start, terminate, or PassRole route.
- The hash-chained SQLite WAL uses WAL mode, `synchronous=FULL`, database and
  directory fsync, exact duplicate read-only adoption, and conflicting
  operation rejection. Prepared WAL and DynamoDB readback occur before the
  Task 8 reserve and atomic `POSSIBLY_SENT`/liability transition.
- `Task8GpuReserveAdapter` calls the existing Task 8 authenticated reserve
  function and does not reconstruct or widen spend authority.
- The same-token completer accepts only activation plus allocation ordinal,
  exact-reads token and parameter bytes, consumes the liability action/shared
  counter before every call including the first, makes at most six serialized
  calls within 360 seconds, and requires botocore total attempts one.
  Positive rejection and ambiguous transport have separate ledger outcomes.

### Retained watcher, termination, and settlement

- The watcher uses the ledger owner/takeover boundary, closed parameterless
  schedule plus non-authoritative event accelerators, absolute 60-second
  cadence, and a checked 20-second scan/action deadline.
- Correlation binds exact ClientToken, exact expected tag identity, CloudTrail,
  EC2 state, launch, and spend evidence. Zero-instance scans do not settle.
  Late/multiple instances are allocation-open and drain-only unless work is
  independently authorized.
- Every termination requires an exact typed, action-recorded, same-instance
  permit. Up to six calls in a nominal 360-second window are accepted with no
  hidden SDK retry. The 30-day incident state preserves scanning and
  termination-only continuation.
- Post-terminal allocation creation uses the frozen Task 2 schema and binds
  the real campaign identity. Settlement re-reads the complete view, sorts
  post-terminal and merged arrays by ordinal/instance, requires closed
  terminal/spend evidence, supports both exact settlement kinds, and uses one
  no-retry transaction with exact ambiguous-response adoption.
- Nonsettled liability blocks replacement activation/token/allocation.
  Reserve release/refund is unavailable before authenticated settlement.
  The approved 900 seconds, `$13.76`, and separate `$0.01` root tail are
  reserves; the canonical contract explicitly records that AWS has no hard
  post-acceptance billing cap.

### Production adapters and generated contract

- STS/account/region is checked before each EC2 effect. SDK configuration must
  have `total_max_attempts=1`. The closed operation allowlist is
  `sts:GetCallerIdentity`, `ec2:RunInstances`, and
  `ec2:TerminateInstances`.
- A qualified published Lambda ARN and exact resource-policy identity are
  authenticated before the closed dispatcher calls the completion or watcher
  core. Caller-supplied token, parameters, instance, or alternate operation are
  not accepted.
- The published function must be an exact numeric, immutable Lambda version
  ARN; unqualified functions, aliases, `$LATEST`, extra qualifiers, and
  non-lowercase/non-hex policy identities are rejected before dispatch.
- The default H.1d production probe binds the live response to the code-owned
  Task 9 contract: the attestation relay server certificate, service-account
  user and sole role, effective-controller consolidation identity, and complete
  Sky identity-contract SHA must all match. A foreign response remains rejected
  even when it carries the right fresh nonce and recomputes its own self-hash.
- `Task3LaunchCustodyStore` implements every store operation used by the
  intent-only provisioner, same-token completer, retained watcher, and
  settlement coordinator. Reads use the real consistent/coherent/query Task 3
  boundaries; mutations use named, exact Task 3 methods and accept only exact
  live-owner commits, with settlement additionally permitting the frozen
  exact durable-adoption path.
- `build_production_task9_dispatcher` creates one-attempt STS, EC2, and
  DynamoDB clients, the real `DynamoLedgerAdapter`, one shared production
  store, the same-token sender, exact-instance terminator, retained watcher,
  settlement coordinator, and published-function guard. No SDK work occurs at
  import time.
- Task 3 now exposes only the missing closed contracts:
  exhaustive activation-family query, committed-WAL binding, liability
  evidence revision, three-record completion result, liability incident, and
  worker reconciliation. Each revalidates frozen records, transitions,
  activation binding, raw-nonce ownership, one-shot transaction bytes, and
  exact readback.
- Every one of those six additions has an operation-specific surface. The five
  writes reject every unrelated field in their affected record schemas, reject
  a foreign owner before the SDK, never adopt a conditional failure, and accept
  a lost-response readback only for the exact live owner. The query requires
  exact `ACTIVATION#<nonempty>#<family>#` grammar and an exact five-family
  record-type cross-product.
- The provisioner passes the authenticated raw owner nonce into prepared,
  committed-WAL, and possibly-sent store writes. Production writes therefore
  prove durable nonce ownership instead of relying on a caller-supplied hash.
- RunInstances distinguishes positive 4xx service rejection from ambiguous
  transport/5xx. Termination exact-reads the authorized instance before one
  call. The launch relay fixes DNS, port, path, mTLS peer identity, body size,
  and timeout.
- Exception-shaped responses are positive rejection only when their HTTP
  status is exactly in `400..499` and `Error.Code` is a nonempty exact string.
  Exception-shaped 2xx/3xx responses, 5xx responses, missing codes, non-string
  codes, and empty codes remain ambiguous and cannot disable same-token
  completion.
- IAM-expressible launch shape/tag controls are distinct from ClientToken
  enforcement. No nonexistent `ec2:ClientToken` condition key is invented;
  token custody is bound to the published function and ledger.
- `task9-contract-v1.json` is canonical, self-hashed, byte-regenerable, and
  binds the installed Sky identities, repository patch/lock/config hashes,
  RBAC/relay decomposition, effect boundary, launch shape, deadlines, retry
  counts, and residual no-hard-cap semantics.

## Literal RED to GREEN

Initial focused collection before the Task 9 modules existed:

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_sky_admission.py \
    tests/test_glm52_enforcement_launch_custody.py
ERROR tests/test_glm52_enforcement_sky_admission.py
ModuleNotFoundError: No module named 'glm52_enforcement.sky_admission'
ERROR tests/test_glm52_enforcement_launch_custody.py
ModuleNotFoundError: No module named 'glm52_enforcement.launch_custody'
2 errors during collection
```

The first core implementation checkpoint was `59 passed`. Expanding to the
prepared-WAL crash matrix, production effect adapters, and Task 2/3
predecessors exposed a real dataclass binding regression:

```text
6 failed, 327 passed in 0.67s
```

All six failures were the same missing prior-settlement field on the durable
worker intent. The corrected schema then produced:

```text
333 passed in 0.77s
```

Subsequent named REDs were observed before their implementations:

```text
ImportError: cannot import name 'TerminationPermit'
ModuleNotFoundError: No module named 'glm52_enforcement.task9_contract'
AttributeError: module ... has no attribute 'PublishedFunctionIdentityGuard'
AttributeError: module ... has no attribute 'Task9EffectDispatcher'
Failed: DID NOT RAISE LaunchCustodyError
```

Those REDs respectively drove typed termination-call custody, generated
contract identity, published-version/resource-policy enforcement, the closed
production dispatcher, and the real 20-second deadline check.

Independent review then identified the missing Task 3-backed production store.
The remediation was also literal RED to GREEN:

```text
$ .venv/bin/python -m pytest -q tests/test_glm52_enforcement_dynamodb.py \
    -k 'committed_wal_binding or liability_evidence_revision or activation_family_query'
3 failed, 184 deselected

$ .venv/bin/python -m pytest -q tests/test_glm52_enforcement_dynamodb.py \
    -k 'committed_wal_binding or liability_evidence_revision or activation_family_query'
3 passed, 184 deselected

$ .venv/bin/python -m pytest -q tests/test_glm52_enforcement_dynamodb.py \
    -k 'completion_result_is_one or liability_incident_is_a_closed or worker_reconciliation_is_a_closed'
3 failed, 187 deselected

$ .venv/bin/python -m pytest -q tests/test_glm52_enforcement_dynamodb.py \
    -k 'completion_result_is_one or liability_incident_is_a_closed or worker_reconciliation_is_a_closed'
3 passed, 187 deselected

$ .venv/bin/python -m pytest -q tests/test_glm52_task9_adapters.py \
    -k 'production_store or production_dispatcher_constructor'
2 failed, 10 deselected

$ .venv/bin/python -m pytest -q tests/test_glm52_task9_adapters.py \
    -k 'production_store or production_dispatcher_constructor'
2 passed, 10 deselected
```

The second independent review exposed operation-surface and live-identity
escapes. The exact round-two REDs were:

```text
$ .venv/bin/python -m pytest -q tests/test_glm52_enforcement_dynamodb.py \
    -k 'liability_scan_evidence_rejects_sensitive or activation_family_query_rejects_nonexact'
11 failed, 5 passed, 190 deselected

$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_task9_adapters.py \
    -k 'published_function_guard_rejects_nonversion_or_nonsha_identity or production_task9_probe_rejects_self_rehashed_foreign_identity'
10 failed, 1 passed, 293 deselected

$ .venv/bin/python -m pytest -q tests/test_glm52_enforcement_dynamodb.py \
    -k 'completion_rejects_unrelated or incident_rejects_unrelated or reconciliation_rejects_unrelated'
3 failed, 206 deselected
```

The fixes then produced:

```text
18 passed, 188 deselected
7 passed, 297 deselected
6 passed, 203 deselected
27 passed, 209 deselected
236 passed
```

The 27-test slice is the exhaustive six-API matrix: five exact write surfaces
cover all unrelated fields plus foreign owner, conditional failure, and
lost-response readback; activation-family query covers every valid and foreign
pair in the five-family cross-product.

The third independent review found that exception-shaped 2xx/3xx responses
were still being treated as positive EC2 rejection. The exact classifier and
real-completer matrix produced:

```text
$ .venv/bin/python -m pytest -q tests/test_glm52_task9_adapters.py \
    -k 'ec2_exception_status_code_matrix_is_exact_nonempty_4xx_only or same_token_completer_disables_only_for_real_4xx'
5 failed, 7 passed, 18 deselected

$ .venv/bin/python -m pytest -q tests/test_glm52_task9_adapters.py \
    -k 'ec2_exception_status_code_matrix_is_exact_nonempty_4xx_only or same_token_completer_disables_only_for_real_4xx'
12 passed, 18 deselected
```

The named cases are HTTP `200`, `301`, `399`, `400`, `499`, and `500`, plus
missing, non-string, and empty error codes. The integration cases prove the
real same-token completer retains `WATCHING` for 301/500 anomalies and moves
to `REJECTION_PROVED_AWAITING_TERMINAL_V2` only for a genuine 400 response.
The complete production-adapter file is `30 passed`.

Final original Task 9 focused GREEN:

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_sky_admission.py \
    tests/test_glm52_enforcement_launch_custody.py \
    tests/test_glm52_task9_adapters.py \
    tests/test_glm52_task9_contract.py
93 passed in 0.15s
```

Expanded Task 3, Task 9, and production H.1d integration GREEN:

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_dynamodb.py \
    tests/test_glm52_enforcement_launch_custody.py \
    tests/test_glm52_enforcement_sky_admission.py \
    tests/test_glm52_enforcement_live_authority.py \
    tests/test_glm52_task9_adapters.py \
    tests/test_glm52_task9_contract.py
633 passed in 2.44s
```

The earlier Task 3 plus Task 9 integration checkpoint was:

```text
$ .venv/bin/python -m pytest -q \
    tests/test_glm52_enforcement_dynamodb.py \
    tests/test_glm52_enforcement_launch_custody.py \
    tests/test_glm52_enforcement_sky_admission.py \
    tests/test_glm52_task9_adapters.py \
    tests/test_glm52_task9_contract.py
283 passed in 0.31s
```

## Verification

Tasks 1–9 enforcement aggregate, H.1g migration, H.1d materializer, and
Task 9 production adapters/artifact:

```text
$ uv run --frozen pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py \
    tests/test_glm52_h1d_trusted_source_materializer.py \
    tests/test_glm52_task9_adapters.py \
    tests/test_glm52_task9_contract.py
1116 passed, 2 warnings in 6.52s
```

The warnings are the accepted pre-existing SWIG deprecation warnings in source
authority fixture parity.

Frozen CloudFormation:

```text
$ uv run --frozen pytest -q tests/test_glm52_sky_cloudformation.py
26 passed in 1.50s
```

Runtime/static/artifact gates:

```text
Python 3.9.6 import-light blocked-import proof: PASS
Python 3.9.6 source compile proof: PASS, 7 production files
Python 3.13.13 target source compile: PASS, 3 changed production files
Python 3.9.6 round-three handler import-light/source compile: PASS
Canonical Task 9 artifact --check: PASS
build_h1g_task9_contract.py mode: -rwxr-xr-x
Task 9 trailing-whitespace/final-newline proof: PASS
```

The import-light guard rejects `boto3`, `botocore`, `mlx`, `mlx_vq`, `numpy`,
`torch`, and `transformers` while importing all five package modules.
The frozen environment currently has no `ruff` executable, so the prior
round's Ruff result was not represented as fresh round-two evidence.

## SHA-256

```text
8d82bdc4017cac3371bb4a9de04879775e0a6c0d502f31465ff52afc00c5ecfe  src/glm52_enforcement/sky_admission.py
a49f51d2b9fb2c3b6f4e96239e5ac7940b9da6e669c4eb599c06d5753605e3fc  src/glm52_enforcement/launch_custody.py
d2ad224d59fe9aab42148391506f6b88eb870e4ce538589f7519bb6afbfb7b13  src/glm52_enforcement/launch_wal.py
66ea0f86c3140bf48697c3631c2dc14d8d00cb51290858e660a694105d8ca3d2  src/glm52_enforcement/task9_contract.py
299b176d3b4a0fdc84ef7a30efdb189b69fcc608648760251c7e34e5aab30db5  src/glm52_enforcement/dynamodb.py
b9d633d4f4756bec51f431ca5e35c23a7db4c30387bc2da618bbc4a006e7f423  aws/glm52-gpu/skypilot/worker_launch_intent_patch.py
0a6eeed4861ba30701f9d2419dc46a3708e105d04bfb70fcdca569a9a214233e  aws/glm52-gpu/skypilot/skypilot-0.13.0-lock.txt
0225303547ca56474fbfa6505be0e50bbcf49b4ee0566758e72a090a30fe9407  aws/glm52-gpu/skypilot/server-config-v1.json
3dcc20250652aa82fd90ace1336c4e25ddf2e8cd064125024cc80f239a14ae69  aws/glm52-gpu/lambda/worker_launch_custody_handler.py
99934c136f72ed15bf624ff75133bb4fd8d92fc1b15ae867bd7c592ea02895f9  aws/glm52-gpu/scripts/build_h1g_task9_contract.py
9fd19da61c3ec4435ed620628b24d05d7d8e0cc5f0a82a4300bd5afe71ace1c6  aws/glm52-gpu/scripts/inspect_h1d_live_authority.py
b379d2fbac7c55fdca25cf77d5ae9f3f9339fe9326cd16ef02a3d93872338e6a  aws/glm52-gpu/cfn/h1g/task9-contract-v1.json
d980c71e87b8a68f97f0e72c4589b06084c60b1c7af3e2f4ceb72af8e4feddc3  tests/test_glm52_enforcement_sky_admission.py
8b290ad4922584bf74ba525efd7921c89dfd47fca6e192b1afb849d19fffb2aa  tests/test_glm52_enforcement_launch_custody.py
525ec6a2282188a08467b6452b9360dd9281be6aabeb487afa430774346115bc  tests/test_glm52_enforcement_dynamodb.py
2c118338f57a75a1d64a205033afc330ed243227ff27ff2cafe3f9817b877414  tests/test_glm52_enforcement_live_authority.py
11eed21d9ceebd24a021650ab6c36e2700b639ff2370314acec7852a1926663c  tests/test_glm52_task9_adapters.py
0d65e83494a292318a5ec6d9103afeb437e85a48d0cfabddc561ff167d4993a3  tests/test_glm52_task9_contract.py
```

## Self-review and limitations

- The real executable routes are dependency-injected boundaries. Tests prove
  their exact request construction, operation order, classification,
  readback, retry absence, and mutation handling; they do not make a live
  service or deployed-truth claim.
- The production constructor requires the deployment-owned authenticated
  ledger plan/parameter authority, function-policy reader, discovery adapter,
  and exact-instance authority. Task 9 defines and tests their closed call
  surface; packaging their concrete cloud readers and Lambda entrypoint is a
  later deployment task.
- The repository patch authenticates the inspected installed SkyPilot source
  but does not modify that environment. Packaging and deployment are owned by
  later tasks.
- DynamoDB atomicity and frozen record transitions remain exclusively in the
  Task 3 adapter and are included in the 1,047-test aggregate. Task 9 adds no
  alternate ledger writer or generic mutation route.
- Termination-call windows and counters are issued atomically by the injected
  Task 3-backed store; the watcher independently requires and validates each
  typed permit before every external call.
- No live AWS, Sky POST, launch, termination, spend, reserve, billing,
  deployment, or campaign-running claim is made by Task 9.

## Independent approval

Independent read-only re-review approved immutable commit `700219d3` with no
Critical or Important findings. The reviewer independently verified the exact
integer `400–499` positive-rejection boundary, malformed/2xx/3xx/5xx ambiguity,
downstream `WATCHING` preservation, `13/13` targeted tests, and `633/633`
focused Task 3/8/9 integration tests.
