# Task 9 brief — Sky admission, deterministic launch custody, and liability

## Position in the implementation

Task 9 starts only after Task 8's fresh H.1d and spend-authority implementation
is independently approved. It implements candidate-thirteen decomposition
slice 9. It must not expose the operator production CLI, build worker units,
submit a real Sky request, call live AWS, or claim decision-closure latency.

Read these sources first:

- `.superpowers/sdd/goal-objective/task-3ph1g-production-enforcement-architecture.md`,
  especially the launch-liability, post-terminal-allocation, Sky identity,
  IAM, relay, admission, residual-tail, implementation-decomposition slice 9,
  and acceptance sections;
- `docs/superpowers/plans/2026-07-28-glm52-candidate13-implementation-freeze-addendum.md`;
- `docs/superpowers/plans/2026-07-27-glm52-full-campaign-run-handoff.md`,
  Phase One sections 5.3 and 5.4;
- `docs/superpowers/approvals/2026-07-28-glm52-campaign-owner-approvals.md`;
- completed Task 2, Task 3, Task 7, and Task 8 implementations and reports;
- current guarded submission, Sky, CloudFormation, and break-glass code under
  `aws/glm52-gpu/`;
- the installed/pinned SkyPilot `0.13.0` source only for read-only identity
  inspection. Do not mutate the installed environment.

## Global constraints

- Work only in `/Users/jack.mazac/Developer/keep`.
- You are not alone in the repository. Preserve other work and never revert
  unrelated changes.
- Use strict named RED-to-GREEN TDD and record literal evidence.
- No Git mutation, AWS/network call, deployment, launch, termination, billing
  effect, or installed-environment mutation.
- `src/glm52_enforcement` remains import-light and Python 3.9 compatible.
- Account, region, run, profile, market, and shape are exactly
  `246813579024`, `us-west-2`, `glm52-sky-20260724`, `keep-gpu`,
  on-demand, and one `p5.48xlarge`.
- No Spot, Capacity Block, alternate region/cloud/shape, raw controller
  launch, caller-created token, caller-created request parameters, library
  retry, or generic PassRole path.
- The Task 8 reserve and authenticated residual-liability approval are
  mandatory before `POSSIBLY_SENT`; do not duplicate or widen spend authority.

## Required production contracts

### Pinned loopback Sky and closed identity

Implement exact contracts that bind:

- SkyPilot version `0.13.0`, wheel/dist-info, interpreter, dependency lock,
  server configuration, and original plus patched provisioner source hashes;
- loopback-only API-server reachability;
- one least-privileged authenticated Sky service account and closed RBAC;
- distinct mutually unusable attestation, launch-admission, numeric-binding,
  and retained-cancellation identities/ports;
- an effective-controller/consolidation result that proves the combined host
  is the sole accepted Sky controller without an inert controller resource
  pin;
- no Sky secret, backend path, launch token, or direct host route in the
  decision function;
- `jobs.launch` remains `retryable=False`.

The attestation request accepts only its closed freshness nonce and contains no
operator-controlled command, IMDS, launch, or cancellation parameter.
Admission accepts one canonical closed request body, independently re-reads
the required Task 8 authority, verifies the action/reserve/current owner, and
issues at most one POST. Caller-asserted probe success is never authority.

### Prepared-WAL provisioner and deterministic token

Patch the repository-owned/pinned Sky provisioner integration so the original
provisioner:

1. computes the exact immutable launch parameter identity and deterministic
   EC2 ClientToken;
2. persists a durable prepared launch intent before `POSSIBLY_SENT`;
3. performs the one Task 8 reserve transaction;
4. atomically creates/binds the exact WORKER_LAUNCH and
   WORKER_LAUNCH_LIABILITY records;
5. waits for coherent `WATCHING` ownership;
6. signals only that an owned launch attempt is ready; and
7. polls exact ledger results.

The original provisioner must have no production-reachable `RunInstances`
call, nested create-instances retry, terminate/start authority, or PassRole.
No pre-POST mount upload or subprocess is allowed. A replacement token may be
allocated only after exact prior terminal, settlement, spend-close, and
allocation-close evidence.

### Sole same-token completer

Implement one exact retained same-token completer as the sole sender for
attempt one and all later attempts. It:

- accepts only activation and allocation ordinal;
- exact-reads immutable launch/token/parameter bytes itself;
- consumes one liability action and shared atomic attempt counter before
  every call, including attempt one;
- sends at most six serialized zero-SDK-retry calls in the first six minutes;
- uses byte-identical parameters and the same token every time;
- cannot create or select a token, ordinal, subnet, role, tags, volume shape,
  request body, or Sky action;
- enforces one on-demand `p5.48xlarge`, approved AMI, exact subnet/security
  group/profile/tags, IMDSv2, and one encrypted delete-on-termination
  300-GiB gp3 root at 3,000 IOPS and 125 MiB/s with no data disk;
- treats a transport-ambiguous response as possibly sent and reconciles only
  by exact readback;
- transitions a positive service rejection to the exact rejection-pending
  state and disables further completion;
- never calls a second token for the same allocation.

IAM tests must distinguish expressible launch-shape/tag enforcement from the
function-level ClientToken boundary. Do not invent a nonexistent EC2 IAM
ClientToken condition key.

### Retained liability watcher and termination custody

Implement the exact retained one-minute liability owner/watcher:

- atomic acquisition/takeover with the existing owner fields, nonce, revision,
  execution/version/dispatch identity, and hard expiry;
- one-minute absolute cadence, 20-second scan/action deadline, parameterless
  failover schedule, event accelerators that are never deadline authority,
  and no hidden retries;
- exact-token CloudTrail, exact-tag/window EC2, state-change, launch evidence,
  and spend evidence correlation;
- no-instance scans never settle liability;
- late/multiple instances open charged intervals and are drain-only when work
  is not authorized;
- exact-instance termination continues after useful-compute authorization
  expires until terminality, spend close, and settlement authenticate;
- up to six explicit same-instance termination calls within the nominal
  360-second settling objective, all action-recorded and reconciled;
- unresolved control-plane delay becomes an incident but never disables
  monitoring or termination;
- no later activation, allocation ordinal, or token while any liability is
  nonsettled.

The nominal 900 seconds and `$13.76` plus separate 300-GiB root tail at most
`$0.01` are approved reserves, not AWS hard caps. Preserve the explicit
no-hard-billing-cap semantics.

### Post-terminal final view and settlement

Use the already frozen Task 2 schemas and Task 3 adapter. Implement:

- one POST_TERMINAL_ALLOCATION per late instance, with the exact closed
  `DISCOVERED -> ALLOCATION_OPEN -> INSTANCE_TERMINAL ->
  ALLOCATION_CLOSED` chain;
- canonical ordinal/instance-sorted post-terminal and merged-final arrays;
- exact terminal instance and spend-close evidence for every instance;
- the two exact settlement kinds:
  `NO_INSTANCE_POSITIVE_REJECTION` and
  `ALL_INSTANCES_TERMINAL_AND_SPEND_CLOSED`;
- one no-retry transaction that conditionally creates settlement, moves
  liability to the compatible terminal state, and binds/releases the reserve;
- coherent lost-response adoption and exact duplicate read-only adoption;
- no caller-supplied array, cardinality, hash, terminal, settlement, or spend
  head;
- incident preservation after later valid settlement;
- refund/release impossible before authenticated settlement and every
  allocation close.

## Real executable adapters

Repository-owned AWS/Lambda/Sky glue must be production-capable through
dependency-injected clients:

- STS/account/region guard before any effect;
- botocore total attempts exactly one and no library retry;
- exact closed operation allowlist;
- published-version and resource-policy identity checks;
- canonical request/effect/readback evidence;
- no fixture loader as the only production route.

Task 9 may add dedicated scripts/handlers and generated contract artifacts,
but must not modify the installed Sky environment or call live services.

## Required named RED-to-GREEN coverage

At minimum:

1. pinned Sky/interpreter/config/original/patched source identities;
2. loopback, RBAC, effective controller, and four mutually unusable identities;
3. attestation freshness and caller-asserted success rejection;
4. admission accepted/rejected/timeout/connection-loss/replay/process-death
   matrix with at most one POST;
5. prepared WAL before reserve and `POSSIBLY_SENT`;
6. deterministic token and exact parameter identity;
7. original provisioner production `RunInstances` unreachability;
8. no pre-POST mount/subprocess;
9. first-call liability action and shared attempt-counter consumption;
10. six-call/six-minute same-token bound and zero hidden retry;
11. wrong token/parameter/AMI/market/type/subnet/SG/profile/tag/IMDS/volume
    mutants;
12. crash before and after each durable boundary with no second send;
13. positive rejection versus ambiguous transport classification;
14. watcher acquisition, takeover, cadence, event acceleration, and
    no-zero-scan settlement;
15. late/multiple instance allocation, exact termination, repeated
    same-instance termination, and post-authorization custody;
16. 30-day incident without monitoring/termination loss;
17. post-terminal allocation ordering and merged-final view;
18. both settlement kinds, lost-response adoption, exact duplicate,
    foreign/missing/open-member rejection;
19. no-settlement-no-refund and exact reserve-release identity;
20. no new activation/token/allocation while nonsettled;
21. Python 3.9 import-light and Python 3.12 target-adapter import;
22. focused, aggregate, frozen CloudFormation, Ruff, script syntax, canonical
    artifacts, executable modes, and `git diff --check`.

## Verification and report

Run focused Task 9 tests, Tasks 1–9 enforcement aggregate including
`tests/test_glm52_h1g_stack_migration.py`, frozen
`tests/test_glm52_sky_cloudformation.py`, Python 3.9 compile/import-light,
Python 3.12 adapter import proof, Ruff `E4,E7,E9,F`, shell syntax where
applicable, canonical artifact regeneration, and whitespace checks.

Write
`.superpowers/sdd/2026-07-27-glm52-full-campaign-run-handoff/task-9-report.md`
with status, owned files/integration edits, literal RED/GREEN, exact commands
and counts, hashes, self-review, and limitations. State explicitly that tests
use injected boundaries and make no live AWS, Sky POST, launch, termination,
spend, or deployed-truth claim.

Return only status, owned-file summary, one-line verification counts, and
concerns. Leave all changes unstaged.
