# Task 11 brief — private decision-to-POST workflow and measured closure

## Position

Task 11 begins only after Task 10 is independently approved. It implements
candidate-thirteen slice 11: the exact private decision create, fresh
authority suffix, action arm/consume, admission-owned one-wire POST, explicit
deadlines, and measurable closure budgets. It must not call live AWS/Sky or
claim numeric binding, terminal-v2, drain, or finalization; those are Task 12.

Read first:

- candidate-thirteen sections 10 and 11, phase grammar, action/admission
  transitions, H.1d clocks, archive/review gates, and slice 11;
- accepted H.1e/H.1f contracts and completed Tasks 3–10;
- freeze addendum/full-run handoff implementation and verification sections;
- current decision/workflow/Lambda/Step Functions/generated-template code.

## Constraints

- Work only in the repository; preserve concurrent work.
- Strict named RED-to-GREEN TDD with literal evidence.
- No Git mutation, live AWS/network/Sky call, deployment, launch, or billing
  effect.
- Python 3.9 import-light enforcement and Python 3.12 adapter compatibility.
- No stored/cached report, head, audit, direct-response surrogate, or
  caller-provided success can authorize a later step.
- No generic retry, no unbounded wait, no exported intermediate launch
  capability, and no function that can consume a stored decision or retry a
  POST.

## Exact production workflow

Implement the architecture's 28-step private decision-to-POST sequence
literally, including:

1. prove the preauthorized five-source batch template and transition slot;
2. acquire/reconcile CloudFormation quiescence;
3. revalidate runtime attachments, cutoff credentials, baseline and
   maintenance-seal SCPs;
4. mint the source clock and invoke the five one-shot publishers sequentially
   in canonical VersionId-dependent order;
5. create the single batch successor, separately arm/audit/consume
   change-set create and execute, invoke the exact fence executor once, and
   complete the two-readback stabilization;
6. prove executor terminal/quiescent and atomically seal/acquire barrier;
7. fresh Sky identity/RBAC/token/effective-state attestation;
8. claim writer arm, independent fresh H.1f walk, consume, create/recover;
9. honest current-clock launch/expire model;
10. decision writer arm, independent fresh H.1f walk, consume, create;
11. accept only a direct unambiguous exact `200` conditional-create response;
12. fully paginate/exact-read the decision;
13. H.1e modeled submit-once validation using that invocation's direct
    response;
14. complete fresh Task 8 H.1d/spend/deadline reinspection;
15. repeat fresh Sky attestation;
16. recheck seal/barrier/activation/epoch/revision/head/decision;
17. mint private Sky-arm nonce and own the exact `ARMED` result;
18. mint independent decision nonce and consume the sole action as an
    admission reservation;
19. coherent index/control/action readback after every transaction;
20. invoke the exact Task 9 launch-admission version once, no retry;
21. admission atomically owns `CONSUMED -> POST_STARTED`;
22. admission performs its own fresh H.1f full-namespace walk;
23. admission binds that audit/envelope in
    `POST_STARTED -> POST_AUTHORIZED`, proves nonce/revision, then issues at
    most one mTLS relay request;
24. conditionally classify accepted/known-rejected/ambiguous with no retry;
25. run another independent H.1f walk before handoff create;
26. persist correlation and immutable evidence;
27. release only to numeric-binding reconciliation; and
28. expose no stored-decision submit/retry function.

Keep the five source, batch successor, change-set create, change-set execute,
claim, decision, POST, and handoff audits distinct. Earlier/cached audit bytes
must be rejected at every later boundary.

## Failure semantics

Implement exact fail-closed behavior:

- `409`, `412`, timeout, loss, malformed response, process death, or restart
  before `POST_AUTHORIZED` yields zero POST;
- failure after authorization/sending can never retry or rearm;
- arm without consume can only become `ABANDONED` after owner hard expiry,
  terminal owner, and exact zero-side-effect proof;
- a direct decision create followed by invocation death before action
  consumption is permanently reconcile-only and later becomes
  `STORED_DECISION_NO_POST`, never a resubmission;
- every transaction ambiguity uses coherent exact readback and private nonce
  ownership;
- any live drift invalidates the all-authority seal and yields zero POSTs;
- exact callee versions only; aliases, `$LATEST`, unqualified/alternate
  functions/roles/policies fail.

## Workflow and deadline budgets

Render one exact published Standard Workflow/version with globally unique
phase names and ordered timeout classes. Bind:

- Lambda configured timeout: 840 seconds;
- closure deadline: 720 seconds;
- Step Functions Task timeout: 1,200 seconds;
- remaining-Lambda-time checks before decision construction, decision PUT,
  and Sky-token consumption;
- fewer than 120 seconds remaining before token consumption fails without
  consuming;
- finite client connect/read timeouts and zero SDK/library retry.

Implement nonoverlapping phase ledgers:

| Complete closure phase | Ceiling |
| --- | ---: |
| template/change-set/quiescence/maintenance-seal preflight | 150 s |
| client warming and immutable construction | 60 s |
| non-authoritative sizing | 90 s |
| stable TLS/Sky identity preflight | 30 s |
| fresh authority suffix | 58 s |
| post-admission handoff/evidence | 60 s |
| ambiguity/failure unwind | 55 s |
| unused reserve | 97 s |
| total measured work | 600 s |

The authority suffix starts at the earliest fresh source observation and ends
at admission relay receipt:

| Authority suffix phase | Ceiling |
| --- | ---: |
| five source cycles | 7 s |
| successor + change-set create/execute cycles | 8 s |
| two policy/deny readbacks at least ten seconds apart | 13 s |
| independent claim walk/transport | 6 s |
| independent decision walk/PUT/readback/H.1e | 8 s |
| complete Task 8 live reinspection plus Sky probe | 7 s |
| arm/reserve/POST walk/authorization/relay receipt | 7 s |
| unallocated reserve | 2 s |
| total | 58 s |

Cached prefix work cannot count as suffix authority. Phase duplication,
omission, overlap, negative duration, clock rollback, wrong order, wrong
ceiling, or a suffix/closure overrun fails closed without widening freshness.

## Measured no-POST gate

Implement the production rehearsal collector/verifier:

- at least 20 deployed-shaped no-POST rehearsals, at least five cold starts;
- exact production clients, pagination, non-VPC decision path,
  isolated-endpoint admission/attestation path, host NAT path, and frozen
  namespace scale;
- same conditional-write/direct-response/full-pagination/GET/HEAD/H.1e
  workload at a canary coordinate that can never satisfy production key
  authority and is unreadable by workers/admission;
- every phase within its ceiling, worst complete run at most 600 seconds,
  authority suffix at most 58 seconds;
- injected throttling, pagination, and one network ambiguity fail within the
  bounded unwind;
- no production decision/action/source/claim or POST effect.

Until an actual later live deployment supplies 20 valid measurements, the
production-enablement result is `CLOSURE_BUDGET_UNPROVEN`. Local simulators
prove the collector/verifier and fixture-shaped measurement logic but must not
fabricate live measurements or mark the deployed gate passed.

## Required RED-to-GREEN coverage

At minimum:

1. exact 28-step order and unique phase grammar;
2. five sequential sources and service-assigned VersionId dependency;
3. eleven distinct authority walks with cross-use/replay mutants;
4. separate change-set create versus execute authority;
5. two readbacks separated by at least ten seconds;
6. direct decision response custody and `200` metadata identity;
7. decision pagination/exact read/H.1e modeled validation;
8. Task 8 fresh reinspection and repeated Task 9 attestation;
9. seal/barrier/epoch/revision/head/action drift mutants;
10. arm/consume/post-start/post-authorized nonce ownership and readback;
11. at most one admission invocation and one relay receipt;
12. accepted/rejected/ambiguous classification without resend;
13. every pre/post-authorization crash boundary;
14. stored-decision-no-POST and abandoned-arm rules;
15. exact published-version IAM/resource-policy graph;
16. 840/720/1,200 and 120-second remaining-time gates;
17. all complete-closure and suffix phase ceilings/order/nonoverlap;
18. 58-second and 600-second boundary/overrun tests;
19. 20-run/five-cold collector requirements and malformed measurement matrix;
20. canary coordinate production-unreachability and no-POST proof;
21. throttling/pagination/network-ambiguity bounded failure;
22. Python 3.9/3.12, focused/aggregate/CloudFormation, Ruff, canonical
    workflow/artifact, and whitespace gates.

## Report

Write
`.superpowers/sdd/2026-07-27-glm52-full-campaign-run-handoff/task-11-report.md`
with files/integration edits, literal RED/GREEN, exact commands/counts,
identities/hashes, self-review, and limitations. Explicitly distinguish local
simulated measurement verifier GREEN from the still-unproven deployed
20-run gate. Make no live AWS/Sky/deployment/POST truth claim.

Return status, owned files, one-line verification counts, and concerns. Leave
changes unstaged.
