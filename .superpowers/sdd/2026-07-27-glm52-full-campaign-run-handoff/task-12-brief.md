# Task 12 brief — correlation, terminal-v2, retained drain, and finalization

## Position

Task 12 begins only after Task 11 is independently approved. It implements
candidate-thirteen decomposition slices 12 and 13: request correlation,
numeric binding, runtime/cardinality observation, exhaustive terminal-v2,
bounded cancellation, controller quiescence, graceful/forced drain,
recovery-to-teardown sealing, retained terminal/final writers, forensic
snapshot cleanup, grant revocation, orphan audit, and retained evidence.

It must not call live AWS/Sky, deploy, launch, cancel, terminate, snapshot, or
delete. Task 13 supplies exhaustive acceptance/rehearsal; later controller
work supplies live rollout.

Read first:

- candidate-thirteen sections 12 onward through implementation decomposition,
  including every exact schema, outcome, state grammar, timeout, writer,
  cancellation, drain, settlement, snapshot, cleanup, rollover-lineage,
  retained-resource, and orphan rule;
- accepted H.1e/H.1f plus Tasks 2–11 implementations/reports;
- freeze addendum and full-run handoff deadline/terminal phases;
- existing terminal, cancellation, watchdog, break-glass, recovery,
  checkpoint, snapshot, and audit code/tests.

## Constraints

- Repository-only work; preserve concurrent changes.
- Strict named RED-to-GREEN TDD with literal evidence.
- No Git mutation or live side effect.
- Python 3.9 import-light enforcement and Python 3.12 adapter compatibility.
- HTTP acceptance, request correlation, numeric binding, worker allocation,
  terminal job state, allocation/spend closure, terminal-v2, support
  deletion, and final drain are distinct facts and records.
- No absence-by-time, cached report, approximate count, filename, alarm OK,
  SSM Online, or caller-provided array/hash becomes authority.
- No terminal outcome rearms this activation or allocates another generation.

## Request correlation and numeric binding

Implement the exact immutable `SKY_POST_HANDOFF.json` boundary. Request ID is
always present and either a nonempty exact string or JSON null.
`binding_state` remains `reconcile-required`.

The exact published numeric-binding path:

- uses only the separate read-only binding relay, never launch admission or a
  POST path;
- with UUID: exact-reads that request;
- without UUID: exhaustively paginates request database and append-only relay
  journal over the immutable user/request-kind/campaign/activation/
  generation/action/envelope/body/task/job-name/time-window tuple;
- distinguishes zero, one, and multiple request matches;
- conditionally records the sole match in
  `SKY_REQUEST_CORRELATED.json` using its own fresh H.1f ACTION protocol;
- rejects any missing correlation field exposed by the pinned database;
- creates `SKY_JOB_BOUND.json` only for one exact positive numeric job with
  matching request/name/time/body/task/controller/allocation evidence;
- never treats binding as terminality.

Multiple matches are an incident: retain-cancel all and choose none. Zero is
not absence until the full post-window/cancellation/quiescence proof. A direct
accepted request may remain PENDING/WAITING/RUNNING. No-job disposition
requires exact terminal FAILED/CANCELLED plus two complete zero-observation
sets separated by the frozen quiescence interval. `SUCCEEDED` without exact
binding is live drift and triggers retained drain scanning.

## Runtime observation and exhaustive terminal-v2

Implement exact current request/job/controller/worker/allocation/spend/
liability observation through closed injected protocols and conditionally
create terminal-v2 only through the retained exact-version writer.

Reproduce the complete architecture terminal-v2 schema and exhaustive outcome
set, including:

- `STORED_DECISION_NO_POST`;
- `CONSUMED_PROVED_NO_POST`;
- known rejected or ambiguous proved-no-request/no-worker;
- accepted request FAILED/CANCELLED with no job/worker;
- pre-allocation failures;
- normal success/failure/cancellation with exact binding/allocation;
- unbound worker, multiple bound/unbound worker, request/cardinality, and
  one-request/multiple-worker incidents;
- unresolved execution start;
- unresolved request with controller quiesced;
- source/fence/live-authority/admission/closure failures;
- activation-rollover deployment failed no launch;
- authorization-expired unstarted and drain/deferred outcomes;
- every closed no-in-seal-continuation branch defined by the architecture.

Terminal-v2 binds canonical ordered discriminated request evidence,
worker-launch evidence, liability, normal allocation, terminal-instance, and
spend-close arrays with exact hashes/cardinalities/nullability. Request and
worker cardinality are orthogonal. A matching instance first discovered after
terminal-v2 is represented only by Task 9 POST_TERMINAL_ALLOCATION plus the
settlement-bound merged final view; terminal-v2 itself is immutable.

Every writer invocation uses a fresh record-specific H.1f walk, nonce/audit
ownership, conditional create, exact duplicate read-only adoption, and
coherent lost-response readback.

## Cancellation and controller quiescence

Implement bounded exact request/job cancellation through the distinct
cancellation relay/certificate/journal:

- exact closed cancel variants/endpoints/bodies only;
- send intent/result fsynced and hash chained;
- launch/cancel identities and ports mutually unusable;
- no cancellation path can address launch or arbitrary state;
- zero/one/multiple request branches retain all evidence;
- controller/scheduler work must be terminal/quiescent before no-job closure
  or forced worker termination;
- cancellation ambiguity uses readback and never resends without explicit
  action ownership;
- controller quiescence precedes forced worker drain/termination.

## Retained recovery and drain

Implement exact retained workflows/versions and owners:

- recovery sealing first blocks every launch/admission route while permitting
  only closed no-launch recovery actions;
- prove support execution terminal, correlate/classify/handoff/cancel,
  request graceful worker stop, observe, and publish terminal-v2 before
  teardown sealing;
- local Task 10 timer/worker evidence remains primary; retained
  parameterless SSM drain is backup;
- exact combined host may be stopped; termination only after its bounded
  deadline and separately consumed last-resort action;
- exact-tag P5 termination only after controller quiescence;
- no start/replacement/profile/network/launch-template/foreign mutation;
- useful compute stops at authorization exhaustion, but Task 9 exact-instance
  termination/liability settlement continues;
- every allocation closes only after terminal instance evidence and exact
  spend close;
- incidents preserve resumable evidence and never invent success.

## Retained terminal/final writers and support teardown

Implement exact published retained:

- execution-start reconciler/observer;
- terminal-v2 writer;
- finalization owner/writer;
- H1G-drained writer;
- worker-drain signal;
- operator-disposition writer;
- orphan auditor;
- support deletion/readback integration using Task 7's retained Standard
  Workflow version.

Support deletion begins only after terminal-v2 and teardown sealing, retains
all authority needed to prove absence, revoke grants, clean snapshots, audit
orphans, settle late liability, and publish final drain. Deletion ambiguity
uses retained workflow/readback and never process-local resend.

`H1G_DRAINED` requires exact support absence, no active/unsettled work,
terminal-v2, finalization, all allocation/spend/liability closures or the
explicit architecture-permitted durably owned incident state, exact KMS grant
baseline restoration, retained-resource inventory, orphan audit, and cleanup
lineage. It cannot erase a late allocation or incident.

## Forensic snapshot cleanup and rollover lineage

Implement the independently owned seven-day deletion-time snapshot lifecycle:

- exact source volume/KMS/tags and captured service-assigned snapshot ID;
- immutable SNAPSHOT_CLEANUP_CONTROL/ACTION chain;
- cleanup deadline and schedule identity;
- arm/consume/delete/describe/readback only for that snapshot;
- no early delete, alternate snapshot, volume/compute mutation, or same-ID
  retry without exact action/readback;
- cleanup versus rollover atomic race;
- rollover binds terminal-v2, complete current Task 9 settlement/final
  allocation view, H1G-drained, cleanup state/lineage, and any mandatory
  operator disposition;
- three-activation transitive cleanup lineage;
- old cleanup writer cannot mutate a new activation or index.

All activation-specific KMS grants, including service-created grants, require
baseline/diff/ListGrants/CloudTrail/resource/context/settling attribution and
must be retired/revoked to the frozen retained baseline. No snapshot recovery
grant exception.

## Orphan and retained-cost audit

Audit exact campaign P5s, combined host, volumes, EIPs, ENIs, security groups,
subnets/routes/endpoints/NAT, schedules, workflows, functions/versions/logs/
alarms/DLQs/SNS, secrets, grants, temporary buckets, fence/support stacks,
Sky requests/jobs/controllers, open allocations, liability, and snapshots.

Enumerate intentional retained foundation/fence/ledger/KMS/evidence/
publisher/lifecycle/liability resources separately and bind their ongoing
cost/retention identities. Unknown, duplicate, foreign, still-billable, or
unreadable state blocks final drain.

## Required RED-to-GREEN coverage

At minimum:

1. handoff schema/request-ID nullability and immutable tuple;
2. UUID and no-UUID exhaustive pagination/correlation;
3. zero/one/multiple request paths and duplicate/cyclic page mutants;
4. exact numeric binding and job/controller/allocation mutants;
5. accepted async request versus terminal no-job proof;
6. request and worker cardinality cross product, including one request/many
   workers;
7. every terminal-v2 outcome/schema/nullability/ordered-array/hash;
8. fresh audit/writer/duplicate/lost-response boundaries;
9. immutable terminal-v2 plus late allocation corrected final view;
10. cancellation endpoint/body/journal/cert separation and ambiguity;
11. controller quiescence before forced termination;
12. recovery seal, teardown seal, and no-launch ordering;
13. graceful stop/observer/checkpoint evidence and forced incident path;
14. allocation/spend/liability closure and authorization-exhaustion behavior;
15. exact retained writer/version/IAM/resource-policy graph;
16. support deletion/workflow/absence/finalization restart boundaries;
17. H1G-drained complete prerequisite matrix;
18. snapshot capture/control/action/deadline/delete/readback;
19. cleanup-rollover atomic race and three-activation lineage;
20. complete KMS grant attribution/revocation/baseline restoration;
21. orphan versus intentional-retained complete inventory/cost matrix;
22. Python 3.9/3.12, focused/aggregate/CloudFormation, Ruff, canonical
    workflow/artifacts, script syntax, and whitespace.

## Report

Write
`.superpowers/sdd/2026-07-27-glm52-full-campaign-run-handoff/task-12-report.md`
with owned/integration files, literal RED/GREEN, commands/counts, hashes,
self-review, and limitations. Make no live cancellation, termination,
snapshot, deletion, drain, cost, or AWS-truth claim.

Return status, owned files, one-line verification counts, and concerns. Leave
changes unstaged.
