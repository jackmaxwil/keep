# Task 13 brief — exhaustive transport, clean rehearsal, and acceptance archive

## Position

Task 13 begins only after Task 12 is independently approved. It closes local
candidate-thirteen implementation: reproduce and pass every H.1e `T01`–`T25`
row, the 22 required transport mutants, complete integration/race/security/
operability matrices, disabled-stack rehearsal tooling, deterministic archive
and inventory gates, and independent reviews. It must not call live AWS/Sky,
deploy, launch, terminate, or fabricate deployed rehearsal evidence.

Read first:

- the complete candidate-thirteen architecture, especially the exact
  `T01`–`T25` table, 22 transport-mutant campaign, compatibility/race/stale
  process/archive/review gates, implementation decomposition slices 14–15,
  and live rollout order;
- accepted H.1e and H.1f source reports;
- all Task 1–12 briefs, reports, code, generated artifacts, and review
  findings;
- freeze addendum;
- full-run handoff Phases Two through Four and Definition of Done;
- current campaign, submission, fence, CloudFormation, terminal,
  qualification, checkpoint/cache/training, archive, inventory, S3 audit,
  rehearsal, and break-glass test suites/scripts.

## Constraints

- Work only in the repository and preserve concurrent/user changes.
- Strict named RED-to-GREEN TDD for every missing acceptance property.
- No Git mutation or live/cloud/network/Sky effect.
- Do not reduce the matrix, skip/xfail/deselect tests, replace behavioral
  assertions with markers, or count duplicate tests as distinct rows.
- Local/injected disabled-stack and hardware-boundary rehearsals are not
  deployed truth. Record exact limitations.
- No dirty-checkout upload, repack between modes, production action, model
  promotion/publication/upload, or new authority.

## Exact transport matrix

Copy every `T01`–`T25` row verbatim into one machine-readable acceptance
manifest. For each row bind:

- row ID and exact architecture text;
- source state/preconditions and owner/epoch/revision/nonce;
- adapter, action kind, state-machine phase, deadline class, exact external
  operation count, direct-response class, durable pre/post states;
- fresh H.1f walk kind and whether it occurs before/after the effect;
- coherent readback/adoption path;
- expected terminal/no-launch/recovery/incident result;
- exact test selector and immutable evidence identity.

One executor must run the manifest and prove all 25 rows exactly once. Reject
missing, duplicate, unknown, reordered, text-drifted, unbound, or testless
rows. The matrix must cover direct success, definite rejection, conditional
conflict, pre-send death, post-send ambiguity, response loss, owner death,
takeover, duplicate adoption, terminal transport, and every frozen H.1e
distinction without collapsing cases.

## Twenty-two transport mutants

Copy the architecture's 22 obligations verbatim into a second closed
machine-readable manifest. Each mutant must make a real accepted test fail
when injected and must be rejected by the production validator/adapter—not a
test-only string match. At minimum preserve the architecture coverage for:

- nonce/epoch/revision/owner substitution;
- audit reuse or wrong audit kind/head/revision;
- direct-response substitution;
- request/token/body/effect mismatch;
- same-name/input versus foreign execution;
- duplicate/ambiguous/lost-response mutation;
- stale or forked ledger/head;
- cross-record/cross-role/cross-version authority;
- unsupported retry/resend;
- partial pagination/readback;
- reordered/missing/duplicate arrays and lineage;
- cleanup/rollover and terminal/late-allocation races.

Report literal RED for each mutant against the corresponding final assertion
and one full GREEN run with all mutants enabled.

## Complete integration and race acceptance

Build one closed implementation matrix mapping every candidate-thirteen
component to production code, rendered/generated artifact, focused tests,
integration tests, owner stack/role/version, and independent review status.
Unclassified or fixture-only production components fail.

Run the full architecture-required matrices, including:

- original provisioner launch unreachability and genuine guarded production
  route;
- first-call liability action, same-token ambiguity, late/multiple instances,
  settlement, and no-settlement-no-refund;
- every owner death/takeover point for execution, recovery, finalization,
  cleanup, and liability;
- writer death after rollover, exact duplicate callback adoption,
  rollback-to-no-launch closure, cleanup/rollover race, three-activation
  lineage, delayed/multiple workers, honest sort-key boundary;
- all legal CloudFormation parameter combinations with invariant policy
  owner/bytes;
- constrained role/SCP/pass-role/service-linked-role denial matrix;
- three-stack ownership/import graph and disabled launch state;
- support topology, TLS/secrets/KMS/grants, limits, cost, lifecycle, deletion;
- Sky parser/request/database/relay/RBAC/attestation compatibility;
- Task 10 reboot/systemd/deadline/child-survival scenarios;
- Task 11 phase and deployed-measurement verifier;
- Task 12 cardinality/terminal/cancellation/drain/snapshot/orphan matrix.

## Deterministic clean rehearsal and archive

Implement/run the local exact Phase Two archive path without writing live S3:

- all focused campaign and enforcement suites;
- Python 3.9 and Python 3.12 target import;
- Ruff `E4,E7,E9,F`;
- every campaign shell syntax check;
- CloudFormation/policy lint and denial tests;
- canonical JSON/YAML/artifact regeneration;
- `git diff --check`;
- accepted H.1e/H.1f sentinels.

Build one repository archive from the exact reviewed source using the existing
archive script into a dedicated `mktemp` directory. Prove:

- content/hash determinism on repeated builds from the same source;
- no dirty/untracked/user-local/secret/cache/pyc content;
- every required production script/unit/policy/absolute path exists;
- extracted clean-room tests/imports/rehearsal use only the archive;
- qualification, cache-seed, and production builders reference the same
  archive bytes and cannot repack;
- staged descriptor/archive/inventory writes are no-overwrite, content first,
  readiness last;
- local artifact audit rejects zero-length/truncated/multipart/foreign/
  unversioned/checksum/tensor-range mutants;
- rehearsal stops honestly at the exact real CUDA/H100 or live-AWS boundary.

Do not create fake S3 checksum completion, H100 qualification,
`H100_RESUME_READY.json`, teacher cache, campaign markers, deployed
CloudFormation, denial probes, measured canaries, or live IDs.

## Disabled-stack rehearsal package

Generate the exact no-execute/deployment input and local simulator package
needed for the controller's later live rollout:

- finite retained/fence/support templates and manifests;
- exact parameterless fence and disabled support state;
- change-set/template/role/tag/cost/resource inventory;
- account/region/profile/credential-expiry guard sequence;
- post-deploy readback and denial probe command manifest;
- alert/DLQ/log/schedule/workflow/network/host/P5-zero checklist;
- Task 11 20-run canary measurement collector inputs;
- rollback/cleanup/reconciliation commands that retain evidence.

Every command is closed, canonical, profile/region/account explicit, and safe
to inspect before execution. Task 13 validates syntax/operation allowlists
through injected runners but does not execute it.

## Independent reviews

After implementation is green, create deterministic review packages and
obtain three independent read-only reviews:

1. specification/coverage;
2. race/security/custody;
3. AWS/operability/deployment.

Each reviewer targets immutable commit/diff/artifact hashes, runs independent
adversarial probes, reports Critical/Important findings, and makes no edits or
live calls. Fix findings with fresh RED/GREEN and re-review before local
acceptance. Do not claim local implementation complete until all three have
no Critical or Important findings.

## Required final verification

At minimum report:

- `T01`–`T25`: 25/25 exact rows;
- transport mutants: 22/22 killed;
- all Task 1–13 focused enforcement tests;
- full `tests/test_glm52_enforcement_*.py` plus H.1g migration;
- frozen CloudFormation;
- the handoff's focused campaign/submission/fence/terminal/H100/checkpoint/
  cache/training suites;
- Python 3.9 compile/import-light;
- Python 3.12 target import;
- Ruff, shell syntax, executable modes, canonical artifacts, policy/template
  ceilings, workflow history/resource cardinality, and whitespace;
- archive identity, extracted inventory, clean-room rehearsal status, and
  exact live/hardware stop boundary;
- complete implementation matrix green;
- three independent review approvals.

## Report

Write
`.superpowers/sdd/2026-07-27-glm52-full-campaign-run-handoff/task-13-report.md`
with exact manifests, files, literal RED/GREEN, commands/counts, hashes,
archive/rehearsal identities, matrix, reviews, self-review, and limitations.
Clearly list every remaining live AWS, deployed canary, CUDA/H100,
qualification, campaign, cost, and model-disposition gate for controller
execution.

Return status, files, one-line counts, archive identity, review verdicts, and
concerns. Leave changes unstaged.
