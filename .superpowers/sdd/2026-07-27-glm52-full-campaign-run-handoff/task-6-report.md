# Task 6 Report — three-stack migration, finite fence, and support lifecycle

Date: 2026-07-28

## Scope

Implemented only the Task 6-owned pure modules, canonical builder, three inert
bootstrap assets, four focused test files, and this tracked report. The frozen
current stack, package initializer, Tasks 1–5, deployment scripts, AWS,
Organizations, network services, and live resources were not changed.

## Implemented contracts

- Exact retained/fence/support names, tags, termination-protection evidence,
  account, region, run, and nonoverlapping ownership families.
- Exact parameterless `ContainerAnchor` bootstrap and separate immutable
  fence/support coordinates.
- Pure retain/remove/import/final-fence/support-replacement template
  transformations plus an injected, executable, zero-resubmit migration
  coordinator. The migration requires archived stack IDs, current logical and
  physical ownership, bucket/import identity, exact retained deployment
  RoleId/role ARN, immutable template URLs/VersionIds, and three canonically
  equal direct `GetBucketPolicy` observations.
- The executable path is explicitly two phase. `MigrationCoordinator.bootstrap`
  starts from authority-only durable state, submits each deterministic-name
  `CreateStack` once, validates the actual service-assigned stack ARN by exact
  name/account/region/UUID/role/tags/protection/template readback, and binds
  that effect identity before `COMPLETE`. Its client tokens derive only from
  request-known names, immutable bootstrap coordinates, role, tags, run, and
  action identity—never a future stack UUID.
- Only the exact completed `MigrationBootstrapResult` can unlock
  `build_stack_migration`; fence/support evidence and all immutable artifact
  coordinates must use those bound IDs. Migration execution reloads the
  durable state and reauthenticates both bound names and IDs before any
  retained update. A fresh process resumes between either create with zero
  duplicate create calls.
- The coordinator persists canonical `NOT_SUBMITTED` -> `SUBMITTED` ->
  `COMPLETE` mutation intent around every effect through an exact
  conditional-revision state-store port. State schema v2 durably records only
  request-known authority at initialization—account, region, deterministic
  names, exact tags, role, bootstrap URL/VersionId, change-set name, run, and
  action identity—while service-assigned stack and change-set ARNs begin null.
  Corrupt, stale, out-of-order, caller-preseeded, missing, replaced, or
  rolled-back state fails closed.
- After bootstrap, the path performs both retained updates; creates, verifies,
  executes, and rereads the one exact IMPORT change set while the anchor
  remains; then removes the anchor and installs the reviewed disabled-support
  template. The exact returned/read-back ChangeSet ARN is persisted before
  completion and reauthenticated on every resume. Completed reruns return that
  same ARN, reject a replacement, and permit disappearance only when the prior
  authenticated ARN, completed execute record, exact final fence template, and
  final policy readbacks all remain coherent. Every mutation is submitted at
  most once.
- Legal `CREATE_IN_PROGRESS`, `UPDATE_IN_PROGRESS`,
  `UPDATE_COMPLETE_CLEANUP_IN_PROGRESS`, change-set create progress, and
  `IMPORT_IN_PROGRESS` states use bounded injected readback loops with no sleep
  and no second mutation. Ambiguous CreateStack, UpdateStack, CreateChangeSet,
  and ExecuteChangeSet outcomes reconcile by exact readback only.
- CloudFormation readbacks use executable service shapes: missing stacks and
  change sets are typed adapter exceptions; `TemplateBody` accepts either a
  mapping or nonempty JSON string and is parsed only inside the trusted module.
  Duplicate keys, non-finite constants, malformed JSON, non-object roots,
  non-string/noncanonical mapping keys or values, and every YAML/non-JSON body
  fail closed. No injected normalizer can assert substitute semantics.
  DescribeChangeSet proof uses exact name/ID, stack, include flag, import
  `Changes`, and canonical-JSON Original/Processed semantic templates without
  inventing RoleARN, ChangeSetType, or ResourcesToImport response fields.
- The canonical builder CLI has two reachable modes. `initialize-bootstrap`
  requires only execution authority and emits the v2 initial state without
  reading evidence or constructing artifacts. `build-migration` requires a
  completed bootstrap state, derives and validates both bound IDs before
  reading evidence/coordinates, then emits the five templates, manifest,
  execution state, and exact bootstrap-result artifact.
- Support inventory ownership is physical-identity based, not logical-name
  based. Retained aliases and arbitrary S3 buckets are rejected; only the exact
  Task 7 rehearsal bucket family is admitted. Import direction uses explicit,
  disjoint export-owner allowlists, including alias-only names such as
  `SupportHostId`.
- Canonical five-artifact migration output and resource-owner/deletion
  manifest. No direct `PutBucketPolicy` fallback exists.
- Finite generation-one manifest for preparation, batch five-source,
  reservation-only, closed-source, and terminal transitions. The batch slot
  carries the frozen class projection, never a fabricated child hash.
- One-shot Create/Describe/GetTemplate/Execute fence boundary with independent
  wire-omission evidence, literal versioned-template proof, exact single
  nonreplacement policy change, associated-role proof, and readback-only
  ambiguity reconciliation. External effects require an injected exact-live
  Task 3 arm/consume transaction around a fresh H.1f audit. Post-execute proof
  requires `EXECUTE_COMPLETE`, a strictly newer stack update timestamp,
  Original and Processed template readback by exact stack ID, canonical direct
  bucket-policy equality, and fresh two-observation H.1f stabilization.
- Finalized support deletion consumes the exact live
  `glm52_production_finalization_action` `SUPPORT_DELETE` record paired with a
  coherent `SUPPORT_FINALIZED` control before any CloudFormation mutation. It
  requires one stable protected `UPDATE_COMPLETE` prestate, one
  protection-disable call, coherent false readback, one exact-role delete, two
  consecutive exact stack-absence observations, and two complete reviewed
  resource-inventory absence audits. Strict UUID stack ARNs structurally reject
  retained/fence substitutions and residual replacement/reappearance.

## Ownership and lifecycle matrices

| Owner | Exact stack | Families | Deletion |
| --- | --- | --- | --- |
| retained | `keep-glm52-gpu` | existing infrastructure; ledger/KMS; lifecycle/finalizers; stable publisher identities | retained |
| fence | `keep-glm52-h1g-fence` | sole model-bucket policy | retained |
| support | `keep-glm52-h1g-support` | private network; host/volumes; TLS/secrets; production workflow/functions; reviewed rehearsal bucket | finalized bounded deletion |

| Transition | Exact delta |
| --- | --- |
| retention-only | add both `Retain` attributes to only the current policy |
| post-retain | remove only that policy |
| fence import | retain anchor and import one literal unconditional policy |
| final fence | remove only anchor |
| disabled support | replace only anchor with reviewed SHA-bound inventory |

## Literal RED to GREEN

Each production behavior began with one named node and an observed failure:

1. Missing three-stack module -> exact names/tags/ownership GREEN.
2. Missing bootstrap validator/asset -> exact inert anchor GREEN.
3. Missing retention builder -> only two retain attributes GREEN.
4. Missing post-retain builder -> only owned policy removed GREEN.
5. Missing fence import/final builders -> exact literal policy and
   anchor-only removal GREEN.
6. Missing support replacement/graph validators -> reviewed inventory and
   one-way import graph GREEN.
7. Missing migration evidence/builder/script -> five canonical artifacts,
   manifest, live ownership, import identity, and policy equality GREEN.
8. Missing finite manifest -> five frozen generation-one slots GREEN.
9. Missing batch projection -> closed class projection GREEN.
10. Missing request/authority/executor -> exact omitted-field wire and
    nonce-owned one-shot execution GREEN.
11. Replacement change mutant -> rejected before execute GREEN.
12. Missing returned template/role evidence -> Original/Processed/raw
    template/policy/role hashes GREEN.
13. Ambiguous create/execute exceptions -> one call each plus readback-only
    reconciliation GREEN.
14. Missing closed operation surfaces -> direct policy/role/alternate-stack
    authority absent GREEN.
15. Missing support deletion adapter -> exact sequence GREEN.
16. Protected/read-progress ambiguity -> repeated reads only, never a second
    mutation, GREEN.
17. Missing retained/fence protection guard -> alternate targets rejected
    before effects GREEN.
18. Self-review RED: non-string stack ID raised `AttributeError` -> fail-closed
    `ValueError` GREEN.
19. Self-review RED: reviewed support rehearsal bucket was over-rejected ->
    allowed while retained bucket/policy/KMS/ledger authority remains rejected,
    GREEN.
20. Self-review RED: direct dataclass construction could target the support
    stack before failing after CreateChangeSet -> full manifest reconstruction
    now rejects it before any create, GREEN.
21. Review RED: ambiguous Execute against an unchanged `UPDATE_COMPLETE`
    prestate did not raise -> exact `EXECUTE_COMPLETE`, newer stack update
    identity/time, exact stack-template reread, direct policy equality, and
    fresh post-policy H.1f stabilization GREEN.
22. Review RED: invented syntactic fence authority did not raise and could
    reach CreateChangeSet -> injected exact-live Task 3 arm/consume plus fresh
    H.1f authority now rejects it with zero effects, GREEN.
23. Review RED: no executable migration coordinator could be imported ->
    closed seven-operation, one-submit-per-mutation coordinator with exact
    CreateStack adoption and IMPORT execution GREEN.
24. Review RED: retained-bucket aliases/arbitrary support buckets were accepted,
    and exact export-owner arguments did not exist -> physical identity,
    rehearsal-bucket, and explicit export-owner graph enforcement GREEN.
25. Review RED: `UPDATE_IN_PROGRESS` support prestate had already received a
    protection mutation before failure -> exact protected `UPDATE_COMPLETE`
    prestate now precedes every mutation, GREEN.
26. Review RED: malformed retained stack UUID was accepted -> strict UUID ARN
    grammar for all three stack identities, plus reviewed residual-resource
    absence/replacement checks, GREEN.
27. Review-round-two RED: accepted-then-lost CreateStack stopped on ordinary
    `CREATE_IN_PROGRESS` -> bounded exact stack stabilization GREEN for both
    fence and support creates.
28. Review-round-two RED: ambiguous UpdateStack, CreateChangeSet, and
    ExecuteChangeSet exceptions propagated or stranded the import -> all eight
    individual mutation calls now persist intent first, submit once, and
    reconcile by readback only, GREEN.
29. Review-round-two RED: a completed first invocation performed CreateStack
    again on rerun -> exact observed-stage inference plus completed-stage
    reauthentication now emits zero second-run mutations, GREEN.
30. Review-round-two RED: no durable execution-state boundary existed ->
    canonical projection/parser, conditional revision backend port, and
    builder `--execution-state` round trip survive a fresh adapter process,
    GREEN.
31. Review-round-two RED: persisted completion could be caller-asserted or
    out of order -> exact request digest, external template/stack/change-set
    proof, prefix ordering, and stale/corrupt journal rejection with zero
    mutation, GREEN.
32. Review-round-two RED: definite AlreadyExists could not be safely adopted ->
    exact deterministic name, stack ID, role, tags, protection, and known
    template adoption GREEN; wrong-template ownership fails after one call.
33. Review-round-two RED: terminal-only single reads rejected legal progress ->
    bounded CREATE/UPDATE/IMPORT progress, timeout, rollback, replacement, and
    failure contracts GREEN without a second mutation.
34. Live-shape RED: string TemplateBody and actual DescribeChangeSet fields
    failed simulator-only assumptions -> mapping/JSON-string semantic
    authentication, exact import Changes, Original/Processed
    submitted-template proof, and fail-closed non-JSON handling GREEN.
35. Review-round-three RED: `MigrationCoordinator` had no bootstrap-only API
    and creation required caller-provided future stack UUIDs -> authority-only
    bootstrap now learns, validates, durably binds, and returns legitimate
    service-assigned IDs that deliberately differ from the old fixture GREEN.
36. Review-round-three RED: the pure builder accepted evidence before either
    create completed -> a required exact bootstrap result now gates every new
    stack artifact and coordinate GREEN.
37. Review-round-three RED: completed migration could synthesize the
    deterministic change-set name as `MigrationExecutionResult` identity ->
    direct and ambiguous paths persist the exact service ARN; completed reruns
    return it unchanged and detect replacement/missing state GREEN.
38. Review-round-three RED: the builder CLI had one route that required live
    evidence even when initializing state -> reachable `initialize-bootstrap`
    and `build-migration` phases now prove the authority-only and completed-ID
    gates end to end GREEN.
39. Review-round-three adversarial regressions: restart between containers,
    exact-digest caller-preseeded IDs, wrong account/region/name/malformed
    stack ARNs, post-bind stack replacement, untracked in-progress import,
    ambiguous import response, and bounded final-state change-set disappearance
    all fail closed or reconcile only under the exact documented contract.
40. Self-audit RED: calling the bootstrap phase after a container had already
    transitioned to a migration template still returned an identity ->
    bootstrap-only resume now requires the exact `CREATE_COMPLETE` bootstrap
    template, while migration resume derives the bound IDs from durable state
    and reauthenticates its exact allowed stage GREEN.
41. Review-round-four RED: a wrong YAML body plus a SHA-echoing injected
    normalizer could substitute the expected dictionary and authorize later
    mutations -> the semantic-normalization boundary was removed entirely.
    Trusted strict JSON parsing now rejects wrong or semantically correct YAML,
    duplicate keys, NaN/Infinity, malformed/scalar/list roots, non-string or
    noncanonical mapping keys/values, and non-JSON ChangeSet `Original`
    templates before any subsequent mutation GREEN.

## Verification

Focused Task 6:

```text
88 passed in 0.34s
```

Tasks 1–6 aggregate:

```text
557 passed, 2 warnings in 4.05s
```

The two warnings are the accepted Task 5 SWIG deprecations from fixture parity.
Frozen current CloudFormation regression:

```text
26 passed in 1.55s
```

Python/import/template/compile/whitespace:

```text
/usr/bin/python3 3.9.6 task6-round4-python39-ok
/usr/bin/python3 -m py_compile ...                         PASS
.venv/bin/python -m compileall -q ...                     PASS
blocked import subprocess                                PASS
all aws/glm52-gpu/cfn/h1g/*.json canonical parse         PASS
owned-file trailing-whitespace scan                       PASS
git diff --check over Task 6-owned tracked files          PASS
```

No new shell file exists; shell syntax verification is not applicable. Both
builder CLI phases are executable and were exercised end-to-end with canonical
temporary authority/state/evidence/archive inputs. The selected Ruff check
could not run because no `ruff` executable is installed in the local
environment; compilation, import-boundary, focused, aggregate, and frozen
regression gates all passed.

## Final SHA-256

```text
cc1cb6e0071aef53fcb5eedba9e1d9c6ed4cf300ca2d939d3c4dc97d157d4755  src/glm52_enforcement/cloudformation_stacks.py
45d7c5295c2b5712a0062b5663d8ab7d4d25a9abd444b6b831a1bac5aec94663  src/glm52_enforcement/fence_executor.py
241984785e41a23d42cd73bee1765074909581023d6e4752e0b6d9d33021395a  aws/glm52-gpu/scripts/build_h1g_stack_migration.py
03e7caa5252a308df80a87674c106f21cb8ad18241f4c1b83b8bea10b6fe9d09  aws/glm52-gpu/cfn/h1g/container-bootstrap-v1.json
03e7caa5252a308df80a87674c106f21cb8ad18241f4c1b83b8bea10b6fe9d09  aws/glm52-gpu/cfn/h1g/fence-bootstrap-v1.json
03e7caa5252a308df80a87674c106f21cb8ad18241f4c1b83b8bea10b6fe9d09  aws/glm52-gpu/cfn/h1g/support-bootstrap-v1.json
c5e1a02d1e7b6ee011ff50173f2964a3caa517fd69ce795a1434496f9f7deb6e  tests/test_glm52_enforcement_cloudformation_stacks.py
affa835c03fc54b6a3e72f5de5d3d6fdcb94df89f8ebf647b5a345fdfa5e0045  tests/test_glm52_enforcement_fence_executor.py
fc215f0cdfd8f4fdcebfac15935c2de5c774dcf971aa49c1f79dd5973ac33c07  tests/test_glm52_enforcement_cloudformation_import_boundary.py
6c87bd9339cde12aa7153a8e8416bf02e0c0ec4ac80faa311cb2a5e160f468d7  tests/test_glm52_h1g_stack_migration.py
```

## Limitations

This is executable injected-adapter contract evidence only. It does not claim
any live CloudFormation import, stack deployment, Organizations enforcement,
direct bucket-policy readback, or support deletion. The durable state store
defines and tests the exact read/conditional-write protocol but does not choose
the live persistence service. Task 7 must supply and separately review the
exact support resource inventory before disabled deployment.
