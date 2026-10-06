# 2026-07-31 H.1e / H.1f identity sentinel re-review

Status: **implementation review complete; owner acceptance pending.**

Handoff `docs/superpowers/plans/2026-07-27-glm52-full-campaign-run-handoff.md`
§6 requires: *“Confirm all preexisting accepted H.1e and H.1f identity sentinels
remain exact, or document and re-review any deliberately replaced dependency.”*
Two identities no longer match the owner-accepted 2026-07-30 baseline. This is
the exact-byte re-review.

## Accepted baseline

The owner accepted
`docs/superpowers/decisions/2026-07-30-glm52-h1e-h1f-sentinel-manifest.json`
at `2026-07-30T21:32:44Z`.

- Baseline manifest SHA-256: `3f931ed1dd7195c897ee91adb20975ce0fb7ef58022be00a7c273c342696d1c8`
- Baseline Git commit: `c97cdae83e2f5bc72dd84629b24e0daa1d2d18ee`
- Authority coverage: 38 rows — 20 H.1e pins and 18 H.1f pins — across 30 distinct paths.

A fresh path-by-path rehash found **28 exact and 2 drifted** identities against
that accepted baseline.

## Exact current manifest

Machine-readable authority:
`docs/superpowers/decisions/2026-07-31-glm52-h1e-h1f-sentinel-manifest.json`

Manifest SHA-256:
`40e65a659591ae9a152f9af79c27f8b91c693f480dc9a7a976a9fb66f833d12a`

The manifest retains three complete identities for every path: the original
H.1e/H.1f authority bytes, the 2026-07-30 owner-accepted baseline bytes, and the
current bytes. Every identity uses a full 64-character SHA-256 plus byte and
line counts.

| Path | Accepted baseline SHA-256 | Accepted size | Current SHA-256 | Current size |
|---|---|---:|---|---:|
| `aws/glm52-gpu/scripts/submit_sky_campaign.py` | `113c327825017b8c53b85959104910ca653c33a7b537db689b9977b64ca02c25` | 7704 lines / 276096 bytes | `67b0427b71392ff63f0c8eebdb795fe8fa3ddb5726b9c50e9293707e4c9a3a83` | 8793 lines / 318317 bytes |
| `tests/test_glm52_sky_submission_integration.py` | `41a7ae27fb753afb6cff7d74a3f8f309b3dd09292f339ad4cb838210d917e93a` | 7392 lines / 246136 bytes | `9066262522938e06f4a12db14fb09f0d1382bb3e63743aec268ddf536c1b9b42` | 8110 lines / 271287 bytes |

Both current byte sequences are committed in
`4e0becabe5d4af0db0fb5a795fd231cb24da8ee2` (`feat: close GLM-5.2 production
joiners`). They are Git-recoverable independently of this report.

### Recovery coverage

Current identities are Git-recoverable for **28 of 30** distinct paths. The two
unchanged exceptions remain:

- `.superpowers/sdd/goal-objective/task-3ph1e-production-generation-authority-brief.md`
- `.superpowers/sdd/goal-objective/task-3ph1e-production-generation-authority-report.md`

Those ignored `.superpowers` documents are absent from Git and are not embedded
in the manifest. Their workspace bytes rehash exactly to both the accepted and
current identities. The manifest identifies but cannot reconstruct them after
workspace loss. Owner acceptance does not create a durable recovery copy.

## Semantic classification

The production script change replaces a placeholder dynamic must-start stack
adapter with a read-only, fail-closed inspector. It now:

1. authenticates the exact retained CloudFormation stack, reviewed template,
   account, region, parameters, outputs, tags, and complete resource set;
2. reads and cross-checks the Lambda, IAM, EventBridge, Scheduler, SQS, Logs,
   SSM, and bucket identities that make up the run-bound must-start mechanism;
3. accepts only the exact reviewed policy/template projection and exact
   run/intent/controller-baseline coordinates;
4. returns no launch authority when a required resource is absent, incomplete,
   foreign, malformed, or inconsistent; and
5. introduces no create, update, delete, submit, launch, or other mutating AWS
   call.

The test-file change adds adversarial coverage for that inspector and verifies
the complete service-client wiring. It covers exact success, absent resources,
foreign coordinates, malformed SDK responses, policy/template drift, role and
resource drift, disabled or incomplete schedules/rules, and ordinary AWS
transport failure.

These are deliberate **contract-tightening changes**. They close the prior
readiness-inspection gap; they do not loosen an H.1e/H.1f production invariant.

## Governing-constraint audit

| Constraint | Finding |
|---|---|
| Account `246813579024` | Exact stack ARN, parameters, tags, and resource identities are required; no account widening |
| Region `us-west-2` | Exact stack and resource region remains required |
| One on-demand `p5.48xlarge` | Worker count, instance type, and market constraints are unchanged |
| 24 h / `$1,320.96` GPU envelope | Unchanged |
| 900 s / `$13.76` residual reserve | Unchanged |
| 300-GiB root tail ≤ `$0.01` | Unchanged |
| Spot / Capacity Blocks / alternate shape / raw launch | No launch path added; the new code is read-only and required before launch |
| Exact JSON `null` semantics | No authority schema null field was widened or defaulted |
| No automatic promotion or publication | Unchanged; inspection yields readiness evidence only |

## Verification reproduced after the drift

- Complete H.1g focused gate: **1601 passed**.
- Named enforcement gate, excluding only the two documented pointer-dependent
  files: **2423 passed**.
- Launch-plan repository gate: **825 passed**.
- Canonical fence executor/support integration focus: **221 passed**.
- Python 3.9 target-import proof: **1 passed**.
- Ruff selected `E4,E7,E9,F`: clean.
- Shell syntax over every campaign shell script: clean.
- CloudFormation lint on `gpu-teacher-stack.yaml`: clean.

## Owner acceptance required

Acceptance must be explicit. It would approve only:

1. superseding the two 2026-07-30 accepted whole-file identities with the exact
   current identities recorded above and in the v2 manifest; and
2. treating the read-only dynamic must-start inspector and its tests as a
   deliberate tightening of the guarded production-submission contract.

It would not approve a governing-constraint exception, GPU spend, stack
mutation, launch, publication, or disposition action.
