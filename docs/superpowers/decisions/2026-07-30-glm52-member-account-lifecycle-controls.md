# 2026-07-30 GLM-5.2 member-account lifecycle controls

## Owner decision

For future GLM-5.2 execution requirements, do not create, attach, enumerate,
or prove bootstrap, baseline, or maintenance-seal AWS Organizations Service
Control Policies (SCPs). Use existing member-account permissions and
member-account lifecycle controls instead. This supersedes candidate-v13 SCP
clauses and does not assert current deployed truth.

## Retained guards

- Fresh STS proof must identify account `246813579024`; record
  exact `DescribeOrganization` identity, and use `us-west-2`.
- Bind the immutable reviewed artifact, version, and checksum; retain approval
  records and the authenticated spend ledger.
- Permit at most one on-demand `p5.48xlarge`, with watchdog, termination, and
  reconciliation controls.
- Do not use Spot, Capacity Blocks, or a raw launch path.
