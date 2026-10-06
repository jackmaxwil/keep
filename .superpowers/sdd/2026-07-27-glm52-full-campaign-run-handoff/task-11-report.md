# Task 11 report — private decision-to-POST workflow and measured closure

## Status

Task 11 is locally complete and GREEN.

This task implemented the import-light, dependency-injected private
decision-to-POST closure and wired its exact published-version workflow into
the generated support-plane contract. It made no AWS, SkyPilot, network,
deployment, launch, POST, or billing effect.

The local 20-run rehearsal fixture verifies the collector and malformed-input
matrix only. It deliberately produces `CLOSURE_BUDGET_UNPROVEN`. No deployed
20-run/five-cold measurement authority is claimed or fabricated.

The frozen candidate-thirteen authority read for this task was
`.superpowers/sdd/goal-objective/task-3ph1g-production-enforcement-architecture.md`
at SHA-256
`1daaa963d42e5fb94ed712c07d83c6b43f16fbcc9e89b0df0c7721587d8c051e`.

## Implemented contract

### Exact private closure

`src/glm52_enforcement/decision_closure.py` now provides:

- the exact, unique 28-step sequence;
- five sequential source publications whose authenticated, service-assigned
  VersionId is the next publication's predecessor;
- eleven distinct authority audits in the frozen order:
  five sources, batch successor, change-set create, change-set execute, claim,
  decision, and Sky POST;
- an independent `SKY_POST_HANDOFF` audit that cannot replay or cross-use any
  of the eleven authority audits;
- exact direct decision-response custody: status `200`, bucket owner,
  VersionId, ETag, checksum, request ID, Date, candidate identity, decision
  audit, and current-invocation custody nonce;
- direct object-identity delivery of that response to full pagination/exact
  read and H.1e modeled-submit-once validation;
- fresh Task 8 reinspection, repeated Sky attestation, all-authority recheck,
  private arm nonce, independent decision nonce, coherent transaction
  readback, and one exact numeric launch-admission version invocation;
- admission validation for exactly one invocation, at most one relay call,
  zero retry, zero rearm, and accepted/known-rejected/ambiguous
  classification;
- permanent `STORED_DECISION_NO_POST` behavior after a direct decision exists
  but before action consumption;
- the exact `ARMED -> ABANDONED` gate requiring owner hard expiry, terminal
  owner execution, and exact zero-side-effect proof;
- rejection of any exposed stored-decision submit, consume, POST retry, POST
  rearm, raw POST/send, or jobs-launch method.

The route is executable through named injected production dependencies.
Effectful clients are not imported into the enforcement module, cached
reports/audits cannot authorize later phases, and no intermediate launch
capability is returned.

### Deadline and timing authority

The implementation freezes:

- Lambda configured timeout: 840 seconds;
- internal closure deadline: 720 seconds;
- Step Functions Task timeout: 1,200 seconds;
- minimum remaining time before token consumption: 120 seconds;
- three remaining-time checks: before decision construction, before decision
  PUT, and before token consumption;
- eight complete-closure ceilings totaling 600 seconds;
- eight nested authority-suffix ceilings totaling 58 seconds;
- two policy readbacks at least ten seconds apart.

The validator rejects wrong membership/order, duplicate phases, overlap,
positive gaps, negative time, clock rollback, per-phase overrun, wall-span
overrun, closure/suffix overrun, incorrect suffix nesting, and insufficient
policy-readback separation. Adjacent spans must account for all elapsed wall
time; summed durations alone are not accepted.

### Measured no-POST gate

The rehearsal model and verifier require:

- at least 20 unique observations;
- at least five unique cold Lambda environments;
- exact production clients, full version pagination, non-VPC decision path,
  isolated admission/attestation path, host NAT path, frozen namespace scale,
  conditional write, direct response, exact GET/HEAD, and H.1e validation;
- a dedicated rehearsal bucket/key that cannot satisfy production authority
  and is unreadable by workers and admission;
- zero production source, claim, decision, action-consume, Sky POST, and relay
  effects;
- complete closure and suffix timing validation;
- bounded throttling, pagination, and network-ambiguity failure observations,
  each unwinding within 55 seconds.

Local or serialized records are insufficient to publish a proven deployment
gate. The verifier returns `CLOSURE_BUDGET_UNPROVEN` until a later live
deployment authenticates the required measurements.

### Generated support-plane wiring

`src/glm52_enforcement/support_plane.py` now:

- configures the non-VPC Decision Lambda for 840 seconds;
- renders one Standard workflow with an initial
  `CLOSURE_BUDGET_PROVEN` choice;
- fails as `CLOSURE_BUDGET_UNPROVEN` by default;
- invokes `DecisionVersion` with a 1,200-second Task timeout;
- hands off only to `NumericBindingVersion`;
- contains no ASL `Retry`;
- publishes an exact `AWS::StepFunctions::StateMachineVersion`;
- validates the workflow, Lambda timeout, published-version references, and
  workflow-role invoke graph.

The canonical generated asset
`aws/glm52-gpu/cfn/h1g/support-task11-workflow-v1.json` records the complete
step, audit, timing, and workflow definition. The support contract manifest
was regenerated from the pure renderer.

## Literal RED evidence

### Initial contract absence

```text
$ uv run --offline pytest -q tests/test_glm52_enforcement_decision_closure.py
E   ModuleNotFoundError: No module named 'glm52_enforcement.decision_closure'
1 error in 0.05s
```

### Unmeasured timing-gap mutant

```text
$ uv run --offline pytest -q tests/test_glm52_enforcement_decision_closure.py -k timing_mutants
FAILED ... test_task11_red_timing_mutants_fail_closed[gap]
1 failed, 9 passed, 3 deselected in 0.03s
```

### Executable route absence

```text
$ uv run --offline pytest -q tests/test_glm52_task11_decision_route.py
E   ImportError: cannot import name 'AdmissionClosureResult'
1 error in 0.06s
```

### Rehearsal verifier absence

```text
$ uv run --offline pytest -q tests/test_glm52_task11_rehearsals.py
E   ImportError: cannot import name 'build_rehearsal_measurement'
1 error in 0.05s
```

### Disabled/short support workflow

```text
$ uv run --offline pytest -q tests/test_glm52_enforcement_support_plane.py::test_task11_red_support_workflow_is_budget_gated_and_version_pinned
E       assert 720 == 840
1 failed in 0.13s
```

### Arm-abandonment and crash-boundary contract absence

```text
$ uv run --offline pytest -q tests/test_glm52_task11_decision_route.py -k 'crash_boundaries or arm_abandonment'
E   ImportError: cannot import name 'transition_abandoned_arm'
1 error in 0.06s
```

## GREEN evidence

All commands were local and offline.

```text
$ uv run --offline pytest -q tests/test_glm52_enforcement_decision_closure.py
13 passed in 0.02s

$ uv run --offline pytest -q tests/test_glm52_task11_decision_route.py
24 passed

$ uv run --offline pytest -q tests/test_glm52_task11_rehearsals.py
15 passed in 0.06s

$ uv run --offline pytest -q \
    tests/test_glm52_enforcement_support_plane.py::test_task11_red_support_workflow_is_budget_gated_and_version_pinned \
    tests/test_glm52_enforcement_support_plane.py::test_checked_in_support_contract_artifacts_are_generated_canonical_and_nonlive
2 passed in 0.11s
```

Final focused command:

```text
$ uv run --offline pytest -q \
    tests/test_glm52_enforcement_decision_closure.py \
    tests/test_glm52_task11_decision_route.py \
    tests/test_glm52_task11_rehearsals.py \
    tests/test_glm52_enforcement_support_plane.py
102 passed in 0.57s
```

Frozen enforcement/stack/CloudFormation aggregate:

```text
$ uv run --offline pytest -q \
    tests/test_glm52_enforcement_*.py \
    tests/test_glm52_h1g_stack_migration.py \
    tests/test_glm52_sky_cloudformation.py
1099 passed, 2 warnings in 10.35s
```

The two pytest warnings are the existing SWIG `SwigPyPacked` and
`SwigPyObject` deprecation warnings; the interpreter also emits the existing
`swigvarlink` shutdown warning.

Additional gates:

```text
Python 3.9 py_compile: passed
Python 3.12 py_compile: passed
Python 3.9 import-light with mlx/numpy/boto3/botocore blocked: passed
Python 3.12 import-light with mlx/numpy/boto3/botocore blocked: passed
uvx --offline ruff check --select E4,E7,E9,F: All checks passed!
Explicit owned/integration trailing-whitespace/final-newline scan: passed
Canonical checked-in artifact equality: passed
Generated workflow manifest SHA validation: passed
```

## Files

New:

- `src/glm52_enforcement/decision_closure.py`
- `tests/test_glm52_enforcement_decision_closure.py`
- `tests/test_glm52_task11_decision_route.py`
- `tests/test_glm52_task11_rehearsals.py`
- `aws/glm52-gpu/cfn/h1g/support-task11-workflow-v1.json`
- this report

Integration edits:

- `src/glm52_enforcement/support_plane.py`
- `tests/test_glm52_enforcement_support_plane.py`
- `aws/glm52-gpu/cfn/h1g/support-contract-manifest-v1.json`

## Frozen SHA-256

The report is intentionally not self-hashed.

```text
b0c75bff2152c7f9aa952d797df0454706054fdb3b6787ea393fda3d0d759aeb  src/glm52_enforcement/decision_closure.py
3a8a6a62a5b315f6d0945cf701c22383d0d70dd109c030d3519edea23985b546  src/glm52_enforcement/support_plane.py
825c2a31f07362c8fc5de1e472fdd4f80a9e1adbba1b3be822a7b5f29f4cc5b1  tests/test_glm52_enforcement_decision_closure.py
e36d491ca7075d3dbea1ae9944a48cc25e93244ff1b25a4b4211cf8e3e6265a7  tests/test_glm52_task11_decision_route.py
ffd1679aedee8a081a04c0b443b39a153e1f4f6d97da34f023d52197baf9dbf5  tests/test_glm52_task11_rehearsals.py
53ed257fbc14e8b007dd1629512e047ffbc311a64512281a6c985322cb9d2b12  tests/test_glm52_enforcement_support_plane.py
7b1fd63c582335ee72f3d2833185b90c992366a11bcda91362a291ca795f697f  aws/glm52-gpu/cfn/h1g/support-task11-workflow-v1.json
343475470b581f8f3b8d76945ac602a48e546c3c695d2e8d5a9af344f0f10351  aws/glm52-gpu/cfn/h1g/support-contract-manifest-v1.json
```

## Self-review and limitations

- No stored decision or stored direct-response surrogate reaches the
  submission path.
- The two CloudFormation change-set audits are separate and replay-resistant.
- The admission boundary is invoked once; accepted, rejected, and ambiguous
  classifications have no resend edge.
- Failures after direct decision creation and before action consumption become
  permanently reconcile-only.
- Failures after consumption or authorization expose no retry/rearm method.
- All timing is monotonic, contiguous, nested, and fully accounted.
- The workflow gate is fail-closed and remains unproven from local evidence.
- This task did not deploy the workflow, collect live closure measurements,
  submit a Sky job, create a worker, claim numeric binding, create
  terminal-v2, drain, or finalize. Those are later handoff tasks.
