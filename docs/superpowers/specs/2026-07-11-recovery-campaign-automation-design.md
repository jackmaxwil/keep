# Recovery Campaign Automation Design

## Status

Approved for implementation by the user on 2026-07-11 through the attached
execution brief. This document makes that brief precise for the KEEP codebase.

## Goal

Replace manual GLM-5.2 recovery orchestration with a conservative campaign
controller that observes, plans, verifies, records, and renders the campaign
without weakening existing artifact authorities or human approval boundaries.

## Non-negotiable constraints

- No push, publish, upload, paid service, or external spend.
- Never set custom MLX wired-memory limits.
- Only the selection split may guide recovery. Holdout and report remain
  reporting-only.
- Every heavy operation cooperates with `.keep-heavy-job.lock`; overlapping
  heavy work is rejected.
- `status` and `advance --dry-run` are read-only.
- `advance` executes only a transition declared in the campaign matrix and
  records the exact command and evidence identities.
- Accepted baseline identities are immutable. Every recovery output receives a
  new candidate identity.
- Missing evidence is reported as absent, never inferred or rendered as zero.
- The diagnostic reevaluator remains diagnostic; production loading uses the
  authenticated recovery-candidate loader.
- The protected untracked lock, handoffs, and `runs/` are never staged or
  rewritten by campaign commands.

## Architecture

The implementation lives under `mlx_vq.recovery_campaign` and is registered by
the existing `keep` argparse CLI.

### Domain and configuration

`models.py` defines immutable experiment, authority, observation, transition,
and ledger-event values. `config.py` loads one strict YAML campaign matrix.
Unknown fields, duplicate experiment names, incomplete gate/up/down rate maps,
non-selection tuning inputs, unsafe output paths, and payloads over the declared
limit fail closed.

The tracked matrix is
`recipes/glm52_recovery_campaign_v1_20260711.yaml`. It declares the ordered
policy (`full75-e8`, `worst8-e8p`, `ebss`, `rotations`), immutable authorities,
expected artifacts, command templates, payload budgets, and decision gates. It
does not contain mutable run status.

### Observation

`observer.py` produces a read-only `CampaignObservation` from current state:

- tests the advisory lock with a nonblocking `flock`;
- discovers matching local processes from `ps` output;
- counts canonical recovered groups and mixed-artifact links;
- authenticates a published conversion manifest when present;
- reads evaluation/attribution identities when present;
- derives throughput and ETA only from observed completed groups and elapsed
  process time;
- reports contradictions such as a complete manifest with missing groups.

The observer never creates directories, locks, manifests, or ledger entries.

### Ledger and evidence registry

`ledger.py` manages an append-only JSONL ledger in ignored artifact storage.
Every event contains a schema version, monotonically increasing sequence,
timestamp, event kind, experiment name, canonical payload, previous-event hash,
and its own canonical SHA-256. Loading verifies the complete chain and rejects
truncation, mutation, duplicate sequence numbers, or unknown fields.

Evidence entries record paths plus content hashes, candidate relationships,
diagnostic/release eligibility, recovery levers, commands, and verification
results. Paths are descriptive; hashes are authoritative.

### Controller

`controller.py` is a deterministic state machine. It consumes the matrix,
observation, and ledger and returns one of:

- `wait`: a declared heavy transition is healthy and running;
- `resume`: an incomplete resumable transition has no active owner;
- `audit`, `reevaluate`, or `attribute`: the predecessor evidence is complete;
- `ready`: the experiment packet is complete and the next policy decision is
  human-owned;
- `blocked`: evidence is contradictory or an undeclared process owns the lock;
- `complete`: every declared experiment and required evidence is complete.

`advance --dry-run` prints the exact next transition. `advance` re-observes the
world, acquires the heavy lock when required, writes a `transition_started`
ledger event, launches the exact declared command, and writes terminal evidence
only after exit and fresh authentication. It never guesses a command from shell
history.

Detached operations use a small launcher record containing PID, command hash,
start time, stdout/stderr path, and experiment identity. Stale PIDs do not prove
completion.

### Verification profiles

`verification.py` exposes named profiles:

- `glm52-recovery-mixed-rate`
- `glm52-recovery-loader`
- `glm52-recovery-candidate`
- `glm52-recovery-campaign`

Each profile declares exact owned files, pytest modules, compile targets, CLI
help checks, Metal/heavy-lock policy, protected paths, and required review
record. The verifier runs commands without a shell, reports every exit code,
checks staged paths, and emits a machine-readable commit-readiness result.

### Authenticated artifact primitives

`mlx_vq.io.authenticated_artifacts` consolidates stable JSON reads,
`O_NOFOLLOW` regular-file authentication, descriptor hashing, transactional
manifest publication, COW snapshots with explicit bounded fallback, and
descriptor lifecycle management. Recovery producer, auditor, loader, and route
capture migrate only after compatibility tests prove byte-identical identities.

### Review and lane records

`review.py` stores declarative review scopes: owned files, frozen decisions,
out-of-scope paths, threat model, verification evidence, and resolved finding
fingerprints. It classifies imported findings as new, duplicate, resolved,
out-of-scope, or known diagnostic behavior. It does not decide that a new
finding is invalid; ambiguous classifications stay new.

`lanes.py` records companion lane requests and results, including read/write
mode, ownership fence, RED/GREEN commands, lock-wait detection, and final diff
scope. External worker invocation remains an adapter so the domain is testable
without Codex.

### Rendering

`render.py` produces terminal/JSON status and a generated Markdown handoff from
the matrix, verified ledger, and current observation. The handoff includes the
branch, commits, active jobs, ETA, evidence hashes, quality curve, blockers,
next permissible action, protected paths, and approval boundaries.

## CLI

The public surface is:

```text
keep recovery campaign status [--config PATH] [--ledger PATH] [--json]
keep recovery campaign advance [--config PATH] [--ledger PATH] [--dry-run]
keep recovery campaign handoff [--config PATH] [--ledger PATH] [--out PATH]
keep recovery campaign verify PROFILE [--json]
keep recovery lanes run NAME [--dry-run]
keep verify PROFILE [--json]
```

`keep verify` and `keep recovery campaign verify` share the same implementation.

## Error handling

- Configuration and ledger errors exit 2 with a precise validation message.
- Contradictory evidence exits 3 and does not mutate state.
- A held or ambiguous heavy lock makes mutating advance exit 4.
- A transition subprocess failure is recorded and returns its nonzero result.
- Missing optional evidence keeps status successful but marks the field absent.
- Status exits nonzero only when current evidence is contradictory or corrupt.

## Testing

All behavior starts with headless tests. Required coverage includes strict
matrix parsing, real-flock observation, process/ETA parsing, manifest states,
ledger integrity, controller transitions, dry-run non-mutation, verification
profiles, review classification, deterministic handoff rendering, CLI parsing,
and live read-only proof against the running full-75 job.

Real heavy commands are never run by unit tests. Host execution begins only
after dry-run output is compared with the already authenticated manual command.

## Human-owned decisions

Automation must stop for quality-slope investment decisions, acceptance of a
new size/quality tradeoff, quiet-machine scheduling, threshold or threat-model
changes, external spend, and publication.
