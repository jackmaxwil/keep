# Recovery Campaign Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved KEEP recovery campaign controller, verification profiles, authenticated artifact primitives, evidence registry, orchestration records, and generated handoff.

**Architecture:** A pure `mlx_vq.recovery_campaign` domain consumes a strict tracked matrix and append-only ignored ledger. Thin CLI adapters observe or advance the state machine; trust-sensitive filesystem behavior is centralized under `mlx_vq.io`.

**Tech Stack:** Python 3.11+, argparse, dataclasses, PyYAML, JSONL, `fcntl.flock`, subprocess, hashlib, pytest.

## Global Constraints

- Status and dry-run are read-only.
- Heavy work never overlaps and never sets custom MLX wired limits.
- Only selection evidence guides recovery.
- Accepted identities are immutable; missing evidence is never fabricated.
- Protected untracked paths are not staged or rewritten.
- No push, publication, upload, paid service, or external spend.

---

### Task 1: Strict campaign matrix and domain values

**Files:** Create `src/mlx_vq/recovery_campaign/{models,config,__init__}.py`, `recipes/glm52_recovery_campaign_v1_20260711.yaml`; test `tests/test_recovery_campaign_config.py`.

**Interfaces:** `load_campaign_config(path) -> CampaignConfig` and immutable experiment, authority, observation, and transition values.

- [ ] Write failing exact-key, unique-name, selection-only, rate-triplet, safe-path, and budget tests.
- [ ] Run the focused module and verify missing-API RED.
- [ ] Implement strict dataclass parsing and canonical budget math.
- [ ] Add the tracked full75/worst8/EBSS/rotation matrix with current authorities.
- [ ] Re-run GREEN and commit `feat: define GLM52 recovery campaign`.

### Task 2: Read-only observer and status CLI

**Files:** Create `observer.py`, `render.py`, `cli.py`; modify `src/mlx_vq/build/cli.py`; test observer and CLI modules.

**Interfaces:** `observe_campaign(config, repo_root, ps_output=None) -> CampaignObservation`, `render_status(...)`, and `configure_recovery_parser(...)`.

- [ ] Write failing real-flock, process, group, manifest, throughput/ETA, contradiction, and CLI tests.
- [ ] Implement observation without creating or changing paths.
- [ ] Register `keep recovery campaign status` and JSON rendering.
- [ ] Prove live status against full75 with identical filesystem state before/after.
- [ ] Commit `feat: observe recovery campaigns`.

### Task 3: Hash-chained ledger and evidence registry

**Files:** Create `ledger.py`; test `tests/test_recovery_campaign_ledger.py`.

**Interfaces:** `load_ledger(path) -> tuple[LedgerEvent, ...]`, `append_event(...) -> LedgerEvent`.

- [ ] Write RED tests for canonical hashes, continuity, mutation, truncation marker, duplicate fields, and append-only writes.
- [ ] Implement atomic append plus full-chain verification and rerun GREEN.
- [ ] Commit `feat: record recovery evidence ledger`.

### Task 4: Deterministic controller and guarded advance

**Files:** Create `controller.py`, `launcher.py`; modify campaign CLI; test controller and advance.

**Interfaces:** `plan_next_transition(...) -> Transition`, `advance_campaign(..., dry_run) -> AdvanceResult`.

- [ ] Write RED tests for wait/resume/audit/reevaluate/attribute/ready/blocked/complete, exact commands, dry-run non-mutation, and overlap refusal.
- [ ] Implement pure planning and guarded subprocess/ledger adapters.
- [ ] Prove live `advance --dry-run` returns wait for full75.
- [ ] Commit `feat: advance recovery campaign safely`.

### Task 5: Named verification profiles

**Files:** Create `verification.py`; modify campaign/build CLI; test profiles.

**Interfaces:** `get_verification_profile(name)`, `run_verification_profile(...)`.

- [ ] Write RED tests for commands, staged scope, protected paths, Metal/lock policy, and review records.
- [ ] Implement four named profiles and both CLI entry points.
- [ ] Run every profile permitted while full75 owns the lock.
- [ ] Commit `feat: add recovery verification profiles`.

### Task 6: Shared authenticated artifact primitives

**Files:** Create `src/mlx_vq/io/authenticated_artifacts.py`; migrate materializer, auditor, and composite loader; test primitives and all existing recovery suites.

**Interfaces:** stable JSON authentication, `AuthenticatedFile`, transactional manifest publication, and `DescriptorSnapshot`.

- [ ] Write RED ABA, mutation, symlink, interruption, fallback, free-space, lifecycle, and identity-compatibility tests.
- [ ] Implement primitives and migrate one consumer at a time without changing canonical hashes.
- [ ] Run complete recovery/loader suites and adversarial review.
- [ ] Commit `refactor: centralize authenticated artifacts`.

### Task 7: Review and lane orchestration records

**Files:** Create `review.py`, `lanes.py`; modify campaign CLI; test both.

- [ ] Write RED ownership, scope, deduplication, frozen-boundary, stuck-lock, RED/GREEN, and result tests.
- [ ] Implement pure classification and an injectable companion adapter.
- [ ] Register `keep recovery lanes run NAME --dry-run`.
- [ ] Commit `feat: orchestrate recovery review lanes`.

### Task 8: Generated handoff and completion audit

**Files:** Modify `render.py`, campaign CLI, and `docs/GLM52_PIPELINE_READINESS.md`; test handoff rendering.

- [ ] Write RED tests for branch/commit, jobs/ETA, hashes, quality curve, honest absence, blocker, next action, protected paths, and approvals.
- [ ] Implement terminal/JSON/Markdown output and atomic `--out` publication.
- [ ] Generate and authenticate a live handoff.
- [ ] Run all new/affected suites, CLI help, compile/diff checks, and adversarial review.
- [ ] Commit `feat: generate recovery campaign handoffs`.

## Execution choice

The user explicitly requested execution, so this session uses inline execution with TDD checkpoints and an adversarial gate after each substantive wave.
