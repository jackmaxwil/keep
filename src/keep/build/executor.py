"""Execute a planned recipe graph with content-addressed resume.

Resume semantics:

1. A ``completed`` ledger entry with a matching step key and intact outputs
   is reused — never re-run, never errored. This resolves the "trainer
   refuses an existing output dir" friction: freshness comes from key-named
   directories, so the runner never passes ``--allow-existing``.
2. ``gate_failed`` entries are immutable rejections; the same key is not
   re-executed (``--regate`` re-evaluates recorded evidence instead).
3. Partial directories (started without a terminal event, or present with
   no ledger entry) are quarantined by rename, never deleted.
4. ``--from <step>`` force-invalidates that step and its DAG descendants.
5. Executing a noncacheable step also force-invalidates its DAG descendants,
   so live evidence cannot feed a cached downstream decision.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from keep.build.gates import GateResult, evaluate_gate
from keep.build.ledger import Ledger
from keep.build.runner import BuildPlan, PlannedStep

RECOVERY_TAGS = {
    "train-low-rank": lambda params: f"sidecar:layer{params.get('layer')}",
}
MODEL_LOADING_OPS = frozenset(
    {
        "audit",
        "benchmark-lane-s",
        "eval",
        "eval-repair",
        "glm52-production-generation-probe",
        "qwen-family-candidate-benchmark-latency-probe",
        "qwen-family-control-benchmark-latency-probe",
        "qwen-moe-artifact-audit",
        "resident-byte-audit",
    }
)
HEAVY_JOB_LOCK_PATH = Path(__file__).resolve().parents[3] / ".keep-heavy-job.lock"


def _descendants(plan: BuildPlan, root_id: str) -> set[str]:
    reached = {root_id}
    changed = True
    while changed:
        changed = False
        for planned in plan.steps:
            if planned.spec.id in reached:
                continue
            for ref in planned.spec.inputs.values():
                if ref.kind == "step" and ref.name in reached:
                    reached.add(planned.spec.id)
                    changed = True
                    break
    return reached


def _missing_outputs(planned: PlannedStep) -> list[str]:
    """Return declared output names that are absent/empty after a run.

    This is the fail-loud guard: a step whose subprocess exits 0 but does not
    materialize a declared output (e.g. materialize-sweep asked for budget 2.0 but
    the tier allocator emitted only a 3.0 candidate, so ``artifact_bpw2p0`` never
    appears) must NOT be marked completed — otherwise a downstream step silently
    resolves the missing output to a stale/fallback path and produces a bogus verdict.
    """
    missing: list[str] = []
    for name, path in planned.output_paths.items():
        if name in planned.op_def.evidence_flags or name in planned.op_def.file_outputs:
            if not path.exists() or path.stat().st_size == 0:
                missing.append(name)
        else:
            if not path.is_dir():
                missing.append(name)
            elif planned.op_def.produces_artifact and not (
                path / planned.op_def.artifact_manifest_name
            ).exists():
                missing.append(name)
        if (
            planned.spec.op == "materialize-sweep"
            and name.startswith("artifact_bpw")
            and path.is_dir()
            and _requested_sweep_rerounds(planned.spec.params)
            and not _sweep_reround_actions_recorded(path)
        ):
            missing.append(f"{name}:reround_action_count")
    return missing


def _requested_sweep_rerounds(params: dict[str, Any] | Any) -> bool:
    if not isinstance(params, dict):
        params = dict(params)
    for key, value in params.items():
        if "reround" not in key:
            continue
        if value is None or value is False:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value <= 0:
                continue
        return True
    return False


def _sweep_reround_actions_recorded(candidate_dir: Path) -> bool:
    summary_paths = (
        candidate_dir / "sweep-summary.json",
        candidate_dir / "dynamic-materialization-summary.json",
    )
    for summary_path in summary_paths:
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            return False
        if not isinstance(summary, dict):
            return False
        count = summary.get("reround_action_count")
        if count is None:
            reround_summary = summary.get("reround_summary")
            if isinstance(reround_summary, dict):
                count = reround_summary.get("reround_action_count")
        return isinstance(count, int) and not isinstance(count, bool) and count > 0
    return False


def _uses_heavy_job_lock(planned: PlannedStep) -> bool:
    if os.environ.get("GLM45_DISABLE_HEAVY_LOCK") or (
        planned.op_def.self_managed_heavy_lock
    ):
        return False
    return planned.op_def.produces_artifact or planned.spec.op in MODEL_LOADING_OPS


@contextmanager
def _heavy_job_lock(planned: PlannedStep):
    if not _uses_heavy_job_lock(planned):
        yield
        return
    HEAVY_JOB_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HEAVY_JOB_LOCK_PATH.open("a") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _outputs_intact(planned: PlannedStep) -> bool:
    return not _missing_outputs(planned)


def _quarantine(planned: PlannedStep) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    target = planned.step_dir.with_name(
        f"{planned.step_dir.name}.quarantine-{stamp}-{uuid4().hex}"
    )
    planned.step_dir.rename(target)
    return target


def _prepare_resumable_partial(planned: PlannedStep) -> list[str]:
    """Keep atomic group checkpoints but invalidate stale completion markers."""

    removed: list[str] = []
    marker_paths = [planned.step_dir / "step-result.json"]
    if planned.op_def.produces_artifact:
        manifest_path = planned.out_dir / planned.op_def.artifact_manifest_name
        if manifest_path.is_file():
            previous_manifest = manifest_path.with_name(
                f"{manifest_path.name}.resume-previous.json"
            )
            previous_manifest.unlink(missing_ok=True)
            manifest_path.replace(previous_manifest)
            removed.append(str(manifest_path))
    marker_paths.extend(
        path
        for name, path in planned.output_paths.items()
        if name in planned.op_def.evidence_flags and path.suffix != ".jsonl"
    )
    for path in marker_paths:
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    if planned.out_dir.is_dir() and not planned.op_def.manages_partial_files:
        for path in planned.out_dir.glob("*.partial-*"):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    return removed


def _gate_context(planned: PlannedStep) -> dict[str, Path | None]:
    artifact_dir: Path | None = None
    evidence_path: Path | None = None
    for name, path in planned.output_paths.items():
        if name in planned.op_def.evidence_flags:
            if evidence_path is None:
                evidence_path = path
        elif name in planned.op_def.file_outputs:
            continue
        elif artifact_dir is None:
            artifact_dir = path
    control = planned.resolved_inputs.get("control_evidence")
    return {
        "artifact_dir": artifact_dir,
        "evidence_path": evidence_path,
        "control_evidence": control,
    }


def _evaluate_step_gate(planned: PlannedStep) -> GateResult | None:
    if planned.spec.gate is None:
        return None
    context = _gate_context(planned)
    return evaluate_gate(
        profile=planned.spec.gate.profile,
        overrides=planned.spec.gate.thresholds,
        artifact_dir=context["artifact_dir"],
        evidence_path=context["evidence_path"],
        control_evidence=context["control_evidence"],
    )


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


def _seed_ancestry(planned: PlannedStep) -> tuple[list[dict[str, Any]], list[str]]:
    seed_dir = planned.resolved_inputs.get("seed_artifact")
    if seed_dir is None:
        return [], []
    manifest_path = Path(seed_dir) / "conversion-manifest.json"
    if not manifest_path.exists():
        return [], []
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return [], []
    build_steps = manifest.get("build_steps")
    recovery = manifest.get("recovery")
    return (
        build_steps if isinstance(build_steps, list) else [],
        recovery if isinstance(recovery, list) else [],
    )


def emit_build_steps(
    planned: PlannedStep,
    plan: BuildPlan,
    gate_result: GateResult | None,
) -> list[Path]:
    """Record cumulative promotable lineage in each output artifact manifest."""

    entry: dict[str, Any] = {
        "step_id": planned.spec.id,
        "op": planned.spec.op,
        "op_version": planned.op_def.version,
        "step_key": planned.step_key,
        "params": dict(planned.spec.params),
        "input_hashes": dict(planned.input_hashes),
        "resolved_argv": list(planned.argv),
        "evidence_paths": [
            str(path)
            for name, path in sorted(planned.output_paths.items())
            if name in planned.op_def.evidence_flags
        ],
        "gate_result": gate_result.to_json_dict() if gate_result is not None else None,
        "recipe": {
            "name": plan.recipe.name,
            "source_path": plan.recipe.source_path,
        },
    }
    ancestry, ancestor_recovery = _seed_ancestry(planned)
    recovery = list(ancestor_recovery)
    tagger = RECOVERY_TAGS.get(planned.spec.op)
    if tagger is not None:
        tag = tagger(planned.spec.params)
        if tag not in recovery:
            recovery.append(tag)

    updated: list[Path] = []
    for name, path in planned.output_paths.items():
        if name in planned.op_def.evidence_flags or name in planned.op_def.file_outputs:
            continue
        manifest_path = Path(path) / "conversion-manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        manifest["build_steps"] = [*ancestry, entry]
        manifest["recovery"] = recovery
        _atomic_write_json(manifest_path, manifest)
        updated.append(manifest_path)
    return updated


def _print(message: str) -> None:
    print(message, flush=True)


def _write_step_result(
    planned: PlannedStep,
    *,
    status: str,
    subprocess_returncode: int,
    gate_result: GateResult | None,
    failure_reason: str | None = None,
    regated: bool = False,
) -> None:
    payload: dict[str, Any] = {
        "step_id": planned.spec.id,
        "step_key": planned.step_key,
        "status": status,
        "op": planned.spec.op,
        "subprocess_returncode": subprocess_returncode,
        "resolved_argv": planned.argv,
        "input_hashes": planned.input_hashes,
        "output_paths": {
            name: str(path) for name, path in planned.output_paths.items()
        },
        "gate_result": gate_result.to_json_dict() if gate_result else None,
    }
    if failure_reason is not None:
        payload["failure_reason"] = failure_reason
    if regated:
        payload["regated"] = True
    _atomic_write_json(planned.step_dir / "step-result.json", payload)


def _recorded_subprocess_returncode(
    planned: PlannedStep,
    terminal: dict[str, Any],
) -> int | None:
    for field in ("subprocess_returncode", "returncode"):
        value = terminal.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            return value

    step_result_path = planned.step_dir / "step-result.json"
    if step_result_path.exists():
        try:
            step_result = json.loads(step_result_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if step_result.get("step_key") != planned.step_key:
            return None
        value = step_result.get("subprocess_returncode")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    # Completed ledgers predate explicit subprocess-returncode recording. Before
    # nonzero blocked return codes were supported, completion implied exit 0.
    return 0 if terminal.get("event") == "completed" else None


def execute_plan(
    plan: BuildPlan,
    *,
    from_step: str | None = None,
    until_step: str | None = None,
    regate: bool = False,
    continue_on_gate_fail: bool = False,
    spawn=subprocess.run,
) -> int:
    ledger = Ledger(plan.build_root / "ledger.jsonl")
    for warning in plan.warnings:
        _print(f"warning: {warning}")

    if regate:
        return _regate_plan(plan, ledger)

    step_ids = [planned.spec.id for planned in plan.steps]
    if from_step is not None and from_step not in step_ids:
        _print(f"error: --from step {from_step!r} is not in the recipe")
        return 1
    if until_step is not None and until_step not in step_ids:
        _print(f"error: --until step {until_step!r} is not in the recipe")
        return 1
    explicit_forced = _descendants(plan, from_step) if from_step is not None else set()
    forced = set(explicit_forced)
    had_gate_failures = False

    for planned in plan.steps:
        spec = planned.spec
        terminal = ledger.latest_terminal(planned.step_key)
        force_this = spec.id in forced
        explicit_force_this = spec.id in explicit_forced

        if not force_this and planned.op_def.cacheable and terminal is not None:
            if terminal["event"] == "completed" and _outputs_intact(planned):
                _print(f"skip {spec.id}: completed ({planned.step_key})")
                if until_step == spec.id:
                    break
                continue
            if terminal["event"] == "gate_failed":
                if continue_on_gate_fail and _outputs_intact(planned):
                    had_gate_failures = True
                    _print(
                        f"skip {spec.id}: gate_failed ({planned.step_key}); "
                        "continuing because --continue-on-gate-fail is set"
                    )
                    if until_step == spec.id:
                        break
                    continue
                _print(
                    f"stop {spec.id}: gate_failed recorded for key {planned.step_key}; "
                    "change the step spec or re-check thresholds with --regate"
                )
                return 1

        if not planned.op_def.cacheable:
            forced.update(_descendants(plan, spec.id))

        if planned.step_dir.exists() and (
            not planned.op_def.resume_partial or explicit_force_this
        ):
            quarantined = _quarantine(planned)
            ledger.append(
                "quarantined",
                step_id=spec.id,
                step_key=planned.step_key,
                quarantined_to=str(quarantined),
            )
            _print(f"quarantined prior output of {spec.id} -> {quarantined}")
        elif planned.step_dir.exists():
            invalidated = _prepare_resumable_partial(planned)
            ledger.append(
                "resuming_partial",
                step_id=spec.id,
                step_key=planned.step_key,
                output_dir=str(planned.out_dir),
                invalidated_completion_markers=invalidated,
            )
            _print(f"resume partial output of {spec.id} in {planned.step_dir}")

        planned.evidence_dir.mkdir(parents=True, exist_ok=True)
        ledger.append(
            "started",
            step_id=spec.id,
            step_key=planned.step_key,
            op=spec.op,
            argv=planned.argv,
            output_dir=str(planned.out_dir),
        )
        _print(f"run {spec.id}: {' '.join(planned.argv)}")
        log_path = planned.step_dir / "step.log"
        with log_path.open("w") as log_handle:
            with _heavy_job_lock(planned):
                result = spawn(planned.argv, stdout=log_handle, stderr=subprocess.STDOUT)
        if result.returncode not in planned.op_def.accepted_returncodes:
            ledger.append(
                "failed",
                step_id=spec.id,
                step_key=planned.step_key,
                returncode=result.returncode,
                log=str(log_path),
            )
            _print(f"fail {spec.id}: exit {result.returncode} (log: {log_path})")
            return 1

        missing = _missing_outputs(planned)
        if missing:
            ledger.append(
                "failed",
                step_id=spec.id,
                step_key=planned.step_key,
                returncode=result.returncode,
                log=str(log_path),
                missing_outputs=missing,
            )
            _print(
                f"fail {spec.id}: subprocess exited {result.returncode} but declared "
                "outputs missing/empty "
                f"{missing} — refusing to mark completed (would let downstream consume a "
                f"fallback). log: {log_path}"
            )
            return 1

        gate_result = _evaluate_step_gate(planned)
        if result.returncode != 0 and (
            gate_result is None or gate_result.passed
        ):
            failure_reason = "accepted_nonzero_returncode_with_passing_gate"
            ledger.append(
                "failed",
                step_id=spec.id,
                step_key=planned.step_key,
                returncode=result.returncode,
                log=str(log_path),
                reason=failure_reason,
                gate_result=gate_result.to_json_dict() if gate_result else None,
            )
            _write_step_result(
                planned,
                status="failed",
                subprocess_returncode=result.returncode,
                gate_result=gate_result,
                failure_reason=failure_reason,
            )
            _print(
                f"fail {spec.id}: accepted nonzero exit {result.returncode} denotes a "
                "blocked result but the structured gate passed; refusing completion"
            )
            return 1
        if gate_result is not None and not gate_result.passed:
            had_gate_failures = True
            ledger.append(
                "gate_failed",
                step_id=spec.id,
                step_key=planned.step_key,
                subprocess_returncode=result.returncode,
                gate_result=gate_result.to_json_dict(),
                evidence_paths=[
                    str(path)
                    for name, path in sorted(planned.output_paths.items())
                    if name in planned.op_def.evidence_flags
                ],
            )
            _print(
                f"gate fail {spec.id} [{gate_result.profile}]: "
                + ", ".join(gate_result.reasons)
            )
            step_result = {
                "step_id": spec.id,
                "step_key": planned.step_key,
                "status": "gate_failed",
                "op": spec.op,
                "subprocess_returncode": result.returncode,
                "resolved_argv": planned.argv,
                "input_hashes": planned.input_hashes,
                "output_paths": {
                    name: str(path) for name, path in planned.output_paths.items()
                },
                "gate_result": gate_result.to_json_dict(),
            }
            _atomic_write_json(planned.step_dir / "step-result.json", step_result)
            if not continue_on_gate_fail:
                return 1
            if until_step == spec.id:
                break
            continue

        manifest_updates: list[Path] = []
        if spec.step_class == "promotable":
            manifest_updates = emit_build_steps(planned, plan, gate_result)
        ledger.append(
            "completed",
            step_id=spec.id,
            step_key=planned.step_key,
            subprocess_returncode=result.returncode,
            gate_result=gate_result.to_json_dict() if gate_result else None,
            output_dir=str(planned.out_dir),
            manifest_updates=[str(path) for path in manifest_updates],
        )
        step_result = {
            "step_id": spec.id,
            "step_key": planned.step_key,
            "status": "completed",
            "op": spec.op,
            "subprocess_returncode": result.returncode,
            "resolved_argv": planned.argv,
            "input_hashes": planned.input_hashes,
            "output_paths": {name: str(path) for name, path in planned.output_paths.items()},
            "gate_result": gate_result.to_json_dict() if gate_result else None,
        }
        _atomic_write_json(planned.step_dir / "step-result.json", step_result)
        gate_note = (
            f" gate={gate_result.profile}:pass" if gate_result is not None else ""
        )
        _print(f"done {spec.id}{gate_note}")
        if until_step == spec.id:
            break

    promotion_rc = _evaluate_promotion(
        plan,
        ledger,
        allow_gate_failures=continue_on_gate_fail,
    )
    return 1 if had_gate_failures else promotion_rc


def _promotion_gate_results(plan: BuildPlan) -> list[tuple[str, GateResult]]:
    promotion = plan.recipe.promotion
    if promotion is None:
        return []
    results: list[tuple[str, GateResult]] = []
    for gate in promotion.gates:
        planned = plan.step(gate.evidence_step)
        context = _gate_context(planned)
        results.append(
            (
                gate.evidence_step,
                evaluate_gate(
                    profile=gate.profile,
                    overrides=gate.thresholds,
                    artifact_dir=context["artifact_dir"],
                    evidence_path=context["evidence_path"],
                    control_evidence=context["control_evidence"],
                ),
            )
        )
    return results


_WOW_ONLY_CHECKS = {"mean_kld", "p999_kld", "top1", "domain_top1"}


def _evaluate_promotion(
    plan: BuildPlan,
    ledger: Ledger,
    *,
    allow_gate_failures: bool = False,
) -> int:
    promotion = plan.recipe.promotion
    if promotion is None:
        return 0
    acceptable_terminal_events = {"completed"}
    if allow_gate_failures:
        acceptable_terminal_events.add("gate_failed")
    for gate in promotion.gates:
        planned = plan.step(gate.evidence_step)
        terminal = ledger.latest_terminal(planned.step_key)
        if (
            terminal is None
            or terminal["event"] not in acceptable_terminal_events
            or (terminal["event"] == "gate_failed" and not _outputs_intact(planned))
        ):
            _print(
                "promotion: pending (evidence step "
                f"{gate.evidence_step} has not completed)"
            )
            return 0
    results = _promotion_gate_results(plan)
    balanced_pass = all(
        all(passed for name, passed in result.checks.items() if name not in _WOW_ONLY_CHECKS)
        for _, result in results
    )
    wow_pass = all(result.passed for _, result in results)
    status = (
        "balanced_rc_pass_community_wow_pass"
        if balanced_pass and wow_pass
        else "balanced_rc_pass_community_wow_miss"
        if balanced_pass
        else "balanced_rc_fail"
    )
    payload = {
        "artifact_step": promotion.artifact_step,
        "artifact_dir": str(plan.step(promotion.artifact_step).out_dir),
        "status": status,
        "gates": {step_id: result.to_json_dict() for step_id, result in results},
    }
    _atomic_write_json(plan.build_root / "promotion-result.json", payload)
    _print(f"promotion: {status}")
    for step_id, result in results:
        marker = "pass" if result.passed else "MISS: " + ", ".join(result.reasons)
        _print(f"  {step_id} [{result.profile}]: {marker}")
    return 0 if balanced_pass else 1


def _regate_plan(plan: BuildPlan, ledger: Ledger) -> int:
    all_pass = True
    for planned in plan.steps:
        if planned.spec.gate is None:
            continue
        terminal = ledger.latest_terminal(planned.step_key)
        if terminal is None:
            _print(f"regate {planned.spec.id}: no recorded run")
            all_pass = False
            continue
        if terminal["event"] not in {"completed", "gate_failed"}:
            _print(
                f"regate {planned.spec.id}: latest run ended as "
                f"{terminal['event']}; a successful recorded run is required"
            )
            all_pass = False
            continue
        if not _outputs_intact(planned):
            _print(f"regate {planned.spec.id}: declared outputs are missing or invalid")
            all_pass = False
            continue
        recorded_returncode = _recorded_subprocess_returncode(planned, terminal)
        if recorded_returncode is None:
            _print(
                f"regate {planned.spec.id}: recorded subprocess return code is missing "
                "or invalid"
            )
            all_pass = False
            continue
        if terminal["event"] == "completed" and recorded_returncode != 0:
            _print(
                f"regate {planned.spec.id}: completed record has nonzero subprocess "
                f"exit {recorded_returncode}; rerun required"
            )
            all_pass = False
            continue
        try:
            gate_result = _evaluate_step_gate(planned)
        except FileNotFoundError as error:
            _print(f"regate {planned.spec.id}: evidence missing ({error})")
            all_pass = False
            continue
        assert gate_result is not None
        ledger.append(
            "regated",
            step_id=planned.spec.id,
            step_key=planned.step_key,
            gate_result=gate_result.to_json_dict(),
            previous_terminal_event=terminal["event"],
            recorded_subprocess_returncode=recorded_returncode,
        )
        rerun_required = (
            gate_result.passed
            and terminal["event"] == "gate_failed"
            and (
                not planned.op_def.regate_can_complete
                or recorded_returncode != 0
            )
        )
        if (
            gate_result.passed
            and terminal["event"] == "gate_failed"
            and not rerun_required
        ):
            # A threshold change flipped a recorded rejection: promote the
            # recorded evidence to completed without re-running model work.
            ledger.append(
                "completed",
                step_id=planned.spec.id,
                step_key=planned.step_key,
                subprocess_returncode=recorded_returncode,
                gate_result=gate_result.to_json_dict(),
                output_dir=str(planned.out_dir),
                regated=True,
            )
            _write_step_result(
                planned,
                status="completed",
                subprocess_returncode=recorded_returncode,
                gate_result=gate_result,
                regated=True,
            )
        elif gate_result.passed and terminal["event"] == "completed":
            _write_step_result(
                planned,
                status="completed",
                subprocess_returncode=recorded_returncode,
                gate_result=gate_result,
                regated=True,
            )
        elif not gate_result.passed:
            if terminal["event"] == "completed":
                ledger.append(
                    "gate_failed",
                    step_id=planned.spec.id,
                    step_key=planned.step_key,
                    subprocess_returncode=recorded_returncode,
                    gate_result=gate_result.to_json_dict(),
                    evidence_paths=[
                        str(path)
                        for name, path in sorted(planned.output_paths.items())
                        if name in planned.op_def.evidence_flags
                    ],
                    regated=True,
                )
            _write_step_result(
                planned,
                status="gate_failed",
                subprocess_returncode=recorded_returncode,
                gate_result=gate_result,
                regated=True,
            )
        marker = (
            "pass but rerun required"
            if rerun_required
            else "pass"
            if gate_result.passed
            else "fail: " + ", ".join(gate_result.reasons)
        )
        _print(f"regate {planned.spec.id} [{gate_result.profile}]: {marker}")
        all_pass = all_pass and gate_result.passed and not rerun_required
    promotion_rc = _evaluate_promotion(plan, ledger)
    return 0 if all_pass and promotion_rc == 0 else 1
