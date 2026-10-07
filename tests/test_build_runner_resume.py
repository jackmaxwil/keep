from __future__ import annotations

import fcntl
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from keep.build import executor
from keep.build.executor import execute_plan
from keep.build.ledger import Ledger
from keep.build.ops import REGISTRY
from keep.build.recipe import load_recipe
from keep.build.runner import plan_recipe

# Stub "trainer": refuses an existing output dir (like the real primitives),
# then writes a manifest with sidecars and a training-log evidence line.
_TRAIN_STUB = """
import json, os, sys
out = sys.argv[1]
log = sys.argv[2]
if os.path.exists(out):
    sys.exit(3)
os.makedirs(out)
with open(os.path.join(out, "conversion-manifest.json"), "w") as fh:
    json.dump({"continuous_parameters": {"sidecars": [{"layer": 45}],
        "run": {"final_train_loss": 4.71}}}, fh)
with open(log, "a") as fh:
    fh.write(json.dumps({"step": 12, "loss": 4.71}) + "\\n")
"""

_RESUMABLE_TRAIN_STUB = """
import json, os, sys
out = sys.argv[1]
log = sys.argv[2]
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "conversion-manifest.json"), "w") as fh:
    json.dump({"continuous_parameters": {"sidecars": [{"layer": 45}],
        "run": {"final_train_loss": 4.71}}, "fresh": True}, fh)
with open(log, "a") as fh:
    fh.write(json.dumps({"step": 12, "loss": 4.71}) + "\\n")
"""

_FAIL_WITH_CHECKPOINT_STUB = """
import os, sys
checkpoint = sys.argv[1]
os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
with open(checkpoint, "wb") as fh:
    fh.write(b"complete-group")
raise SystemExit(7)
"""

_EVAL_STUB_TEMPLATE = """
import json, sys
rows = []
for i in range(4):
    rows.append({{"prompt_id": "report_math_%03d" % i, "top1_agreement": 0.9,
        "mean_kld": 0.2, "ppl_ratio": 1.0, "nll_delta": 0.01,
        "token_klds": [0.1], "pageouts_delta": {pageouts}, "swapouts_delta": 0}})
with open(sys.argv[1], "a") as fh:
    for row in rows:
        fh.write(json.dumps(row) + "\\n")
"""

_REPAIR_STUB = """
import json, sys
rows = []
for i in range(4):
    rows.append({"prompt_id": "report_math_%03d" % i, "top1_agreement": 0.9,
        "mean_kld": 0.2, "ppl_ratio": 1.0, "nll_delta": 0.01,
        "token_klds": [0.1], "pageouts_delta": 0, "swapouts_delta": 0})
with open(sys.argv[1], "a") as fh:
    for row in rows:
        fh.write(json.dumps(row) + "\\n")
"""


_SWEEP_ZERO_REROUND_STUB = """
import json, os, sys
out = sys.argv[1]
os.makedirs(out)
with open(os.path.join(out, "conversion-manifest.json"), "w") as fh:
    json.dump({"artifact": "stub"}, fh)
with open(os.path.join(out, "dynamic-materialization-summary.json"), "w") as fh:
    json.dump({"reround_summary": {"reround_action_count": 0}}, fh)
"""


def _write_recipe(
    tmp_path: Path,
    *,
    with_promotion: bool = False,
    with_selection: bool = False,
    with_repair: bool = False,
) -> Path:
    seed = tmp_path / "seed-artifact"
    seed.mkdir(exist_ok=True)
    (seed / "conversion-manifest.json").write_text(
        json.dumps({"recovery": ["rkd"], "build_steps": [{"step_id": "seed_step"}]})
    )
    for split in ("report", "select"):
        cache = tmp_path / f"cache-{split}"
        cache.mkdir(exist_ok=True)
        (cache / "metadata.jsonl").write_text(
            json.dumps({"prompt_id": f"{split}_math_001"}) + "\n"
        )
    raw = {
        "schema_version": 1,
        "name": "tiny-exec",
        "external_inputs": {
            "seed_artifact": {"kind": "artifact_dir", "path": str(seed)},
            "teacher_report": {
                "kind": "teacher_cache",
                "path": str(tmp_path / "cache-report" / "metadata.jsonl"),
            },
            "teacher_selection": {
                "kind": "teacher_cache",
                "path": str(tmp_path / "cache-select" / "metadata.jsonl"),
            },
        },
        "steps": [
            {
                "id": "train",
                "op": "train-low-rank",
                "class": "promotable",
                "inputs": {
                    "seed_artifact": "external:seed_artifact",
                    "selection_teacher": "external:teacher_selection",
                    "validation_teacher": "external:teacher_report",
                },
                "params": {"layer": 45, "low_rank": 4},
                "gate": {"profile": "train_sane"},
            },
            {
                "id": "eval_report",
                "op": "eval",
                "class": "verify",
                "inputs": {
                    "artifact": "step:train/artifact",
                    "teacher": "external:teacher_report",
                },
                "params": {"engine": "vq_e1_routed_nax_e8p"},
            },
        ],
    }
    if not with_repair:
        raw["steps"][-1]["gate"] = {
            "profile": "balanced_rc_split",
            "thresholds": {"clean_rows": 4},
        }
    else:
        raw["steps"].append(
            {
                "id": "eval_report_repair",
                "op": "eval-repair",
                "class": "verify",
                "inputs": {
                    "artifact": "step:train/artifact",
                    "teacher": "external:teacher_report",
                    "existing_evidence": "step:eval_report/evidence",
                },
                "params": {"engine": "vq_e1_routed_nax_e8p"},
                "gate": {
                    "profile": "balanced_rc_split",
                    "thresholds": {"clean_rows": 4},
                },
            }
        )
    if with_selection:
        raw["steps"].append(
            {
                "id": "eval_selection",
                "op": "eval",
                "class": "verify",
                "inputs": {
                    "artifact": "step:train/artifact",
                    "teacher": "external:teacher_selection",
                },
                "params": {"engine": "vq_e1_routed_nax_e8p"},
                "gate": {
                    "profile": "balanced_rc_split",
                    "thresholds": {"clean_rows": 4},
                },
            }
        )
    if with_promotion:
        raw["promotion"] = {
            "artifact_step": "train",
            "gates": [
                {
                    "evidence_step": "eval_report_repair"
                    if with_repair
                    else "eval_report",
                    "profile": "community_wow",
                    "thresholds": {"clean_rows": 4},
                }
            ],
        }
        if with_selection:
            raw["promotion"]["gates"].append(
                {
                    "evidence_step": "eval_selection",
                    "profile": "community_wow",
                    "thresholds": {"clean_rows": 4},
                }
            )
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(raw))
    return recipe_path


def _write_sweep_recipe(tmp_path: Path) -> Path:
    imatrix = tmp_path / "imatrix-manifest.json"
    imatrix.write_text(json.dumps({"schema_version": 1}) + "\n")
    baseline = tmp_path / "baseline-artifact"
    baseline.mkdir(exist_ok=True)
    (baseline / "conversion-manifest.json").write_text(json.dumps({"baseline": True}))
    raw = {
        "schema_version": 1,
        "name": "tiny-sweep",
        "external_inputs": {
            "imatrix_manifest": {"kind": "file", "path": str(imatrix)},
            "baseline_artifact": {"kind": "artifact_dir", "path": str(baseline)},
        },
        "steps": [
            {
                "id": "sweep",
                "op": "materialize-sweep",
                "class": "promotable",
                "inputs": {
                    "imatrix_manifest": "external:imatrix_manifest",
                    "baseline_artifact": "external:baseline_artifact",
                },
                "params": {"budget": 2.0, "reround_top_low_experts": 2},
            }
        ],
    }
    recipe_path = tmp_path / "sweep-recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(raw))
    return recipe_path


def _stubbed_plan(
    tmp_path: Path,
    *,
    eval_pageouts: int = 0,
    selection_pageouts: int = 0,
    with_promotion: bool = False,
    with_selection: bool = False,
    with_repair: bool = False,
):
    recipe = load_recipe(
        _write_recipe(
            tmp_path,
            with_promotion=with_promotion,
            with_selection=with_selection,
            with_repair=with_repair,
        )
    )
    plan = plan_recipe(recipe, build_root=tmp_path / "build")
    train = plan.step("train")
    train.argv = [
        sys.executable,
        "-c",
        _TRAIN_STUB,
        str(train.out_dir),
        str(train.evidence_dir / "train_log.jsonl"),
    ]
    eval_step = plan.step("eval_report")
    eval_step.argv = [
        sys.executable,
        "-c",
        _EVAL_STUB_TEMPLATE.format(pageouts=eval_pageouts),
        str(eval_step.evidence_dir / "evidence.jsonl"),
    ]
    if with_repair:
        repair_step = plan.step("eval_report_repair")
        repair_step.argv = [
            sys.executable,
            "-c",
            _REPAIR_STUB,
            str(repair_step.evidence_dir / "evidence.jsonl"),
        ]
    if with_selection:
        selection = plan.step("eval_selection")
        selection.argv = [
            sys.executable,
            "-c",
            _EVAL_STUB_TEMPLATE.format(pageouts=selection_pageouts),
            str(selection.evidence_dir / "evidence.jsonl"),
        ]
    return plan


def test_execute_runs_steps_and_emits_build_steps(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path)
    assert execute_plan(plan) == 0

    train = plan.step("train")
    manifest = json.loads((train.out_dir / "conversion-manifest.json").read_text())
    assert [entry["step_id"] for entry in manifest["build_steps"]] == [
        "seed_step",
        "train",
    ]
    own = manifest["build_steps"][-1]
    assert own["op"] == "train-low-rank"
    assert own["step_key"] == train.step_key
    assert own["resolved_argv"]
    assert own["input_hashes"]["seed_artifact"]
    assert manifest["recovery"] == ["rkd", "sidecar:layer45"]
    assert (train.step_dir / "step-result.json").exists()

    records = Ledger(plan.build_root / "ledger.jsonl").records()
    events = [(record["step_id"], record["event"]) for record in records]
    assert ("train", "started") in events
    assert ("train", "completed") in events
    assert ("eval_report", "completed") in events


def test_glm52_heavy_cli_wrappers_self_manage_the_repository_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_names = (
        "snapshot_dir",
        "prompt_pack_json",
        "non_vq_package_dir",
        "artifact_identities_json",
        "profile_path",
        "ledger_path",
        "checkpoint_dir",
    )
    recipe_path = tmp_path / "teacher-cache-produce-executor.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "glm52-teacher-cache-produce-executor-fixture",
                "description": "fixture",
                "external_inputs": {
                    name: {
                        "kind": (
                            "artifact_dir"
                            if name in {"snapshot_dir", "non_vq_package_dir"}
                            else "file"
                        ),
                        "path": str(tmp_path / name),
                    }
                    for name in input_names
                },
                "steps": [
                    {
                        "id": "teacher_cache",
                        "op": "glm52-teacher-cache-produce",
                        "class": "diagnostic",
                        "inputs": {
                            name: f"external:{name}" for name in input_names
                        },
                    }
                ],
            },
            sort_keys=False,
        )
    )
    plan = plan_recipe(
        load_recipe(recipe_path),
        build_root=tmp_path / "build-executor",
        no_hash=True,
    )
    step = plan.step("teacher_cache")
    step.argv = ["stub-glm52-teacher-cache-producer"]
    lock_path = tmp_path / ".keep-heavy-job.lock"
    monkeypatch.setattr(executor, "HEAVY_JOB_LOCK_PATH", lock_path)
    monkeypatch.delenv("GLM45_DISABLE_HEAVY_LOCK", raising=False)
    child_lock_acquisitions: list[Path] = []

    def _spawn(argv, *, stdout, stderr):
        assert argv == step.argv
        assert stderr is subprocess.STDOUT
        with lock_path.open("a") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                child_lock_acquisitions.append(lock_path)
                step.out_dir.mkdir(parents=True, exist_ok=True)
                (step.out_dir / step.op_def.artifact_manifest_name).write_text(
                    "{}\n"
                )
                step.output_paths["route_trace_root"].mkdir(parents=True)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return subprocess.CompletedProcess(argv, 0)

    assert execute_plan(plan, spawn=_spawn) == 0
    assert child_lock_acquisitions == [lock_path]
    assert step.op_def.self_managed_heavy_lock is True
    assert {
        name
        for name, op_def in REGISTRY.items()
        if op_def.self_managed_heavy_lock
    } == {
        "glm52-teacher-cache-produce",
        "glm52-candidate-full-vocab-eval-produce",
        "glm52-same-machine-benchmark-pair",
    }
    for name, op_def in REGISTRY.items():
        if op_def.produces_artifact and not op_def.self_managed_heavy_lock:
            step.op_def = op_def
            assert executor._uses_heavy_job_lock(step) is True, name


def test_resume_skips_completed_steps(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path)
    assert execute_plan(plan) == 0
    # The train stub exits 3 if its output dir exists, so a re-run would fail
    # unless the executor skips it via the ledger.
    resumed = _stubbed_plan(tmp_path)
    assert execute_plan(resumed) == 0
    records = Ledger(resumed.build_root / "ledger.jsonl").records()
    started = [record for record in records if record["event"] == "started"]
    assert len(started) == 2  # only the first run started anything


def test_noncacheable_live_diagnostic_reruns_after_completion(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path)
    assert execute_plan(plan) == 0

    resumed = _stubbed_plan(tmp_path)
    live_step = resumed.step("eval_report")
    live_step.op_def = replace(live_step.op_def, cacheable=False)
    assert execute_plan(resumed) == 0

    records = Ledger(resumed.build_root / "ledger.jsonl").records()
    eval_starts = [
        record
        for record in records
        if record["event"] == "started" and record["step_id"] == "eval_report"
    ]
    assert len(eval_starts) == 2


def test_noncacheable_live_diagnostic_reruns_cached_descendants(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path, with_repair=True)
    assert execute_plan(plan) == 0

    resumed = _stubbed_plan(tmp_path, with_repair=True)
    live_step = resumed.step("eval_report")
    live_step.op_def = replace(live_step.op_def, cacheable=False)
    assert execute_plan(resumed) == 0

    records = Ledger(resumed.build_root / "ledger.jsonl").records()
    starts_by_step = {
        step_id: sum(
            record["event"] == "started" and record["step_id"] == step_id
            for record in records
        )
        for step_id in ("eval_report", "eval_report_repair")
    }
    assert starts_by_step == {"eval_report": 2, "eval_report_repair": 2}


def test_noncacheable_live_diagnostic_can_rerun_rapidly(tmp_path: Path) -> None:
    for run_index in range(3):
        plan = _stubbed_plan(tmp_path)
        live_step = plan.step("eval_report")
        live_step.op_def = replace(live_step.op_def, cacheable=False)
        assert execute_plan(plan) == 0, run_index

    quarantined = list(
        live_step.step_dir.parent.glob(f"{live_step.step_dir.name}.quarantine-*")
    )
    assert len(quarantined) == 2


def test_resumable_partial_step_keeps_checkpoint_directory(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path)
    eval_step = plan.step("eval_report")
    eval_step.op_def = replace(eval_step.op_def, resume_partial=True)
    eval_step.step_dir.mkdir(parents=True)
    checkpoint = eval_step.step_dir / "completed-group.safetensors"
    checkpoint.write_bytes(b"complete-group")

    assert execute_plan(plan) == 0

    assert checkpoint.read_bytes() == b"complete-group"
    assert not list(
        eval_step.step_dir.parent.glob(f"{eval_step.step_dir.name}.quarantine-*")
    )


def test_strict_resumable_step_owns_partial_file_validation(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path)
    eval_step = plan.step("eval_report")
    eval_step.op_def = replace(
        eval_step.op_def,
        resume_partial=True,
        manages_partial_files=True,
    )
    eval_step.out_dir.mkdir(parents=True)
    unrelated = eval_step.out_dir / "notes.partial-backup"
    unrelated.write_text("caller-owned")

    assert execute_plan(plan) == 0

    assert unrelated.read_text() == "caller-owned"


def test_resumable_artifact_invalidates_stale_completion_markers(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path)
    train = plan.step("train")
    train.op_def = replace(train.op_def, resume_partial=True)
    train.argv = [
        sys.executable,
        "-c",
        _RESUMABLE_TRAIN_STUB,
        str(train.out_dir),
        str(train.evidence_dir / "train_log.jsonl"),
    ]
    train.out_dir.mkdir(parents=True)
    checkpoint = train.out_dir / "layer-00010-gate_proj.safetensors"
    checkpoint.write_bytes(b"complete-group")
    (train.out_dir / "conversion-manifest.json").write_text('{"stale": true}')
    (train.step_dir / "step-result.json").write_text('{"status": "stale"}')

    assert execute_plan(plan) == 0

    assert checkpoint.read_bytes() == b"complete-group"
    manifest = json.loads((train.out_dir / "conversion-manifest.json").read_text())
    assert manifest["fresh"] is True
    step_result = json.loads((train.step_dir / "step-result.json").read_text())
    assert step_result["status"] == "completed"
    records = Ledger(plan.build_root / "ledger.jsonl").records()
    resumed = next(record for record in records if record["event"] == "resuming_partial")
    assert str(train.out_dir / "conversion-manifest.json") in resumed[
        "invalidated_completion_markers"
    ]
    assert str(train.step_dir / "step-result.json") in resumed[
        "invalidated_completion_markers"
    ]


def test_explicit_from_quarantines_resumable_artifact_for_clean_rebuild(
    tmp_path: Path,
) -> None:
    plan = _stubbed_plan(tmp_path)
    train = plan.step("train")
    train.op_def = replace(train.op_def, resume_partial=True)
    train.argv = [
        sys.executable,
        "-c",
        _RESUMABLE_TRAIN_STUB,
        str(train.out_dir),
        str(train.evidence_dir / "train_log.jsonl"),
    ]
    train.out_dir.mkdir(parents=True)
    checkpoint = train.out_dir / "layer-00010-gate_proj.safetensors"
    checkpoint.write_bytes(b"complete-group")

    assert execute_plan(plan, from_step="train") == 0

    assert not checkpoint.exists()
    quarantined = list(
        train.step_dir.parent.glob(f"{train.step_dir.name}.quarantine-*")
    )
    assert len(quarantined) == 1
    assert (
        quarantined[0] / "out" / "layer-00010-gate_proj.safetensors"
    ).read_bytes() == b"complete-group"


def test_failed_resumable_step_reuses_checkpoint_on_normal_rerun(
    tmp_path: Path,
) -> None:
    first = _stubbed_plan(tmp_path)
    first_eval = first.step("eval_report")
    first_eval.op_def = replace(first_eval.op_def, resume_partial=True)
    checkpoint = first_eval.step_dir / "completed-group.safetensors"
    first_eval.argv = [
        sys.executable,
        "-c",
        _FAIL_WITH_CHECKPOINT_STUB,
        str(checkpoint),
    ]

    assert execute_plan(first) == 1
    assert checkpoint.read_bytes() == b"complete-group"

    resumed = _stubbed_plan(tmp_path)
    resumed_eval = resumed.step("eval_report")
    resumed_eval.op_def = replace(resumed_eval.op_def, resume_partial=True)
    assert execute_plan(resumed) == 0

    assert checkpoint.read_bytes() == b"complete-group"
    assert not list(
        resumed_eval.step_dir.parent.glob(
            f"{resumed_eval.step_dir.name}.quarantine-*"
        )
    )
    records = Ledger(resumed.build_root / "ledger.jsonl").records()
    assert any(
        record["event"] == "resuming_partial"
        and record["step_id"] == "eval_report"
        for record in records
    )


def test_noncacheable_ancestor_reruns_resumable_descendant_in_place(
    tmp_path: Path,
) -> None:
    first = _stubbed_plan(tmp_path)
    first.step("train").op_def = replace(
        first.step("train").op_def,
        cacheable=False,
    )
    first_eval = first.step("eval_report")
    first_eval.op_def = replace(first_eval.op_def, resume_partial=True)
    assert execute_plan(first) == 0
    checkpoint = first_eval.step_dir / "completed-group.safetensors"
    checkpoint.write_bytes(b"complete-group")

    resumed = _stubbed_plan(tmp_path)
    resumed.step("train").op_def = replace(
        resumed.step("train").op_def,
        cacheable=False,
    )
    resumed_eval = resumed.step("eval_report")
    resumed_eval.op_def = replace(resumed_eval.op_def, resume_partial=True)
    assert execute_plan(resumed) == 0

    assert checkpoint.read_bytes() == b"complete-group"
    assert not list(
        resumed_eval.step_dir.parent.glob(
            f"{resumed_eval.step_dir.name}.quarantine-*"
        )
    )
    records = Ledger(resumed.build_root / "ledger.jsonl").records()
    eval_starts = [
        record
        for record in records
        if record["event"] == "started" and record["step_id"] == "eval_report"
    ]
    assert len(eval_starts) == 2


def test_partial_output_without_ledger_entry_is_quarantined(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path)
    train = plan.step("train")
    (train.step_dir / "out").mkdir(parents=True)
    (train.step_dir / "out" / "junk.bin").write_bytes(b"partial")

    assert execute_plan(plan) == 0
    quarantined = list(train.step_dir.parent.glob(f"{train.step_dir.name}.quarantine-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "out" / "junk.bin").exists()
    assert (train.out_dir / "conversion-manifest.json").exists()


def test_zero_exit_missing_declared_output_fails_and_stops_downstream(
    tmp_path: Path,
) -> None:
    plan = _stubbed_plan(tmp_path, with_repair=True)
    eval_step = plan.step("eval_report")
    eval_step.argv = [sys.executable, "-c", "raise SystemExit(0)"]

    assert execute_plan(plan) == 1

    records = Ledger(plan.build_root / "ledger.jsonl").records()
    terminal = Ledger(plan.build_root / "ledger.jsonl").latest_terminal(
        eval_step.step_key
    )
    assert terminal is not None and terminal["event"] == "failed"
    assert terminal["missing_outputs"] == ["evidence"]
    assert not any(
        record["event"] == "started" and record["step_id"] == "eval_report_repair"
        for record in records
    )


def test_materialize_sweep_requested_reround_requires_actions(
    tmp_path: Path,
) -> None:
    recipe = load_recipe(_write_sweep_recipe(tmp_path))
    plan = plan_recipe(recipe, build_root=tmp_path / "build")
    sweep = plan.step("sweep")
    candidate_dir = sweep.output_paths["artifact_bpw2p0"]
    sweep.argv = [sys.executable, "-c", _SWEEP_ZERO_REROUND_STUB, str(candidate_dir)]

    assert execute_plan(plan) == 1

    terminal = Ledger(plan.build_root / "ledger.jsonl").latest_terminal(sweep.step_key)
    assert terminal is not None and terminal["event"] == "failed"
    assert terminal["missing_outputs"] == ["artifact_bpw2p0:reround_action_count"]


def test_gate_failed_recorded_and_not_reused(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path, eval_pageouts=7)
    assert execute_plan(plan) == 1
    ledger = Ledger(plan.build_root / "ledger.jsonl")
    eval_step = plan.step("eval_report")
    terminal = ledger.latest_terminal(eval_step.step_key)
    assert terminal is not None and terminal["event"] == "gate_failed"
    # Dirty evidence is preserved as the rejection record.
    assert (eval_step.evidence_dir / "evidence.jsonl").exists()

    # A re-run with the same key refuses to silently retry the gate-failed step.
    rerun = _stubbed_plan(tmp_path, eval_pageouts=7)
    assert execute_plan(rerun) == 1
    records = ledger.records()
    started_eval = [
        record
        for record in records
        if record["event"] == "started" and record["step_id"] == "eval_report"
    ]
    assert len(started_eval) == 1


def test_declared_blocked_returncode_still_emits_structured_gate_result(
    tmp_path: Path,
) -> None:
    plan = _stubbed_plan(tmp_path, eval_pageouts=7)
    eval_step = plan.step("eval_report")
    eval_step.op_def = replace(
        eval_step.op_def,
        accepted_returncodes=frozenset({0, 2}),
    )
    eval_step.argv[2] += "\nraise SystemExit(2)\n"

    assert execute_plan(plan) == 1

    terminal = Ledger(plan.build_root / "ledger.jsonl").latest_terminal(
        eval_step.step_key
    )
    assert terminal is not None and terminal["event"] == "gate_failed"
    assert terminal["subprocess_returncode"] == 2
    step_result = json.loads((eval_step.step_dir / "step-result.json").read_text())
    assert step_result["status"] == "gate_failed"
    assert step_result["subprocess_returncode"] == 2
    assert step_result["input_hashes"] == eval_step.input_hashes


def test_declared_blocked_returncode_cannot_complete_with_passing_gate(
    tmp_path: Path,
) -> None:
    plan = _stubbed_plan(tmp_path, eval_pageouts=0)
    eval_step = plan.step("eval_report")
    eval_step.op_def = replace(
        eval_step.op_def,
        accepted_returncodes=frozenset({0, 2}),
    )
    eval_step.argv[2] += "\nraise SystemExit(2)\n"

    assert execute_plan(plan) == 1

    terminal = Ledger(plan.build_root / "ledger.jsonl").latest_terminal(
        eval_step.step_key
    )
    assert terminal is not None and terminal["event"] == "failed"
    assert terminal["returncode"] == 2
    assert terminal["reason"] == "accepted_nonzero_returncode_with_passing_gate"
    step_result = json.loads((eval_step.step_dir / "step-result.json").read_text())
    assert step_result["status"] == "failed"
    assert step_result["subprocess_returncode"] == 2
    assert step_result["failure_reason"] == terminal["reason"]
    assert step_result["gate_result"]["passed"] is True


def test_continue_on_gate_fail_collects_later_evidence_without_passing_promotion(
    tmp_path: Path,
) -> None:
    plan = _stubbed_plan(
        tmp_path,
        eval_pageouts=7,
        with_promotion=True,
        with_selection=True,
    )

    assert execute_plan(plan, continue_on_gate_fail=True) == 1
    ledger = Ledger(plan.build_root / "ledger.jsonl")
    report_terminal = ledger.latest_terminal(plan.step("eval_report").step_key)
    selection_terminal = ledger.latest_terminal(plan.step("eval_selection").step_key)
    assert report_terminal is not None and report_terminal["event"] == "gate_failed"
    assert selection_terminal is not None and selection_terminal["event"] == "completed"

    promotion = json.loads((plan.build_root / "promotion-result.json").read_text())
    assert promotion["status"] == "balanced_rc_fail"
    assert promotion["gates"]["eval_report"]["passed"] is False
    assert promotion["gates"]["eval_selection"]["passed"] is True

    rerun = _stubbed_plan(
        tmp_path,
        eval_pageouts=7,
        with_promotion=True,
        with_selection=True,
    )
    assert execute_plan(rerun, continue_on_gate_fail=True) == 1
    records = ledger.records()
    started_report = [
        record
        for record in records
        if record["event"] == "started" and record["step_id"] == "eval_report"
    ]
    assert len(started_report) == 1


def test_regate_reevaluates_without_spawning(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path, eval_pageouts=7)
    assert execute_plan(plan) == 1
    eval_step = plan.step("eval_report")

    # Clean the recorded evidence in place (simulates a threshold/evidence
    # correction) and regate: no subprocess may run.
    evidence = eval_step.evidence_dir / "evidence.jsonl"
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    for row in rows:
        row["pageouts_delta"] = 0
    evidence.write_text("".join(json.dumps(row) + "\n" for row in rows))

    regate_plan = _stubbed_plan(tmp_path, eval_pageouts=7)

    def _no_spawn(*args, **kwargs):
        raise AssertionError("--regate must not spawn subprocesses")

    assert execute_plan(regate_plan, regate=True, spawn=_no_spawn) == 0
    ledger = Ledger(regate_plan.build_root / "ledger.jsonl")
    terminal = ledger.latest_terminal(regate_plan.step("eval_report").step_key)
    assert terminal is not None and terminal["event"] == "completed"
    assert terminal.get("regated") is True


def test_regate_fails_when_gated_steps_have_no_recorded_run(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path)

    def _no_spawn(*args, **kwargs):
        raise AssertionError("--regate must not spawn subprocesses")

    assert execute_plan(plan, regate=True, spawn=_no_spawn) == 1
    assert Ledger(plan.build_root / "ledger.jsonl").records() == []


def test_regate_cannot_accept_passing_evidence_from_failed_subprocess(
    tmp_path: Path,
) -> None:
    plan = _stubbed_plan(tmp_path, eval_pageouts=7)
    eval_step = plan.step("eval_report")
    eval_step.op_def = replace(eval_step.op_def, regate_can_complete=False)
    eval_step.argv[2] += "\nraise SystemExit(7)\n"
    assert execute_plan(plan) == 1

    evidence = eval_step.evidence_dir / "evidence.jsonl"
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    for row in rows:
        row["pageouts_delta"] = 0
    evidence.write_text("".join(json.dumps(row) + "\n" for row in rows))

    regate_plan = _stubbed_plan(tmp_path, eval_pageouts=7)
    regate_plan.step("eval_report").op_def = replace(
        regate_plan.step("eval_report").op_def,
        regate_can_complete=False,
    )
    assert execute_plan(regate_plan, regate=True) == 1
    terminal = Ledger(regate_plan.build_root / "ledger.jsonl").latest_terminal(
        regate_plan.step("eval_report").step_key
    )
    assert terminal is not None and terminal["event"] == "failed"
    assert not any(
        record["event"] == "regated"
        and record["step_id"] == "eval_report"
        for record in Ledger(regate_plan.build_root / "ledger.jsonl").records()
    )


def test_regate_requires_rerun_after_nonzero_gate_failure(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path, eval_pageouts=7)
    eval_step = plan.step("eval_report")
    eval_step.op_def = replace(
        eval_step.op_def,
        accepted_returncodes=frozenset({0, 2}),
    )
    eval_step.argv[2] += "\nraise SystemExit(2)\n"
    assert execute_plan(plan) == 1

    evidence = eval_step.evidence_dir / "evidence.jsonl"
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    for row in rows:
        row["pageouts_delta"] = 0
    evidence.write_text("".join(json.dumps(row) + "\n" for row in rows))

    regate_plan = _stubbed_plan(tmp_path, eval_pageouts=7)
    regate_plan.step("eval_report").op_def = replace(
        regate_plan.step("eval_report").op_def,
        accepted_returncodes=frozenset({0, 2}),
    )
    assert execute_plan(regate_plan, regate=True) == 1
    terminal = Ledger(regate_plan.build_root / "ledger.jsonl").latest_terminal(
        regate_plan.step("eval_report").step_key
    )
    assert terminal is not None and terminal["event"] == "gate_failed"
    assert terminal["subprocess_returncode"] == 2
    assert not any(
        record["event"] == "completed"
        and record.get("regated") is True
        and record["step_id"] == "eval_report"
        for record in Ledger(regate_plan.build_root / "ledger.jsonl").records()
    )


def test_regate_records_new_failure_for_previously_completed_gate(
    tmp_path: Path,
) -> None:
    plan = _stubbed_plan(tmp_path, eval_pageouts=0)
    assert execute_plan(plan) == 0

    evidence = plan.step("eval_report").evidence_dir / "evidence.jsonl"
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    for row in rows:
        row["pageouts_delta"] = 7
    evidence.write_text("".join(json.dumps(row) + "\n" for row in rows))

    regate_plan = _stubbed_plan(tmp_path, eval_pageouts=0)
    assert execute_plan(regate_plan, regate=True) == 1
    terminal = Ledger(regate_plan.build_root / "ledger.jsonl").latest_terminal(
        regate_plan.step("eval_report").step_key
    )
    assert terminal is not None and terminal["event"] == "gate_failed"
    assert terminal["subprocess_returncode"] == 0
    assert terminal["regated"] is True
    step_result = json.loads(
        (regate_plan.step("eval_report").step_dir / "step-result.json").read_text()
    )
    assert step_result["status"] == "gate_failed"
    assert step_result["regated"] is True


def test_regate_cannot_promote_composite_gate_that_requires_rerun(
    tmp_path: Path,
) -> None:
    plan = _stubbed_plan(tmp_path, eval_pageouts=7)
    plan.step("eval_report").op_def = replace(
        plan.step("eval_report").op_def,
        regate_can_complete=False,
    )
    assert execute_plan(plan) == 1

    evidence = plan.step("eval_report").evidence_dir / "evidence.jsonl"
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    for row in rows:
        row["pageouts_delta"] = 0
    evidence.write_text("".join(json.dumps(row) + "\n" for row in rows))

    regate_plan = _stubbed_plan(tmp_path, eval_pageouts=7)
    regate_plan.step("eval_report").op_def = replace(
        regate_plan.step("eval_report").op_def,
        regate_can_complete=False,
    )
    assert execute_plan(regate_plan, regate=True) == 1
    terminal = Ledger(regate_plan.build_root / "ledger.jsonl").latest_terminal(
        regate_plan.step("eval_report").step_key
    )
    assert terminal is not None and terminal["event"] == "gate_failed"


def test_from_step_forces_rerun_of_descendants(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path)
    assert execute_plan(plan) == 0

    rerun = _stubbed_plan(tmp_path)
    # Forcing from eval_report must not touch train (its stub would exit 3).
    assert execute_plan(rerun, from_step="eval_report") == 0
    records = Ledger(rerun.build_root / "ledger.jsonl").records()
    started = [
        (record["step_id"], record["event"])
        for record in records
        if record["event"] == "started"
    ]
    assert started.count(("eval_report", "started")) == 2
    assert started.count(("train", "started")) == 1


def test_from_repair_step_preserves_upstream_raw_eval_evidence(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path, eval_pageouts=7, with_repair=True)
    assert execute_plan(plan) == 0

    rerun = _stubbed_plan(tmp_path, eval_pageouts=7, with_repair=True)
    raw_eval = rerun.step("eval_report")
    raw_eval.argv = [sys.executable, "-c", "raise SystemExit(3)"]

    assert execute_plan(rerun, from_step="eval_report_repair") == 0
    records = Ledger(rerun.build_root / "ledger.jsonl").records()
    started = [
        (record["step_id"], record["event"])
        for record in records
        if record["event"] == "started"
    ]
    assert started.count(("eval_report", "started")) == 1
    assert started.count(("eval_report_repair", "started")) == 2


def test_promotion_result_written(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path, with_promotion=True)
    assert execute_plan(plan) == 0
    payload = json.loads((plan.build_root / "promotion-result.json").read_text())
    assert payload["artifact_step"] == "train"
    assert payload["status"] == "balanced_rc_pass_community_wow_pass"
    assert payload["gates"]["eval_report"]["passed"] is True
