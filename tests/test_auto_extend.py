from __future__ import annotations

import sys
from pathlib import Path

import yaml

from keep.build.executor import execute_plan
from keep.build.plan_next import (
    auto_extend,
    load_merged_recipe,
    overlay_path_for,
)
from keep.build.recipe import load_recipe
from keep.build.runner import plan_recipe

from tests.test_build_runner_resume import _TRAIN_STUB, _write_recipe
from tests.test_plan_next import _DOMAIN_EVAL_STUB

_DIRTY_EVAL_STUB = _DOMAIN_EVAL_STUB.replace('"pageouts_delta": 0', '"pageouts_delta": 9')


def _base_recipe_with_selection(tmp_path: Path) -> Path:
    recipe_path = _write_recipe(tmp_path)
    raw = yaml.safe_load(recipe_path.read_text())
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
            "gate": {"profile": "balanced_rc_split", "thresholds": {"clean_rows": 10}},
        }
    )
    raw["steps"][1]["gate"]["thresholds"]["clean_rows"] = 10
    recipe_path.write_text(yaml.safe_dump(raw))
    return recipe_path


def _stub_argv(plan, *, eval_stub: str) -> None:
    for planned in plan.steps:
        if planned.spec.op == "train-low-rank":
            planned.argv = [
                sys.executable,
                "-c",
                _TRAIN_STUB,
                str(planned.out_dir),
                str(planned.evidence_dir / "train_log.jsonl"),
            ]
        elif planned.spec.op == "eval":
            planned.argv = [
                sys.executable,
                "-c",
                eval_stub,
                str(planned.evidence_dir / "evidence.jsonl"),
            ]


def _run_base_build(recipe_path: Path, tmp_path: Path) -> None:
    plan = plan_recipe(load_recipe(recipe_path), build_root=tmp_path / "build")
    _stub_argv(plan, eval_stub=_DOMAIN_EVAL_STUB)
    assert execute_plan(plan) == 0


def test_auto_extend_accepts_passing_proposal(tmp_path: Path) -> None:
    recipe_path = _base_recipe_with_selection(tmp_path)
    _run_base_build(recipe_path, tmp_path)

    accepted = auto_extend(
        recipe_path,
        iterations=1,
        build_root=tmp_path / "build",
        prepare_plan=lambda plan: _stub_argv(plan, eval_stub=_DOMAIN_EVAL_STUB),
    )
    assert accepted == 1

    overlay = yaml.safe_load(overlay_path_for(recipe_path).read_text())
    step_ids = [step["id"] for step in overlay["steps"]]
    assert "train_plan_next_1" in step_ids
    assert overlay["rejected"] == []
    # The human-authored recipe file is never machine-edited.
    assert "train_plan_next_1" not in recipe_path.read_text()

    # The merged recipe parses and the accepted step chains off the base artifact.
    merged = load_merged_recipe(recipe_path)
    proposed = merged.step("train_plan_next_1")
    assert proposed.inputs["seed_artifact"].raw == "step:train/artifact"


def test_auto_extend_rejects_gate_failing_proposal(tmp_path: Path) -> None:
    recipe_path = _base_recipe_with_selection(tmp_path)
    _run_base_build(recipe_path, tmp_path)

    def _prepare(plan) -> None:
        # New (proposed) eval steps produce memory-dirty evidence; the
        # already-completed base steps are skipped via the ledger anyway.
        _stub_argv(plan, eval_stub=_DIRTY_EVAL_STUB)

    accepted = auto_extend(
        recipe_path,
        iterations=1,
        build_root=tmp_path / "build",
        prepare_plan=_prepare,
    )
    assert accepted == 0

    overlay = yaml.safe_load(overlay_path_for(recipe_path).read_text())
    assert overlay["steps"] == []
    assert len(overlay["rejected"]) == 1
    rejected_ids = [step["id"] for step in overlay["rejected"][0]["steps"]]
    assert "train_plan_next_1" in rejected_ids
    # The merged recipe no longer contains the rejected proposal.
    merged = load_merged_recipe(recipe_path)
    assert all(spec.id != "train_plan_next_1" for spec in merged.steps)
