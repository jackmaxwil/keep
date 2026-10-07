from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from keep.build.cli import _format_plan_next_row_summary
from keep.build.executor import execute_plan
from keep.build.plan_next import PlanNextError, propose_next_step
from keep.build.recipe import load_recipe, validate_recipe
from keep.build.ops import REGISTRY
from keep.build.runner import plan_recipe
from keep.quality.plan_next import (
    QUALITY_PLAN_DOMAIN_QUOTAS,
    _quality_plan_candidates,
    _select_quality_plan_rows,
)

from tests.test_build_runner_resume import _TRAIN_STUB, _write_recipe

# Eval stub emitting rows across route/math/instruction domains with a
# spread of top1/kld so quality-plan selection has real signal.
_DOMAIN_EVAL_STUB = """
import json, sys
rows = []
specs = [
    ("route", 0.40), ("route", 0.45), ("route", 0.50), ("route", 0.55),
    ("math", 0.42), ("math", 0.48), ("math", 0.52),
    ("instruction", 0.60), ("instruction", 0.95), ("route", 0.97),
]
for i, (domain, top1) in enumerate(specs):
    rows.append({"prompt_id": "report_%s_%03d" % (domain, i), "row_index": i,
        "top1_agreement": top1, "mean_kld": 1.0 - top1, "ppl_ratio": 1.0,
        "nll_delta": 0.01, "token_klds": [1.0 - top1, 0.1],
        "pageouts_delta": 0, "swapouts_delta": 0})
with open(sys.argv[1], "a") as fh:
    for row in rows:
        fh.write(json.dumps(row) + "\\n")
"""

_NON_EXPERT_GAP_STUB = """
import json, sys
with open(sys.argv[1], "w") as fh:
    json.dump({
        "record_type": "glm45_air_top1_gap_analysis",
        "recommended_op": "bump-non-expert-precision",
        "recommended_surfaces": "embed_tokens,lm_head",
        "recommended_dtype": "bf16",
        "reason": "top1 gap attributed to non-expert surfaces"
    }, fh)
"""

_UNSAFE_NON_EXPERT_GAP_STUB = """
import json, sys
with open(sys.argv[1], "w") as fh:
    json.dump({
        "record_type": "glm45_air_top1_gap_analysis",
        "recommended_op": "bump-non-expert-precision",
        "recommended_surfaces": "attention,lm_head",
        "recommended_dtype": "bf16",
        "reason": "resident bytes are dominated by attention"
    }, fh)
"""

_TAIL_CLEANUP_GAP_STUB = """
import json, sys
with open(sys.argv[1], "w") as fh:
    json.dump({
        "record_type": "glm45_air_tail_cleanup_target_selection",
        "decision": "tail_cleanup_report_train_packet_ready",
        "recommended_training": {
            "train_cache": "validation",
            "train_row_indices": [120, 121, 122, 123],
            "max_positions": 1,
            "aux_loss_position_indices": [0],
            "notes": "Use low LR/NLL-heavy selected-layer training, not another final-logit bias lift."
        },
        "validation_focus_rows": [
            {"split": "selection", "row_index": 120},
            {"split": "holdout", "row_index": 120}
        ]
    }, fh)
"""


def _sparse_residual_plan_payload() -> dict:
    return {
        "summary": {
            "route_source_sparse_residual_plan_rankings": [
                {
                    "layer": 41,
                    "target_key": "report_route_000:0",
                    "token_index": 0,
                    "route_rank": 0,
                    "expert": 93,
                    "projection": "down_proj",
                    "desired_weighted_correction": 1.1,
                    "residual_value_norm": 1.0,
                    "rank": 1,
                }
            ],
            "records": [
                {
                    "layer": 41,
                    "probe_target": {
                        "key": "report_route_000:0",
                        "prompt_id": "report_route_000",
                        "position": 0,
                    },
                    "route_source_sparse_residual_plans": [
                        {
                            "token_index": 0,
                            "route_rank": 0,
                            "expert": 93,
                            "router_score": 0.55,
                            "projection": "down_proj",
                            "source_metric": "source_weighted_route_contribution",
                            "rows": [
                                {
                                    "output_index": 11,
                                    "desired_weighted_correction": 1.1,
                                    "desired_unweighted_correction": 2.0,
                                    "input_norm_sq": 4.0,
                                    "residual_value_norm": 1.0,
                                    "residual_values": [0.5, 0.5],
                                    "reconstructed_unweighted_correction": 2.0,
                                    "reconstruction_abs_error": 0.0,
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    }


def _executed_plan(tmp_path: Path):
    recipe_path = _write_recipe(tmp_path)
    # Extend the fixture with a selection eval so plan-next sees both splits.
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

    recipe = load_recipe(recipe_path)
    plan = plan_recipe(recipe, build_root=tmp_path / "build")
    train = plan.step("train")
    train.argv = [
        sys.executable,
        "-c",
        _TRAIN_STUB,
        str(train.out_dir),
        str(train.evidence_dir / "train_log.jsonl"),
    ]
    for step_id in ("eval_report", "eval_selection"):
        step = plan.step(step_id)
        step.argv = [
            sys.executable,
            "-c",
            _DOMAIN_EVAL_STUB,
            str(step.evidence_dir / "evidence.jsonl"),
        ]
    assert execute_plan(plan) == 0
    return recipe_path, recipe


def test_quality_plan_selection_respects_quotas() -> None:
    focus_rows = [
        {"prompt_id": f"report_route_{i}", "row_index": i, "domain": "route",
         "top1_agreement": 0.4, "mean_kld": 0.6, "max_token_kld": 2.0}
        for i in range(6)
    ] + [
        {"prompt_id": f"report_math_{i}", "row_index": 10 + i, "domain": "math",
         "top1_agreement": 0.5, "mean_kld": 0.5, "max_token_kld": 1.5}
        for i in range(4)
    ] + [
        {"prompt_id": f"report_instruction_{i}", "row_index": 20 + i,
         "domain": "instruction", "top1_agreement": 0.6, "mean_kld": 0.4,
         "max_token_kld": 1.0}
        for i in range(2)
    ]
    report = {
        "eval_splits": {
            "report": {"quality_focus": {"lowest_top1_rows": focus_rows}},
        }
    }
    candidates = _quality_plan_candidates(
        report, split_names=("report",), allowed_domains=set(QUALITY_PLAN_DOMAIN_QUOTAS)
    )
    rows = _select_quality_plan_rows(candidates, quotas=QUALITY_PLAN_DOMAIN_QUOTAS)
    assert len(rows) == sum(QUALITY_PLAN_DOMAIN_QUOTAS.values())
    by_domain = {domain: 0 for domain in QUALITY_PLAN_DOMAIN_QUOTAS}
    for row in rows:
        by_domain[row["domain"]] += 1
    assert by_domain == QUALITY_PLAN_DOMAIN_QUOTAS


def test_propose_next_step_from_recorded_evidence(tmp_path: Path) -> None:
    recipe_path, recipe = _executed_plan(tmp_path)
    proposal = propose_next_step(recipe, build_root=tmp_path / "build")

    assert proposal.base_step_id == "train"
    train_step = proposal.train_step
    assert train_step.op == "train-low-rank"
    assert train_step.step_class == "promotable"
    assert train_step.inputs["seed_artifact"].raw == "step:train/artifact"
    # Inherits the accepted hyperparams, re-targets the rows.
    assert train_step.params["layer"] == 45
    assert train_step.params["low_rank"] == 4
    rows = train_step.params["train_row_indices"]
    assert rows and all(isinstance(index, int) for index in rows)
    # Quota-shaped selection: route rows dominate.
    domains = [row["domain"] for row in proposal.train_rows]
    assert domains.count("route") >= domains.count("instruction")

    # Verify steps are cloned onto the proposed artifact.
    cloned_ids = {spec.id for spec in proposal.verify_steps}
    assert any(step_id.startswith("eval_report_") for step_id in cloned_ids)
    assert any(step_id.startswith("eval_selection_") for step_id in cloned_ids)

    # The rendered argv comes from the same op the runner executes.
    argv = proposal.resolved_argv
    assert argv[3] == "benchmarks/finetune_glm45_air_vq_continuous.py"
    row_at = argv.index("--train-row-indices")
    assert argv[row_at + 1] == ",".join(str(index) for index in rows)

    # The YAML fragment appended to the recipe validates cleanly.
    raw = yaml.safe_load(recipe_path.read_text())
    raw["steps"].extend(yaml.safe_load(proposal.yaml_fragment()))
    extended_path = tmp_path / "extended.yaml"
    extended_path.write_text(yaml.safe_dump(raw))
    extended = load_recipe(extended_path)
    assert validate_recipe(extended, REGISTRY) == []
    # Proposing never mutates the source recipe.
    assert yaml.safe_load(recipe_path.read_text())["steps"][0]["id"] == "train"


def test_propose_next_step_rewires_cloned_eval_repair_to_cloned_raw_eval(
    tmp_path: Path,
) -> None:
    recipe_path, recipe = _executed_plan(tmp_path)
    raw = yaml.safe_load(recipe_path.read_text())
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
            "gate": {"profile": "balanced_rc_split"},
        }
    )
    recipe_path.write_text(yaml.safe_dump(raw))
    extended = load_recipe(recipe_path)

    proposal = propose_next_step(extended, build_root=tmp_path / "build")

    repair_steps = [
        step for step in proposal.verify_steps if step.id.startswith("eval_report_repair_")
    ]
    assert len(repair_steps) == 1
    repair = repair_steps[0]
    assert repair.inputs["artifact"].raw == f"step:{proposal.train_step.id}/artifact"
    assert (
        repair.inputs["existing_evidence"].raw
        == f"step:eval_report_{proposal.train_step.id}/evidence"
    )


def test_propose_next_step_prefers_non_expert_precision_gap_evidence(tmp_path: Path) -> None:
    recipe_path, recipe = _executed_plan(tmp_path)
    raw = yaml.safe_load(recipe_path.read_text())
    raw["steps"].append(
        {
            "id": "top1_gap",
            "op": "top1-gap",
            "class": "diagnostic",
            "inputs": {"evidence": "step:eval_report/evidence"},
        }
    )
    recipe_path.write_text(yaml.safe_dump(raw))
    extended = load_recipe(recipe_path)
    plan = plan_recipe(extended, build_root=tmp_path / "build")
    gap = plan.step("top1_gap")
    gap.argv = [
        sys.executable,
        "-c",
        _NON_EXPERT_GAP_STUB,
        str(gap.evidence_dir / "gap.json"),
    ]
    assert execute_plan(plan) == 0

    proposal = propose_next_step(extended, build_root=tmp_path / "build")

    assert proposal.train_step.op == "bump-non-expert-precision"
    assert proposal.train_step.inputs["seed_artifact"].raw == "step:train/artifact"
    assert proposal.train_step.params["surfaces"] == ["embed_tokens", "lm_head"]
    assert proposal.train_step.params["dtype"] == "bf16"
    assert proposal.resolved_argv[3] == "benchmarks/materialize_glm45_air_non_expert_precision.py"


def test_propose_next_step_rejects_unsafe_non_expert_precision_auto_surfaces(
    tmp_path: Path,
) -> None:
    recipe_path, _recipe = _executed_plan(tmp_path)
    raw = yaml.safe_load(recipe_path.read_text())
    raw["steps"].append(
        {
            "id": "top1_gap",
            "op": "top1-gap",
            "class": "diagnostic",
            "inputs": {"evidence": "step:eval_report/evidence"},
        }
    )
    recipe_path.write_text(yaml.safe_dump(raw))
    extended = load_recipe(recipe_path)
    plan = plan_recipe(extended, build_root=tmp_path / "build")
    gap = plan.step("top1_gap")
    gap.argv = [
        sys.executable,
        "-c",
        _UNSAFE_NON_EXPERT_GAP_STUB,
        str(gap.evidence_dir / "gap.json"),
    ]
    assert execute_plan(plan) == 0

    with pytest.raises(PlanNextError, match="non-auto-safe.*attention"):
        propose_next_step(extended, build_root=tmp_path / "build")


def test_propose_next_step_uses_tail_cleanup_target_selection(tmp_path: Path) -> None:
    recipe_path, _recipe = _executed_plan(tmp_path)
    raw = yaml.safe_load(recipe_path.read_text())
    raw["steps"].append(
        {
            "id": "top1_gap",
            "op": "top1-gap",
            "class": "diagnostic",
            "inputs": {"evidence": "step:eval_report/evidence"},
        }
    )
    recipe_path.write_text(yaml.safe_dump(raw))
    extended = load_recipe(recipe_path)
    plan = plan_recipe(extended, build_root=tmp_path / "build")
    gap = plan.step("top1_gap")
    gap.argv = [
        sys.executable,
        "-c",
        _TAIL_CLEANUP_GAP_STUB,
        str(gap.evidence_dir / "gap.json"),
    ]
    assert execute_plan(plan) == 0

    proposal = propose_next_step(extended, build_root=tmp_path / "build")

    assert proposal.train_step.op == "train-low-rank"
    assert proposal.train_step.inputs["seed_artifact"].raw == "step:train/artifact"
    assert proposal.train_step.params["layer"] == 41
    assert proposal.train_step.params["projections"] == ["gate_proj", "up_proj", "down_proj"]
    assert proposal.train_step.params["trainable"] == "low_rank_residual"
    assert proposal.train_step.params["low_rank"] == 4
    assert proposal.train_step.params["train_cache"] == "validation"
    assert proposal.train_step.params["train_row_indices"] == [120, 121, 122, 123]
    assert proposal.train_step.params["max_positions"] == 1
    assert proposal.train_step.params["aux_loss_position_indices"] == [0]
    assert proposal.train_step.params["loss_scope"] == "selected_layer"
    assert "logit" not in proposal.train_step.op


def test_propose_next_step_uses_tail_cleanup_from_repaired_eval_evidence(
    tmp_path: Path,
) -> None:
    recipe_path, _recipe = _executed_plan(tmp_path)
    raw = yaml.safe_load(recipe_path.read_text())
    raw["steps"].extend(
        [
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
            },
            {
                "id": "top1_gap_tail_cleanup",
                "op": "top1-gap",
                "class": "diagnostic",
                "inputs": {"evidence": "step:eval_report_repair/evidence"},
            },
        ]
    )
    recipe_path.write_text(yaml.safe_dump(raw))
    extended = load_recipe(recipe_path)
    plan = plan_recipe(extended, build_root=tmp_path / "build")
    gap = plan.step("top1_gap_tail_cleanup")
    gap.argv = [
        sys.executable,
        "-c",
        _TAIL_CLEANUP_GAP_STUB,
        str(gap.evidence_dir / "gap.json"),
    ]
    assert execute_plan(plan) == 0

    proposal = propose_next_step(extended, build_root=tmp_path / "build")

    assert proposal.train_step.id == "train_tail_cleanup_1"
    assert proposal.train_step.op == "train-low-rank"
    assert proposal.train_step.params["train_row_indices"] == [120, 121, 122, 123]


def test_propose_next_step_targets_route_gap_with_router_kd_input(tmp_path: Path) -> None:
    recipe_path, _recipe = _executed_plan(tmp_path)
    disagreement = tmp_path / "route-disagreement.json"
    disagreement.write_text(
        json.dumps({"record_type": "air_route_trace_disagreement_summary", "layers": {"36": {}}}),
        encoding="utf-8",
    )
    raw = yaml.safe_load(recipe_path.read_text())
    raw["external_inputs"]["route_disagreement"] = {
        "kind": "file",
        "path": str(disagreement),
    }
    recipe_path.write_text(yaml.safe_dump(raw))
    extended = load_recipe(recipe_path)

    proposal = propose_next_step(extended, build_root=tmp_path / "build")

    assert proposal.train_step.op == "train-router-kd"
    assert proposal.train_step.inputs["seed_artifact"].raw == "step:train/artifact"
    assert proposal.train_step.inputs["disagreement"].raw == "external:route_disagreement"
    assert proposal.train_step.params["scale"] == 1
    assert proposal.train_step.params["max_abs_delta"] == 0.125
    assert proposal.train_step.params["num_experts"] == 128
    assert proposal.resolved_argv[3] == "benchmarks/materialize_glm45_air_router_correction.py"


def test_propose_next_step_can_plan_from_focused_diagnostic_eval(
    tmp_path: Path,
) -> None:
    recipe_path, _recipe = _executed_plan(tmp_path)
    raw = yaml.safe_load(recipe_path.read_text())
    raw["external_inputs"]["route_disagreement"] = {
        "kind": "file",
        "path": str(tmp_path / "route-disagreement.json"),
    }
    Path(raw["external_inputs"]["route_disagreement"]["path"]).write_text(
        json.dumps({"record_type": "air_route_trace_disagreement_summary", "layers": {"36": {}}}),
        encoding="utf-8",
    )
    raw["steps"].extend(
        [
            {
                "id": "train_tail_cleanup_1",
                "op": "train-low-rank",
                "class": "promotable",
                "inputs": {
                    "seed_artifact": "step:train/artifact",
                    "selection_teacher": "external:teacher_selection",
                    "validation_teacher": "external:teacher_report",
                },
                "params": {
                    "prefill_engine": "nax_e8p",
                    "layer": 41,
                    "projections": ["gate_proj", "up_proj", "down_proj"],
                    "trainable": "low_rank_residual",
                    "low_rank": 4,
                    "steps": 1,
                    "learning_rate": 0.00025,
                    "target_nll_weight": 3.0,
                    "loss_scope": "selected_layer",
                    "train_cache": "validation",
                    "train_row_indices": [120, 73, 44, 0],
                    "max_positions": 1,
                },
                "gate": {"profile": "train_sane"},
            },
            {
                "id": "eval_report_tail_cleanup_1_focus",
                "op": "eval",
                "class": "diagnostic",
                "inputs": {
                    "artifact": "step:train_tail_cleanup_1/artifact",
                    "teacher": "external:teacher_report",
                },
                "params": {"engine": "vq_e1_routed_nax_e8p", "row_indices": [120, 73, 44, 0]},
            },
        ]
    )
    recipe_path.write_text(yaml.safe_dump(raw))
    extended = load_recipe(recipe_path)
    plan = plan_recipe(extended, build_root=tmp_path / "build")
    tail_train = plan.step("train_tail_cleanup_1")
    tail_train.argv = [
        sys.executable,
        "-c",
        _TRAIN_STUB,
        str(tail_train.out_dir),
        str(tail_train.evidence_dir / "train_log.jsonl"),
    ]
    focus = plan.step("eval_report_tail_cleanup_1_focus")
    focus.argv = [
        sys.executable,
        "-c",
        _DOMAIN_EVAL_STUB,
        str(focus.evidence_dir / "evidence.jsonl"),
    ]
    assert execute_plan(plan) == 0

    proposal = propose_next_step(extended, build_root=tmp_path / "build")

    assert proposal.base_step_id == "train_tail_cleanup_1"
    assert proposal.train_step.op == "train-router-kd"
    assert proposal.train_step.inputs["seed_artifact"].raw == "step:train_tail_cleanup_1/artifact"


def test_propose_next_step_prefers_declared_sparse_residual_plan(
    tmp_path: Path,
) -> None:
    recipe_path, _recipe = _executed_plan(tmp_path)
    sparse_plan = tmp_path / "sparse-plan.json"
    sparse_plan.write_text(json.dumps(_sparse_residual_plan_payload()))
    raw = yaml.safe_load(recipe_path.read_text())
    raw["external_inputs"]["sparse_residual_plan"] = {
        "kind": "file",
        "path": str(sparse_plan),
    }
    recipe_path.write_text(yaml.safe_dump(raw))
    extended = load_recipe(recipe_path)

    proposal = propose_next_step(extended, build_root=tmp_path / "build")

    assert proposal.base_step_id == "train"
    assert proposal.train_step.id == "sparse_residual_1"
    assert proposal.train_step.op == "sparse-residual"
    assert proposal.train_step.inputs["seed_artifact"].raw == "step:train/artifact"
    assert proposal.train_step.inputs["plan_json"].raw == "external:sparse_residual_plan"
    assert proposal.train_step.params["model_id"] == "zai-org/GLM-4.5-Air"
    assert proposal.train_step.params["revision"] == (
        "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"
    )
    assert proposal.train_step.params["projection"] == "down_proj"
    assert proposal.train_step.params["plan_target"] == "report_route_000:0"
    assert proposal.train_step.params["plan_expert"] == 93
    assert proposal.train_step.params["plan_route_rank"] == 0
    assert proposal.train_step.params["plan_max_rows"] == 4
    assert proposal.train_step.params["residual_scale"] == 0.5
    assert proposal.train_rows[0]["target_key"] == "report_route_000:0"
    argv = proposal.resolved_argv
    assert argv[3] == "benchmarks/materialize_glm45_air_vq_sparse_residual.py"
    assert argv[argv.index("--plan-json") + 1] == str(sparse_plan)
    assert argv[argv.index("--plan-target") + 1] == "report_route_000:0"
    assert argv[argv.index("--plan-expert") + 1] == "93"


def test_propose_next_step_does_not_repeat_declared_sparse_residual_plan(
    tmp_path: Path,
) -> None:
    recipe_path, _recipe = _executed_plan(tmp_path)
    sparse_plan = tmp_path / "sparse-plan.json"
    sparse_plan.write_text(json.dumps(_sparse_residual_plan_payload()))
    disagreement = tmp_path / "route-disagreement.json"
    disagreement.write_text(
        json.dumps({"record_type": "air_route_trace_disagreement_summary", "layers": {"36": {}}}),
        encoding="utf-8",
    )
    raw = yaml.safe_load(recipe_path.read_text())
    raw["external_inputs"]["sparse_residual_plan"] = {
        "kind": "file",
        "path": str(sparse_plan),
    }
    raw["external_inputs"]["route_disagreement"] = {
        "kind": "file",
        "path": str(disagreement),
    }
    raw["steps"].extend(
        [
            {
                "id": "sparse_residual_1",
                "op": "sparse-residual",
                "class": "promotable",
                "inputs": {
                    "seed_artifact": "step:train/artifact",
                    "plan_json": "external:sparse_residual_plan",
                },
                "params": {
                    "model_id": "zai-org/GLM-4.5-Air",
                    "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
                    "projection": "down_proj",
                    "plan_target": "report_route_000:0",
                    "plan_expert": 93,
                    "plan_route_rank": 0,
                    "plan_max_rows": 4,
                    "residual_scale": 0.5,
                },
                "gate": {"profile": "train_sane"},
            },
            {
                "id": "eval_report_sparse_residual_1_focus",
                "op": "eval",
                "class": "diagnostic",
                "inputs": {
                    "artifact": "step:sparse_residual_1/artifact",
                    "teacher": "external:teacher_report",
                },
                "params": {"engine": "vq_e1_routed_nax_e8p", "row_indices": [120, 73, 44, 0]},
            },
        ]
    )
    recipe_path.write_text(yaml.safe_dump(raw))
    extended = load_recipe(recipe_path)
    plan = plan_recipe(extended, build_root=tmp_path / "build")
    sparse = plan.step("sparse_residual_1")
    sparse.argv = [
        sys.executable,
        "-c",
        _TRAIN_STUB,
        str(sparse.out_dir),
        str(sparse.evidence_dir / "residual_log.jsonl"),
    ]
    focus = plan.step("eval_report_sparse_residual_1_focus")
    focus.argv = [
        sys.executable,
        "-c",
        _DOMAIN_EVAL_STUB,
        str(focus.evidence_dir / "evidence.jsonl"),
    ]
    assert execute_plan(plan) == 0

    proposal = propose_next_step(extended, build_root=tmp_path / "build")

    assert proposal.base_step_id == "sparse_residual_1"
    assert proposal.train_step.op == "train-router-kd"
    assert proposal.train_step.id == "train_router_kd_1"
    assert proposal.train_step.inputs["seed_artifact"].raw == "step:sparse_residual_1/artifact"


def test_propose_next_step_routes_around_sparse_residual_low_rank_dead_end(
    tmp_path: Path,
) -> None:
    recipe_path, _recipe = _executed_plan(tmp_path)
    sparse_plan = tmp_path / "sparse-plan.json"
    sparse_plan.write_text(json.dumps(_sparse_residual_plan_payload()))
    raw = yaml.safe_load(recipe_path.read_text())
    raw["external_inputs"]["sparse_residual_plan"] = {
        "kind": "file",
        "path": str(sparse_plan),
    }
    raw["steps"].extend(
        [
            {
                "id": "sparse_residual_1",
                "op": "sparse-residual",
                "class": "promotable",
                "inputs": {
                    "seed_artifact": "step:train/artifact",
                    "plan_json": "external:sparse_residual_plan",
                },
                "params": {
                    "model_id": "zai-org/GLM-4.5-Air",
                    "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
                    "projection": "down_proj",
                    "plan_target": "report_route_000:0",
                    "plan_expert": 93,
                    "plan_route_rank": 0,
                    "plan_max_rows": 4,
                    "residual_scale": 0.5,
                },
                "gate": {"profile": "train_sane"},
            },
            {
                "id": "eval_report_sparse_residual_1_focus",
                "op": "eval",
                "class": "diagnostic",
                "inputs": {
                    "artifact": "step:sparse_residual_1/artifact",
                    "teacher": "external:teacher_report",
                },
                "params": {"engine": "vq_e1_routed_nax_e8p", "row_indices": [120, 73, 44, 0]},
            },
        ]
    )
    recipe_path.write_text(yaml.safe_dump(raw))
    extended = load_recipe(recipe_path)
    plan = plan_recipe(extended, build_root=tmp_path / "build")
    sparse = plan.step("sparse_residual_1")
    sparse.argv = [
        sys.executable,
        "-c",
        _TRAIN_STUB,
        str(sparse.out_dir),
        str(sparse.evidence_dir / "residual_log.jsonl"),
    ]
    focus = plan.step("eval_report_sparse_residual_1_focus")
    focus.argv = [
        sys.executable,
        "-c",
        _DOMAIN_EVAL_STUB,
        str(focus.evidence_dir / "evidence.jsonl"),
    ]
    assert execute_plan(plan) == 0

    proposal = propose_next_step(extended, build_root=tmp_path / "build")

    assert proposal.base_step_id == "train"
    assert proposal.train_step.op == "train-low-rank"
    assert proposal.train_step.id == "train_plan_next_1"
    assert proposal.train_step.inputs["seed_artifact"].raw == "step:train/artifact"


def test_propose_next_step_rejects_repeated_top1_neutral_low_rank_loop(
    tmp_path: Path,
) -> None:
    recipe_path, _recipe = _executed_plan(tmp_path)
    raw = yaml.safe_load(recipe_path.read_text())
    raw["steps"].extend(
        [
            {
                "id": "train_plan_next_1",
                "op": "train-low-rank",
                "class": "promotable",
                "inputs": {
                    "seed_artifact": "step:train/artifact",
                    "selection_teacher": "external:teacher_selection",
                    "validation_teacher": "external:teacher_report",
                },
                "params": {
                    "prefill_engine": "nax_e8p",
                    "layer": 45,
                    "projections": ["gate_proj", "up_proj", "down_proj"],
                    "trainable": "low_rank_residual",
                    "low_rank": 4,
                    "steps": 4,
                    "learning_rate": 0.0005,
                    "target_nll_weight": 4.0,
                    "loss_scope": "selected_layer",
                    "train_cache": "validation",
                    "train_row_indices": [0, 1, 4, 5, 7, 8],
                    "max_positions": 1,
                },
                "gate": {"profile": "train_sane"},
            },
            {
                "id": "eval_report_train_plan_next_1_focus",
                "op": "eval",
                "class": "diagnostic",
                "inputs": {
                    "artifact": "step:train_plan_next_1/artifact",
                    "teacher": "external:teacher_report",
                },
                "params": {
                    "engine": "vq_e1_routed_nax_e8p",
                    "row_indices": [0, 1, 4, 5, 7, 8],
                },
            },
        ]
    )
    recipe_path.write_text(yaml.safe_dump(raw))
    extended = load_recipe(recipe_path)
    plan = plan_recipe(extended, build_root=tmp_path / "build")
    train_next = plan.step("train_plan_next_1")
    train_next.argv = [
        sys.executable,
        "-c",
        _TRAIN_STUB,
        str(train_next.out_dir),
        str(train_next.evidence_dir / "train_log.jsonl"),
    ]
    focus = plan.step("eval_report_train_plan_next_1_focus")
    focus.argv = [
        sys.executable,
        "-c",
        _DOMAIN_EVAL_STUB,
        str(focus.evidence_dir / "evidence.jsonl"),
    ]
    assert execute_plan(plan) == 0

    with pytest.raises(PlanNextError, match="top1-neutral low-rank loop"):
        propose_next_step(extended, build_root=tmp_path / "build")


def test_plan_next_cli_formats_sparse_residual_plan_rows() -> None:
    rows = [
        {
            "target_key": "report_route_000:0",
            "layer": 41,
            "projection": "down_proj",
            "expert": 93,
            "route_rank": 0,
        }
    ]

    assert _format_plan_next_row_summary(rows) == (
        "report_route_000:0 layer41 down_proj expert93 route0"
    )


def test_propose_next_step_requires_completed_evidence(tmp_path: Path) -> None:
    recipe_path = _write_recipe(tmp_path)
    recipe = load_recipe(recipe_path)
    with pytest.raises(PlanNextError, match="no completed promotable step"):
        propose_next_step(recipe, build_root=tmp_path / "build")


def test_propose_next_step_never_selects_from_holdout(tmp_path: Path) -> None:
    recipe_path, recipe = _executed_plan(tmp_path)
    # Add a holdout cache + eval whose evidence would dominate selection if
    # holdout were (wrongly) eligible.
    cache = tmp_path / "cache-holdout"
    cache.mkdir(exist_ok=True)
    (cache / "metadata.jsonl").write_text(
        json.dumps({"prompt_id": "holdout_math_001"}) + "\n"
    )
    raw = yaml.safe_load(recipe_path.read_text())
    raw["external_inputs"]["teacher_holdout"] = {
        "kind": "teacher_cache",
        "path": str(cache / "metadata.jsonl"),
    }
    raw["steps"].append(
        {
            "id": "eval_holdout",
            "op": "eval",
            "class": "verify",
            "inputs": {
                "artifact": "step:train/artifact",
                "teacher": "external:teacher_holdout",
            },
            "params": {"engine": "vq_e1_routed_nax_e8p"},
        }
    )
    recipe_path.write_text(yaml.safe_dump(raw))
    extended = load_recipe(recipe_path)

    plan = plan_recipe(extended, build_root=tmp_path / "build")
    holdout = plan.step("eval_holdout")
    holdout.evidence_dir.mkdir(parents=True, exist_ok=True)
    (holdout.evidence_dir / "evidence.jsonl").write_text(
        json.dumps(
            {
                "prompt_id": "holdout_route_000",
                "row_index": 99,
                "top1_agreement": 0.01,
                "mean_kld": 5.0,
                "ppl_ratio": 1.0,
                "nll_delta": 0.01,
                "token_klds": [5.0],
                "pageouts_delta": 0,
                "swapouts_delta": 0,
            }
        )
        + "\n"
    )
    proposal = propose_next_step(extended, build_root=tmp_path / "build")
    assert all(int(row["row_index"]) != 99 for row in proposal.train_rows)
