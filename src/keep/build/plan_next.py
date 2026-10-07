"""``keep plan-next``: propose the next recipe step from recorded evidence.

Phase-1 driver (propose only): reads the recipe's ledger, finds the latest
gate-passing promotable artifact, scores its eval evidence with the
flip-focused quality-plan machinery, and emits a proposed train step (plus
its verify steps) as a YAML fragment for human approval. The proposed argv
is rendered by the same ``train-low-rank`` op the runner executes — one
source of truth, no drift between proposed and executed commands.

Policy guardrails from the RC quality plan hold structurally: holdout
evidence is never used for row selection, the proposal stays at rank 4, and
the proposed step's own gates re-apply the Lane S and clean-eval discipline
when the human appends it to the recipe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from keep.build.ledger import Ledger
from keep.build.runner import BuildPlan, plan_recipe
from keep.build.recipe import GateSpec, InputRef, Recipe, StepSpec
from keep.quality.plan_next import (
    QUALITY_PLAN_DOMAIN_QUOTAS,
    _quality_plan_candidates,
    _select_quality_plan_rows,
)
from keep.quality.rc_gates import _summarize_eval_split

# Row selection may only ever see these splits; holdout is validation-only.
_TRAINING_SPLITS = ("report", "selection")
_AUTO_SAFE_NON_EXPERT_PRECISION_SURFACES = {
    "embed_tokens",
    "lm_head",
    "router_gates",
}
_SPARSE_RESIDUAL_PLAN_INPUTS = ("sparse_residual_plan", "sparse_plan")
_SPARSE_LOW_RANK_FALLBACK_ERROR_PREFIX = (
    "low-rank fallback cannot consume sparse-residual base step"
)
_DEFAULT_GLM45_AIR_MODEL_ID = "zai-org/GLM-4.5-Air"
_DEFAULT_GLM45_AIR_REVISION = "a24ceef6ce4f3536971efe9b778bdaa1bab18daa"


class PlanNextError(RuntimeError):
    pass


@dataclass
class Proposal:
    base_step_id: str
    train_step: StepSpec
    verify_steps: list[StepSpec]
    train_rows: list[dict[str, Any]]
    resolved_argv: list[str]

    def yaml_fragment(self) -> str:
        def _step_to_raw(spec: StepSpec) -> dict[str, Any]:
            raw: dict[str, Any] = {
                "id": spec.id,
                "op": spec.op,
                "class": spec.step_class,
                "inputs": {name: ref.raw for name, ref in spec.inputs.items()},
                "params": dict(spec.params),
            }
            if spec.gate is not None:
                gate: dict[str, Any] = {"profile": spec.gate.profile}
                if spec.gate.thresholds:
                    gate["thresholds"] = dict(spec.gate.thresholds)
                raw["gate"] = gate
            return raw

        steps = [_step_to_raw(self.train_step)] + [
            _step_to_raw(spec) for spec in self.verify_steps
        ]
        return yaml.safe_dump(steps, sort_keys=False)


def _split_for_teacher_ref(ref: InputRef) -> str | None:
    if ref.kind != "external":
        return None
    name = ref.name
    if name.endswith("report"):
        return "report"
    if name.endswith(("selection", "select")):
        return "selection"
    if name.endswith("holdout"):
        return "holdout"
    return None


def _completed_promotable_steps(plan: BuildPlan, ledger: Ledger) -> list[Any]:
    completed: list[Any] = []
    for planned in reversed(plan.steps):
        if planned.spec.step_class != "promotable":
            continue
        terminal = ledger.latest_terminal(planned.step_key)
        if terminal is not None and terminal["event"] == "completed":
            completed.append(planned)
    return completed


def _latest_completed_promotable(plan: BuildPlan, ledger: Ledger) -> Any:
    completed = _completed_promotable_steps(plan, ledger)
    if completed:
        return completed[0]
    raise PlanNextError(
        "no completed promotable step recorded in the ledger; run `keep build` first"
    )


def _completed_eval_evidence(
    plan: BuildPlan, ledger: Ledger, base_step_id: str
) -> dict[str, Any]:
    """Split name -> eval-summary for completed planning evals of the base artifact.

    Diagnostic focused evals are valid row-selection evidence for the next
    recipe proposal, but they are not cloned as promotion gates.
    """

    summaries: dict[str, Any] = {}
    for planned in plan.steps:
        spec = planned.spec
        if spec.op != "eval" or spec.step_class not in {"verify", "diagnostic"}:
            continue
        artifact_ref = spec.inputs.get("artifact")
        if artifact_ref is None or artifact_ref.kind != "step":
            continue
        if artifact_ref.name != base_step_id:
            continue
        teacher_ref = spec.inputs.get("teacher")
        split = _split_for_teacher_ref(teacher_ref) if teacher_ref else None
        if split is None or split not in _TRAINING_SPLITS:
            continue
        terminal = ledger.latest_terminal(planned.step_key)
        if terminal is None or terminal["event"] != "completed":
            continue
        evidence = planned.output_paths.get("evidence")
        if evidence is None or not evidence.exists():
            continue
        summaries[split] = _summarize_eval_split(evidence)
    return summaries


def _completed_eval_paths(
    plan: BuildPlan, ledger: Ledger, base_step_id: str
) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for planned in plan.steps:
        spec = planned.spec
        if spec.op != "eval" or spec.step_class not in {"verify", "diagnostic"}:
            continue
        artifact_ref = spec.inputs.get("artifact")
        if artifact_ref is None or artifact_ref.kind != "step":
            continue
        if artifact_ref.name != base_step_id:
            continue
        teacher_ref = spec.inputs.get("teacher")
        split = _split_for_teacher_ref(teacher_ref) if teacher_ref else None
        if split is None or split not in _TRAINING_SPLITS:
            continue
        terminal = ledger.latest_terminal(planned.step_key)
        if terminal is None or terminal["event"] != "completed":
            continue
        evidence = planned.output_paths.get("evidence")
        if evidence is not None and evidence.exists():
            paths.append((split, evidence))
    return paths


def _read_eval_rows_by_index(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            continue
        row_index = row.get("row_index")
        if isinstance(row_index, bool) or not isinstance(row_index, int):
            continue
        rows[row_index] = row
    return rows


def _eval_row_is_clean(row: dict[str, Any]) -> bool:
    return (
        int(row.get("pageouts_delta") or 0) == 0
        and int(row.get("swapouts_delta") or 0) == 0
    )


def _seed_step_id_for_train_step(spec: StepSpec) -> str | None:
    seed_ref = spec.inputs.get("seed_artifact")
    if seed_ref is None or seed_ref.kind != "step":
        return None
    return seed_ref.name


def _completed_low_rank_base_top1_stalled(
    plan: BuildPlan, ledger: Ledger, base: Any
) -> dict[str, Any] | None:
    spec = base.spec
    if spec.op != "train-low-rank":
        return None
    if not str(spec.id).startswith("train_plan_next_"):
        return None
    seed_step_id = _seed_step_id_for_train_step(spec)
    if seed_step_id is None:
        return None
    base_eval_paths = _completed_eval_paths(plan, ledger, spec.id)
    seed_eval_paths = dict(_completed_eval_paths(plan, ledger, seed_step_id))
    train_row_indices = set(
        _int_list(spec.params.get("train_row_indices"), name="train_row_indices")
    )
    for split, base_path in base_eval_paths:
        seed_path = seed_eval_paths.get(split)
        if seed_path is None:
            continue
        base_rows = _read_eval_rows_by_index(base_path)
        seed_rows = _read_eval_rows_by_index(seed_path)
        comparable_indices = [
            row_index
            for row_index in sorted(base_rows)
            if row_index in seed_rows
            and (not train_row_indices or row_index in train_row_indices)
        ]
        if not comparable_indices:
            continue
        if any(not _eval_row_is_clean(base_rows[row_index]) for row_index in comparable_indices):
            continue
        improved = False
        for row_index in comparable_indices:
            base_top1 = float(base_rows[row_index].get("top1_agreement") or 0.0)
            seed_top1 = float(seed_rows[row_index].get("top1_agreement") or 0.0)
            if base_top1 > seed_top1 + 1e-12:
                improved = True
                break
        if not improved:
            return {
                "base_step_id": spec.id,
                "seed_step_id": seed_step_id,
                "split": split,
                "row_indices": comparable_indices,
                "evidence_path": str(base_path),
            }
    return None


def _evidence_step_targets_base(
    plan: BuildPlan, *, evidence_step_id: str, base_step_id: str
) -> bool:
    try:
        evidence_step = plan.step(evidence_step_id).spec
    except KeyError:
        return False
    artifact_ref = evidence_step.inputs.get("artifact")
    return (
        evidence_step.op in {"eval", "eval-repair"}
        and artifact_ref is not None
        and artifact_ref.kind == "step"
        and artifact_ref.name == base_step_id
    )


def _completed_top1_gap_recommendation(
    plan: BuildPlan,
    ledger: Ledger,
    *,
    base_step_id: str,
) -> dict[str, Any] | None:
    for planned in reversed(plan.steps):
        spec = planned.spec
        if spec.op != "top1-gap" or spec.step_class != "diagnostic":
            continue
        evidence_ref = spec.inputs.get("evidence") or spec.inputs.get("candidate_evidence")
        if evidence_ref is not None and evidence_ref.kind == "step":
            if not _evidence_step_targets_base(
                plan,
                evidence_step_id=evidence_ref.name,
                base_step_id=base_step_id,
            ):
                continue
        terminal = ledger.latest_terminal(planned.step_key)
        if terminal is None or terminal["event"] != "completed":
            continue
        gap_path = planned.output_paths.get("gap")
        if gap_path is None or not gap_path.exists():
            continue
        payload = json.loads(gap_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return None


def _clone_verify_steps(
    plan: BuildPlan,
    *,
    base_step_id: str,
    next_step_id: str,
) -> list[StepSpec]:
    cloned_ids: dict[str, str] = {}
    for planned in plan.steps:
        spec = planned.spec
        if spec.step_class != "verify":
            continue
        artifact_ref = spec.inputs.get("artifact")
        if artifact_ref is None or artifact_ref.kind != "step":
            continue
        if artifact_ref.name != base_step_id:
            continue
        cloned_ids[spec.id] = f"{spec.id}_{next_step_id}"

    verify_steps: list[StepSpec] = []
    for planned in plan.steps:
        spec = planned.spec
        if spec.step_class != "verify":
            continue
        artifact_ref = spec.inputs.get("artifact")
        if artifact_ref is None or artifact_ref.kind != "step":
            continue
        if artifact_ref.name != base_step_id:
            continue
        new_inputs = {
            name: (
                InputRef.parse(f"step:{next_step_id}/artifact")
                if name == "artifact"
                else InputRef.parse(f"step:{cloned_ids[ref.name]}/{ref.output}")
                if ref.kind == "step" and ref.name in cloned_ids and ref.output is not None
                else ref
            )
            for name, ref in spec.inputs.items()
        }
        verify_steps.append(
            replace(spec, id=cloned_ids[spec.id], inputs=new_inputs)
        )
    return verify_steps


def _teacher_inputs_for_training_lineage(
    plan: BuildPlan, *, base_step_id: str
) -> dict[str, InputRef]:
    seen: set[str] = set()
    step_id = base_step_id
    while step_id not in seen:
        seen.add(step_id)
        spec = plan.step(step_id).spec
        teacher_inputs = {
            name: ref
            for name, ref in spec.inputs.items()
            if name in ("selection_teacher", "validation_teacher")
        }
        if len(teacher_inputs) == 2:
            return teacher_inputs
        seed_ref = spec.inputs.get("seed_artifact")
        if seed_ref is None or seed_ref.kind != "step":
            break
        step_id = seed_ref.name
    return {}


def _train_low_rank_params_for_lineage(
    plan: BuildPlan, *, base_step_id: str
) -> dict[str, Any]:
    seen: set[str] = set()
    step_id = base_step_id
    while step_id not in seen:
        seen.add(step_id)
        spec = plan.step(step_id).spec
        if spec.op == "train-low-rank":
            return dict(spec.params)
        seed_ref = spec.inputs.get("seed_artifact")
        if seed_ref is None or seed_ref.kind != "step":
            break
        step_id = seed_ref.name
    return {}


def _weakest_top1_focus_row(summaries: dict[str, Any]) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for split_name, summary in summaries.items():
        focus = summary.get("quality_focus")
        if not isinstance(focus, dict):
            continue
        for row in focus.get("lowest_top1_rows") or []:
            if not isinstance(row, dict) or row.get("top1_agreement") is None:
                continue
            enriched = dict(row)
            enriched["split"] = split_name
            rows.append(enriched)
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            float(row["top1_agreement"]),
            str(row.get("split") or ""),
            int(row.get("row_index") or 0),
        ),
    )


def _route_gap_should_use_router_kd(recipe: Recipe, summaries: dict[str, Any]) -> bool:
    if "route_disagreement" not in recipe.external_inputs:
        return False
    weakest = _weakest_top1_focus_row(summaries)
    if weakest is None:
        return False
    return (
        weakest.get("domain") == "route"
        and float(weakest.get("top1_agreement") or 1.0) < 0.80
    )


def _surface_list(value: Any) -> list[str]:
    if value is None:
        return ["embed_tokens", "lm_head"]
    if isinstance(value, str):
        surfaces = [surface.strip() for surface in value.split(",")]
    else:
        surfaces = [str(surface).strip() for surface in value]
    return [surface for surface in surfaces if surface] or ["embed_tokens", "lm_head"]


def _auto_safe_non_expert_surface_list(value: Any) -> list[str]:
    surfaces = _surface_list(value)
    unsafe = [
        surface
        for surface in surfaces
        if surface not in _AUTO_SAFE_NON_EXPERT_PRECISION_SURFACES
    ]
    if unsafe:
        safe = ", ".join(sorted(_AUTO_SAFE_NON_EXPERT_PRECISION_SURFACES))
        bad = ", ".join(unsafe)
        raise PlanNextError(
            "top1-gap recommended non-auto-safe non-expert precision surfaces "
            f"{bad}; only {safe} may be auto-proposed by plan-next"
        )
    return surfaces


def _completed_tail_cleanup_recommendation(
    recommendation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(recommendation, dict):
        return None
    if recommendation.get("record_type") != "glm45_air_tail_cleanup_target_selection":
        return None
    if recommendation.get("decision") != "tail_cleanup_report_train_packet_ready":
        return None
    training = recommendation.get("recommended_training")
    if not isinstance(training, dict):
        return None
    rows = training.get("train_row_indices")
    if not isinstance(rows, list) or not rows:
        return None
    return training


def _int_list(value: Any, *, name: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise PlanNextError(f"tail-cleanup recommendation field {name!r} must be a list")
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError) as error:
            raise PlanNextError(
                f"tail-cleanup recommendation field {name!r} contains non-integer value {item!r}"
            ) from error
    return result


def _declared_sparse_residual_plan(
    recipe: Recipe,
) -> tuple[str, dict[str, Any]] | None:
    input_name = next(
        (name for name in _SPARSE_RESIDUAL_PLAN_INPUTS if name in recipe.external_inputs),
        None,
    )
    if input_name is None:
        return None
    external = recipe.external_inputs[input_name]
    try:
        payload = json.loads(Path(external.path).read_text(encoding="utf-8"))
    except OSError as error:
        raise PlanNextError(
            f"sparse residual plan input {input_name!r} is not readable: {external.path}"
        ) from error
    except json.JSONDecodeError as error:
        raise PlanNextError(
            f"sparse residual plan input {input_name!r} is not valid JSON: {external.path}"
        ) from error
    rankings = ((payload.get("summary") or {}).get("route_source_sparse_residual_plan_rankings") or [])
    if not isinstance(rankings, list) or not rankings:
        raise PlanNextError(
            f"sparse residual plan input {input_name!r} has no route_source_sparse_residual_plan_rankings"
        )
    rows = [row for row in rankings if isinstance(row, dict)]
    if not rows:
        raise PlanNextError(
            f"sparse residual plan input {input_name!r} has no usable sparse residual rankings"
        )
    rows.sort(
        key=lambda row: (
            int(row.get("rank") or 1_000_000),
            -abs(float(row.get("desired_weighted_correction") or 0.0)),
            int(row.get("route_rank") or 0),
            int(row.get("expert") or 0),
        )
    )
    best = rows[0]
    required = ("target_key", "expert", "route_rank", "projection")
    missing = [name for name in required if best.get(name) is None]
    if missing:
        raise PlanNextError(
            f"sparse residual plan input {input_name!r} ranking is missing {', '.join(missing)}"
        )
    return input_name, best


def _sparse_residual_plan_already_consumed(recipe: Recipe, input_name: str) -> bool:
    expected = f"external:{input_name}"
    for step in recipe.steps:
        if step.op != "sparse-residual":
            continue
        plan_ref = step.inputs.get("plan_json")
        if plan_ref is not None and plan_ref.raw == expected:
            return True
    return False


def _next_step_id(recipe: Recipe, prefix: str) -> str:
    index = 1
    existing = {spec.id for spec in recipe.steps}
    while f"{prefix}{index}" in existing:
        index += 1
    return f"{prefix}{index}"


def propose_next_step(
    recipe: Recipe,
    *,
    build_root: Any = None,
    no_hash: bool = False,
) -> Proposal:
    plan = plan_recipe(recipe, build_root=build_root, no_hash=no_hash)
    ledger = Ledger(plan.build_root / "ledger.jsonl")
    bases = _completed_promotable_steps(plan, ledger)
    if not bases:
        _latest_completed_promotable(plan, ledger)

    sparse_low_rank_errors: list[PlanNextError] = []
    for base in bases:
        try:
            return _propose_next_step_from_base(
                recipe,
                plan=plan,
                ledger=ledger,
                base=base,
                build_root=build_root,
                no_hash=no_hash,
            )
        except PlanNextError as exc:
            if str(exc).startswith(_SPARSE_LOW_RANK_FALLBACK_ERROR_PREFIX):
                sparse_low_rank_errors.append(exc)
                continue
            raise

    if sparse_low_rank_errors:
        raise sparse_low_rank_errors[-1]
    raise PlanNextError(
        "no completed promotable step recorded in the ledger; run `keep build` first"
    )


def _propose_next_step_from_base(
    recipe: Recipe,
    *,
    plan: BuildPlan,
    ledger: Ledger,
    base: Any,
    build_root: Any = None,
    no_hash: bool = False,
) -> Proposal:

    gap_recommendation = _completed_top1_gap_recommendation(
        plan,
        ledger,
        base_step_id=base.spec.id,
    )
    tail_cleanup = _completed_tail_cleanup_recommendation(gap_recommendation)
    if tail_cleanup is not None:
        teacher_inputs = _teacher_inputs_for_training_lineage(
            plan, base_step_id=base.spec.id
        )
        if len(teacher_inputs) < 2:
            raise PlanNextError(
                f"base step {base.spec.id!r} lineage does not carry teacher-cache inputs to inherit"
            )
        row_indices = _int_list(tail_cleanup.get("train_row_indices"), name="train_row_indices")
        aux_positions = _int_list(
            tail_cleanup.get("aux_loss_position_indices"),
            name="aux_loss_position_indices",
        ) or [0]
        step_id = _next_step_id(recipe, "train_tail_cleanup_")
        params = {
            "prefill_engine": "nax_e8p",
            "layer": int(tail_cleanup.get("layer", 41)),
            "projections": ["gate_proj", "up_proj", "down_proj"],
            "trainable": "low_rank_residual",
            "low_rank": int(tail_cleanup.get("low_rank", 4)),
            "low_rank_init_scale": float(tail_cleanup.get("low_rank_init_scale", 0.05)),
            "steps": int(tail_cleanup.get("steps", 1)),
            "learning_rate": float(tail_cleanup.get("learning_rate", 0.00025)),
            "target_nll_weight": float(tail_cleanup.get("target_nll_weight", 3.0)),
            "teacher_top1_margin_weight": float(
                tail_cleanup.get("teacher_top1_margin_weight", 2.0)
            ),
            "teacher_top1_margin": float(tail_cleanup.get("teacher_top1_margin", 1.0)),
            "tail_kld_weight": float(tail_cleanup.get("tail_kld_weight", 0.0001)),
            "loss_scope": "selected_layer",
            "surrogate_projections": ["gate_proj", "up_proj", "down_proj"],
            "surrogate_output_chunk_size": int(
                tail_cleanup.get("surrogate_output_chunk_size", 256)
            ),
            "train_cache": str(tail_cleanup.get("train_cache") or "validation"),
            "max_train_rows": len(row_indices),
            "train_row_indices": row_indices,
            "max_positions": int(tail_cleanup.get("max_positions", 1)),
            "aux_loss_position_indices": aux_positions,
        }
        tail_step = StepSpec(
            id=step_id,
            op="train-low-rank",
            step_class="promotable",
            inputs={
                "seed_artifact": InputRef.parse(f"step:{base.spec.id}/artifact"),
                **teacher_inputs,
            },
            params=params,
            gate=GateSpec(profile="train_sane"),
        )
        verify_steps = _clone_verify_steps(
            plan,
            base_step_id=base.spec.id,
            next_step_id=step_id,
        )
        extended = replace(recipe, steps=(*recipe.steps, tail_step, *verify_steps))
        extended_plan = plan_recipe(extended, build_root=build_root, no_hash=no_hash)
        return Proposal(
            base_step_id=base.spec.id,
            train_step=tail_step,
            verify_steps=verify_steps,
            train_rows=list(tail_cleanup.get("train_rows") or []),
            resolved_argv=extended_plan.step(step_id).argv,
        )
    if (
        gap_recommendation is not None
        and gap_recommendation.get("recommended_op") == "bump-non-expert-precision"
    ):
        step_id = _next_step_id(recipe, "bump_non_expert_precision_")
        surfaces = _auto_safe_non_expert_surface_list(
            gap_recommendation.get("recommended_surfaces")
        )
        precision_step = StepSpec(
            id=step_id,
            op="bump-non-expert-precision",
            step_class="promotable",
            inputs={
                "seed_artifact": InputRef.parse(f"step:{base.spec.id}/artifact"),
            },
            params={
                "surfaces": surfaces,
                "dtype": str(gap_recommendation.get("recommended_dtype") or "bf16"),
                "reason": str(gap_recommendation.get("reason") or "top1-gap recommendation"),
            },
            gate=GateSpec(profile="train_sane"),
        )
        verify_steps = _clone_verify_steps(
            plan,
            base_step_id=base.spec.id,
            next_step_id=step_id,
        )
        extended = replace(recipe, steps=(*recipe.steps, precision_step, *verify_steps))
        extended_plan = plan_recipe(extended, build_root=build_root, no_hash=no_hash)
        return Proposal(
            base_step_id=base.spec.id,
            train_step=precision_step,
            verify_steps=verify_steps,
            train_rows=[],
            resolved_argv=extended_plan.step(step_id).argv,
        )

    summaries = _completed_eval_evidence(plan, ledger, base.spec.id)
    if not summaries:
        raise PlanNextError(
            f"no completed report/selection eval evidence for step {base.spec.id!r}"
        )
    sparse_plan = _declared_sparse_residual_plan(recipe)
    if sparse_plan is not None:
        input_name, sparse_row = sparse_plan
        if not _sparse_residual_plan_already_consumed(recipe, input_name):
            step_id = _next_step_id(recipe, "sparse_residual_")
            sparse_step = StepSpec(
                id=step_id,
                op="sparse-residual",
                step_class="promotable",
                inputs={
                    "seed_artifact": InputRef.parse(f"step:{base.spec.id}/artifact"),
                    "plan_json": InputRef.parse(f"external:{input_name}"),
                },
                params={
                    "model_id": str(
                        sparse_row.get("model_id") or _DEFAULT_GLM45_AIR_MODEL_ID
                    ),
                    "revision": str(
                        sparse_row.get("revision") or _DEFAULT_GLM45_AIR_REVISION
                    ),
                    "projection": str(sparse_row["projection"]),
                    "plan_target": str(sparse_row["target_key"]),
                    "plan_expert": int(sparse_row["expert"]),
                    "plan_route_rank": int(sparse_row["route_rank"]),
                    "plan_max_rows": int(sparse_row.get("plan_max_rows") or 4),
                    "residual_scale": float(sparse_row.get("residual_scale") or 0.5),
                },
                gate=GateSpec(profile="train_sane"),
            )
            verify_steps = _clone_verify_steps(
                plan,
                base_step_id=base.spec.id,
                next_step_id=step_id,
            )
            extended = replace(recipe, steps=(*recipe.steps, sparse_step, *verify_steps))
            extended_plan = plan_recipe(extended, build_root=build_root, no_hash=no_hash)
            return Proposal(
                base_step_id=base.spec.id,
                train_step=sparse_step,
                verify_steps=verify_steps,
                train_rows=[sparse_row],
                resolved_argv=extended_plan.step(step_id).argv,
            )
    if _route_gap_should_use_router_kd(recipe, summaries):
        step_id = _next_step_id(recipe, "train_router_kd_")
        router_step = StepSpec(
            id=step_id,
            op="train-router-kd",
            step_class="promotable",
            inputs={
                "seed_artifact": InputRef.parse(f"step:{base.spec.id}/artifact"),
                "disagreement": InputRef.parse("external:route_disagreement"),
            },
            params={
                "scale": 1,
                "max_abs_delta": 0.125,
                "num_experts": 128,
            },
            gate=GateSpec(profile="train_sane"),
        )
        verify_steps = _clone_verify_steps(
            plan,
            base_step_id=base.spec.id,
            next_step_id=step_id,
        )
        extended = replace(recipe, steps=(*recipe.steps, router_step, *verify_steps))
        extended_plan = plan_recipe(extended, build_root=build_root, no_hash=no_hash)
        weakest = _weakest_top1_focus_row(summaries)
        return Proposal(
            base_step_id=base.spec.id,
            train_step=router_step,
            verify_steps=verify_steps,
            train_rows=[] if weakest is None else [weakest],
            resolved_argv=extended_plan.step(step_id).argv,
        )
    report_like = {"eval_splits": summaries}
    candidates = _quality_plan_candidates(
        report_like,
        split_names=tuple(sorted(summaries)),
        allowed_domains=set(QUALITY_PLAN_DOMAIN_QUOTAS),
    )
    train_rows = _select_quality_plan_rows(
        candidates, quotas=QUALITY_PLAN_DOMAIN_QUOTAS
    )
    if not train_rows:
        raise PlanNextError("quality-plan selection produced no candidate rows")
    row_indices = [int(row["row_index"]) for row in train_rows]
    if base.spec.op == "sparse-residual":
        raise PlanNextError(
            "low-rank fallback cannot consume sparse-residual base step "
            f"{base.spec.id!r}; run a non-sparse branch or a sparse-aware trainer"
        )
    stalled = _completed_low_rank_base_top1_stalled(plan, ledger, base)
    if stalled is not None:
        raise PlanNextError(
            "top1-neutral low-rank loop at base step "
            f"{stalled['base_step_id']!r}: clean {stalled['split']} evidence "
            f"{stalled['evidence_path']} showed no top1 improvement over seed step "
            f"{stalled['seed_step_id']!r} on rows {stalled['row_indices']}; "
            "choose a different mechanism instead of repeating train-low-rank fallback"
        )

    # Inherit the accepted training recipe from the base step when it was a
    # trainer itself; the proposal only re-targets the row selection.
    base_params = _train_low_rank_params_for_lineage(plan, base_step_id=base.spec.id)
    base_params["train_row_indices"] = row_indices
    base_params.setdefault("max_train_rows", len(row_indices))

    teacher_inputs = _teacher_inputs_for_training_lineage(
        plan, base_step_id=base.spec.id
    )
    if len(teacher_inputs) < 2:
        raise PlanNextError(
            f"base step {base.spec.id!r} lineage does not carry teacher-cache inputs to inherit"
        )

    train_id = _next_step_id(recipe, "train_plan_next_")
    train_step = StepSpec(
        id=train_id,
        op="train-low-rank",
        step_class="promotable",
        inputs={
            "seed_artifact": InputRef.parse(f"step:{base.spec.id}/artifact"),
            **teacher_inputs,
        },
        params=base_params,
        gate=GateSpec(profile="train_sane"),
    )

    verify_steps = _clone_verify_steps(
        plan,
        base_step_id=base.spec.id,
        next_step_id=train_id,
    )

    extended = replace(
        recipe, steps=(*recipe.steps, train_step, *verify_steps)
    )
    extended_plan = plan_recipe(extended, build_root=build_root, no_hash=no_hash)
    resolved_argv = extended_plan.step(train_id).argv

    return Proposal(
        base_step_id=base.spec.id,
        train_step=train_step,
        verify_steps=verify_steps,
        train_rows=train_rows,
        resolved_argv=resolved_argv,
    )


# ---------------------------------------------------------------------------
# Phase 2: --auto-extend. Proposals land in an autogen overlay file
# (recipes/<name>.autogen.yaml); the human-authored recipe is never
# machine-edited. A gate-failing proposal moves to the overlay's `rejected`
# list — preserved as evidence, excluded from future planning — and the next
# iteration proposes from the still-latest gate-passing artifact.
# ---------------------------------------------------------------------------


def overlay_path_for(recipe_path: Any) -> Any:
    from pathlib import Path

    path = Path(recipe_path)
    return path.with_name(path.stem + ".autogen.yaml")


def _load_overlay(overlay_path: Any) -> dict[str, Any]:
    if not overlay_path.exists():
        return {"schema_version": 1, "steps": [], "rejected": []}
    raw = yaml.safe_load(overlay_path.read_text()) or {}
    raw.setdefault("schema_version", 1)
    raw.setdefault("steps", [])
    raw.setdefault("rejected", [])
    return raw


def load_merged_recipe(recipe_path: Any) -> Recipe:
    """Base recipe with accepted autogen-overlay steps appended."""

    from keep.build.recipe import load_recipe, parse_recipe

    overlay = _load_overlay(overlay_path_for(recipe_path))
    if not overlay["steps"]:
        return load_recipe(recipe_path)
    base_raw = yaml.safe_load(open(recipe_path).read())
    base_raw["steps"] = list(base_raw.get("steps") or []) + list(overlay["steps"])
    return parse_recipe(base_raw, source_path=str(recipe_path))


def auto_extend(
    recipe_path: Any,
    *,
    iterations: int,
    build_root: Any = None,
    prepare_plan: Any = None,
) -> int:
    """Run up to ``iterations`` propose -> execute -> gate cycles.

    Returns the number of accepted proposals. ``prepare_plan`` is a test
    hook invoked with each BuildPlan before execution (argv stubbing).
    """

    from keep.build.executor import execute_plan

    overlay_file = overlay_path_for(recipe_path)
    accepted = 0
    for _ in range(iterations):
        recipe = load_merged_recipe(recipe_path)
        try:
            proposal = propose_next_step(recipe, build_root=build_root)
        except PlanNextError as error:
            print(f"auto-extend stop: {error}")
            break

        proposed_raw = yaml.safe_load(proposal.yaml_fragment())
        overlay = _load_overlay(overlay_file)
        overlay["steps"] = list(overlay["steps"]) + proposed_raw
        overlay_file.write_text(yaml.safe_dump(overlay, sort_keys=False))

        merged = load_merged_recipe(recipe_path)
        plan = plan_recipe(merged, build_root=build_root)
        if prepare_plan is not None:
            prepare_plan(plan)
        rc = execute_plan(plan)
        if rc == 0:
            accepted += 1
            print(f"auto-extend accepted {proposal.train_step.id}")
            continue

        # Rejection: preserve the proposal as evidence, remove it from the
        # active step list so the next iteration re-plans from the latest
        # good artifact.
        overlay = _load_overlay(overlay_file)
        overlay["steps"] = [
            step for step in overlay["steps"] if step not in proposed_raw
        ]
        overlay["rejected"] = list(overlay["rejected"]) + [
            {
                "steps": proposed_raw,
                "reason": "step_or_gate_failed_see_ledger",
                "base_step_id": proposal.base_step_id,
            }
        ]
        overlay_file.write_text(yaml.safe_dump(overlay, sort_keys=False))
        print(f"auto-extend rejected {proposal.train_step.id} (see ledger)")
    return accepted
