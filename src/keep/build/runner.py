"""Recipe graph planner and executor.

Planning resolves the step graph bottom-up: external inputs are identity-
hashed, each step's key is derived from (op, op_version, params, input
identities), and every step gets a content-key-named directory under the
recipe's build root. Declaration order is the topological order — the
validator rejects forward references, so cycles cannot exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from keep.build import hashing
from keep.build.ops import REGISTRY, OpDef, StepContext, derive_implicit_inputs
from keep.build.recipe import Recipe, RecipeError, StepSpec, validate_recipe

DEFAULT_BUILD_ROOT = Path("artifacts/build")


@dataclass
class PlannedStep:
    spec: StepSpec
    op_def: OpDef
    step_key: str
    input_hashes: dict[str, str]
    resolved_inputs: dict[str, Path]
    step_dir: Path
    argv: list[str]
    output_paths: dict[str, Path]

    @property
    def out_dir(self) -> Path:
        return self.step_dir / "out"

    @property
    def evidence_dir(self) -> Path:
        return self.step_dir / "evidence"


@dataclass
class BuildPlan:
    recipe: Recipe
    build_root: Path
    steps: list[PlannedStep] = field(default_factory=list)
    external_hashes: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def step(self, step_id: str) -> PlannedStep:
        for planned in self.steps:
            if planned.spec.id == step_id:
                return planned
        raise KeyError(f"no planned step with id {step_id!r}")


def _step_dir_name(step_id: str, step_key: str, *, no_hash: bool) -> str:
    suffix = "nohash" if no_hash else step_key[:8]
    return f"{step_id}-{suffix}"


def plan_recipe(
    recipe: Recipe,
    *,
    build_root: Path | None = None,
    no_hash: bool = False,
    strict_inputs: bool = False,
) -> BuildPlan:
    errors = validate_recipe(recipe, REGISTRY)
    if errors:
        raise RecipeError(errors)

    strict_inputs = strict_inputs or recipe.strict_inputs
    if no_hash and strict_inputs:
        raise RecipeError(
            [
                f"recipe {recipe.name!r} requires strict input hashes and does "
                "not allow --no-hash"
            ]
        )
    root = (build_root or DEFAULT_BUILD_ROOT) / recipe.name
    plan = BuildPlan(recipe=recipe, build_root=root)

    for name, external in recipe.external_inputs.items():
        if no_hash:
            plan.external_hashes[name] = hashing.UNHASHED
            continue
        resolved = hashing.external_input_hash(external.kind, Path(external.path))
        plan.external_hashes[name] = resolved
        if external.expect_hash is not None and external.expect_hash != resolved:
            message = (
                f"external input {name!r} hash mismatch: expected "
                f"{external.expect_hash}, got {resolved}"
            )
            if strict_inputs:
                raise RecipeError([message])
            plan.warnings.append(message)

    planned_by_id: dict[str, PlannedStep] = {}
    for spec in recipe.steps:
        op_def = REGISTRY[spec.op]
        input_hashes: dict[str, str] = {}
        resolved_inputs: dict[str, Path] = {}
        for input_name, ref in spec.inputs.items():
            if ref.kind == "external":
                input_hashes[input_name] = plan.external_hashes[ref.name]
                resolved_inputs[input_name] = Path(
                    recipe.external_inputs[ref.name].path
                )
            else:
                producer = planned_by_id[ref.name]
                input_hashes[input_name] = f"{producer.step_key}/{ref.output}"
                resolved_inputs[input_name] = producer.output_paths[ref.output]

        key = hashing.step_key(
            op=spec.op,
            op_version=op_def.version,
            params=spec.params,
            input_keys=input_hashes,
        )
        step_dir = root / "steps" / _step_dir_name(spec.id, key, no_hash=no_hash)
        ctx = StepContext(
            spec=spec,
            inputs=derive_implicit_inputs(spec.op, resolved_inputs),
            out_dir=step_dir / "out",
            evidence_dir=step_dir / "evidence",
        )
        planned = PlannedStep(
            spec=spec,
            op_def=op_def,
            step_key=key,
            input_hashes=input_hashes,
            resolved_inputs=dict(ctx.inputs),
            step_dir=step_dir,
            argv=op_def.build_argv(ctx),
            output_paths=op_def.output_paths(ctx),
        )
        plan.steps.append(planned)
        planned_by_id[spec.id] = planned
    return plan


def render_dry_run(plan: BuildPlan) -> str:
    lines: list[str] = [
        f"recipe: {plan.recipe.name}",
        f"build root: {plan.build_root}",
    ]
    for warning in plan.warnings:
        lines.append(f"warning: {warning}")
    for planned in plan.steps:
        spec = planned.spec
        gate = spec.gate.profile if spec.gate is not None else "-"
        lines.append("")
        lines.append(
            f"step {spec.id} [{spec.step_class}] op={spec.op} key={planned.step_key} gate={gate}"
        )
        for input_name, input_hash in sorted(planned.input_hashes.items()):
            lines.append(f"  input {input_name}: {input_hash}")
        lines.append(f"  argv: {' '.join(planned.argv)}")
    if plan.recipe.promotion is not None:
        promotion = plan.recipe.promotion
        lines.append("")
        lines.append(f"promotion: artifact_step={promotion.artifact_step}")
        for gate in promotion.gates:
            lines.append(f"  gate {gate.evidence_step}: {gate.profile}")
    return "\n".join(lines)
