"""Recipe schema: declarative build graphs loaded from YAML.

A recipe is an ordered list of steps over external inputs. Step classes
enforce the promotion-integrity rule structurally:

- ``promotable`` steps emit artifact lineage and may only consume external
  inputs or other promotable steps.
- ``verify`` steps emit evidence only and may additionally consume verify
  evidence.
- ``diagnostic`` steps may consume anything, but nothing outside the
  diagnostic class may consume their outputs — eval-fitted transforms (the
  logit-bias bundle) can be expressed and measured but can never enter a
  promotable artifact's ancestry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

RECIPE_SCHEMA_VERSION = 1

STEP_CLASSES = ("promotable", "verify", "diagnostic")
_ALLOWED_INPUT_CLASSES = {
    "promotable": {"promotable"},
    "verify": {"promotable", "verify"},
    "diagnostic": {"promotable", "verify", "diagnostic"},
}
EXTERNAL_INPUT_KINDS = (
    "artifact_dir",
    "teacher_cache",
    "teacher_cache_artifact",
    "file",
)


class RecipeError(ValueError):
    """Raised when a recipe fails to parse or validate."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class ExternalInput:
    name: str
    kind: str
    path: str
    expect_hash: str | None = None


@dataclass(frozen=True)
class InputRef:
    raw: str
    kind: str  # "external" | "step"
    name: str  # external input name or step id
    output: str | None = None  # output name for step refs

    @staticmethod
    def parse(raw: str) -> "InputRef":
        if raw.startswith("external:"):
            name = raw[len("external:"):]
            if not name:
                raise ValueError(f"empty external input reference: {raw!r}")
            return InputRef(raw=raw, kind="external", name=name)
        if raw.startswith("step:"):
            body = raw[len("step:"):]
            step_id, sep, output = body.partition("/")
            if not step_id or not sep or not output:
                raise ValueError(
                    f"step reference must be step:<id>/<output>: {raw!r}"
                )
            return InputRef(raw=raw, kind="step", name=step_id, output=output)
        raise ValueError(
            f"input reference must start with external: or step:, got {raw!r}"
        )


@dataclass(frozen=True)
class GateSpec:
    profile: str
    thresholds: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepSpec:
    id: str
    op: str
    step_class: str
    inputs: Mapping[str, InputRef] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)
    gate: GateSpec | None = None


@dataclass(frozen=True)
class PromotionGate:
    evidence_step: str
    profile: str
    thresholds: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Promotion:
    artifact_step: str
    gates: tuple[PromotionGate, ...] = ()
    target_metric_profile: str | None = None


@dataclass(frozen=True)
class Recipe:
    name: str
    description: str
    external_inputs: Mapping[str, ExternalInput]
    steps: tuple[StepSpec, ...]
    promotion: Promotion | None = None
    strict_inputs: bool = False
    source_path: str | None = None

    def step(self, step_id: str) -> StepSpec:
        for spec in self.steps:
            if spec.id == step_id:
                return spec
        raise KeyError(f"no step with id {step_id!r}")


def _require_mapping(value: Any, context: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{context}: expected a mapping, got {type(value).__name__}")
        return {}
    return dict(value)


def _parse_gate(value: Any, context: str, errors: list[str]) -> GateSpec | None:
    if value is None:
        return None
    gate = _require_mapping(value, context, errors)
    profile = gate.get("profile")
    if not isinstance(profile, str) or not profile:
        errors.append(f"{context}: gate requires a profile name")
        return None
    thresholds = gate.get("thresholds") or {}
    if not isinstance(thresholds, Mapping):
        errors.append(f"{context}: gate thresholds must be a mapping")
        thresholds = {}
    return GateSpec(profile=profile, thresholds=dict(thresholds))


def parse_recipe(raw: Mapping[str, Any], *, source_path: str | None = None) -> Recipe:
    errors: list[str] = []
    if raw.get("schema_version") != RECIPE_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {RECIPE_SCHEMA_VERSION}, got {raw.get('schema_version')!r}"
        )
    strict_inputs = raw.get("strict_inputs", False)
    if not isinstance(strict_inputs, bool):
        errors.append("strict_inputs must be a boolean")
        strict_inputs = False
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        errors.append("recipe requires a non-empty name")
        name = "<unnamed>"

    external_inputs: dict[str, ExternalInput] = {}
    for input_name, value in _require_mapping(
        raw.get("external_inputs") or {}, "external_inputs", errors
    ).items():
        entry = _require_mapping(value, f"external_inputs.{input_name}", errors)
        kind = entry.get("kind")
        if kind not in EXTERNAL_INPUT_KINDS:
            errors.append(
                f"external_inputs.{input_name}: kind must be one of {EXTERNAL_INPUT_KINDS}, got {kind!r}"
            )
            kind = "file"
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            errors.append(f"external_inputs.{input_name}: requires a path")
            path = ""
        expect_hash = entry.get("expect_hash")
        if expect_hash is not None and not isinstance(expect_hash, str):
            errors.append(f"external_inputs.{input_name}: expect_hash must be a string")
            expect_hash = None
        external_inputs[input_name] = ExternalInput(
            name=input_name, kind=str(kind), path=path, expect_hash=expect_hash
        )

    steps: list[StepSpec] = []
    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        errors.append("steps must be a list")
        raw_steps = []
    for index, value in enumerate(raw_steps):
        context = f"steps[{index}]"
        entry = _require_mapping(value, context, errors)
        step_id = entry.get("id")
        if not isinstance(step_id, str) or not step_id:
            errors.append(f"{context}: requires an id")
            step_id = f"<step-{index}>"
        op = entry.get("op")
        if not isinstance(op, str) or not op:
            errors.append(f"{context} ({step_id}): requires an op")
            op = "<missing>"
        step_class = entry.get("class")
        if step_class not in STEP_CLASSES:
            errors.append(
                f"{context} ({step_id}): class must be one of {STEP_CLASSES}, got {step_class!r}"
            )
            step_class = "diagnostic"
        inputs: dict[str, InputRef] = {}
        for input_name, ref in _require_mapping(
            entry.get("inputs") or {}, f"{context}.inputs", errors
        ).items():
            if not isinstance(ref, str):
                errors.append(
                    f"{context} ({step_id}).inputs.{input_name}: expected a reference string"
                )
                continue
            try:
                inputs[input_name] = InputRef.parse(ref)
            except ValueError as error:
                errors.append(f"{context} ({step_id}).inputs.{input_name}: {error}")
        params = _require_mapping(entry.get("params") or {}, f"{context}.params", errors)
        gate = _parse_gate(entry.get("gate"), f"{context} ({step_id})", errors)
        steps.append(
            StepSpec(
                id=step_id,
                op=op,
                step_class=str(step_class),
                inputs=inputs,
                params=params,
                gate=gate,
            )
        )

    promotion: Promotion | None = None
    raw_promotion = raw.get("promotion")
    if raw_promotion is not None:
        entry = _require_mapping(raw_promotion, "promotion", errors)
        artifact_step = entry.get("artifact_step")
        if not isinstance(artifact_step, str) or not artifact_step:
            errors.append("promotion: requires artifact_step")
            artifact_step = "<missing>"
        gates: list[PromotionGate] = []
        for index, value in enumerate(entry.get("gates") or []):
            gate_entry = _require_mapping(value, f"promotion.gates[{index}]", errors)
            evidence_step = gate_entry.get("evidence_step")
            profile = gate_entry.get("profile")
            if not isinstance(evidence_step, str) or not isinstance(profile, str):
                errors.append(
                    f"promotion.gates[{index}]: requires evidence_step and profile"
                )
                continue
            gate_thresholds = gate_entry.get("thresholds") or {}
            if not isinstance(gate_thresholds, Mapping):
                errors.append(f"promotion.gates[{index}]: thresholds must be a mapping")
                gate_thresholds = {}
            gates.append(
                PromotionGate(
                    evidence_step=evidence_step,
                    profile=profile,
                    thresholds=dict(gate_thresholds),
                )
            )
        target = entry.get("target_metric_profile")
        if target is not None and not isinstance(target, str):
            errors.append("promotion.target_metric_profile must be a string")
            target = None
        promotion = Promotion(
            artifact_step=artifact_step,
            gates=tuple(gates),
            target_metric_profile=target,
        )

    if errors:
        raise RecipeError(errors)
    return Recipe(
        name=name,
        description=str(raw.get("description") or ""),
        external_inputs=external_inputs,
        steps=tuple(steps),
        promotion=promotion,
        strict_inputs=strict_inputs,
        source_path=source_path,
    )


def load_recipe(path: str | Path) -> Recipe:
    recipe_path = Path(path)
    raw = yaml.safe_load(recipe_path.read_text())
    if not isinstance(raw, Mapping):
        raise RecipeError([f"{recipe_path}: recipe must be a YAML mapping"])
    return parse_recipe(raw, source_path=str(recipe_path))


def validate_recipe(recipe: Recipe, registry: Mapping[str, Any]) -> list[str]:
    """Validate graph structure against an op registry.

    ``registry`` maps op name -> OpDef (see ``mlx_vq.build.ops``). Returns a
    list of error strings; empty means valid.
    """

    from mlx_vq.build.gate_profiles import GATE_PROFILES, resolve_gate_thresholds

    errors: list[str] = []
    seen: dict[str, StepSpec] = {}
    for spec in recipe.steps:
        if spec.id in seen:
            errors.append(f"duplicate step id: {spec.id}")
            continue

        op_def = registry.get(spec.op)
        if op_def is None:
            errors.append(f"step {spec.id}: unknown op {spec.op!r}")
            seen[spec.id] = spec
            continue
        if spec.step_class not in op_def.allowed_classes:
            errors.append(
                f"step {spec.id}: op {spec.op!r} does not allow class "
                f"{spec.step_class!r} (allowed: {sorted(op_def.allowed_classes)})"
            )

        for input_name, ref in spec.inputs.items():
            if ref.kind == "external":
                if ref.name not in recipe.external_inputs:
                    errors.append(
                        f"step {spec.id}: input {input_name!r} references unknown "
                        f"external input {ref.name!r}"
                    )
                continue
            producer = seen.get(ref.name)
            if producer is None:
                errors.append(
                    f"step {spec.id}: input {input_name!r} references step "
                    f"{ref.name!r} which is not declared earlier in the recipe"
                )
                continue
            producer_op = registry.get(producer.op)
            if producer_op is not None and ref.output not in producer_op.output_names(
                producer.params
            ):
                errors.append(
                    f"step {spec.id}: input {input_name!r} references output "
                    f"{ref.output!r} which op {producer.op!r} does not declare "
                    f"(declared: {sorted(producer_op.output_names(producer.params))})"
                )
            allowed = _ALLOWED_INPUT_CLASSES[spec.step_class]
            if producer.step_class not in allowed:
                errors.append(
                    f"step {spec.id} (class {spec.step_class}) may not consume "
                    f"output of step {producer.id} (class {producer.step_class}): "
                    f"diagnostic outputs never feed promotable or verify lineage"
                )

        missing = [
            name for name in op_def.required_inputs if name not in spec.inputs
        ]
        if missing:
            errors.append(
                f"step {spec.id}: op {spec.op!r} requires inputs {missing}"
            )
        unexpected = [
            name
            for name in spec.inputs
            if name not in op_def.required_inputs and name not in op_def.optional_inputs
        ]
        if unexpected:
            errors.append(
                f"step {spec.id}: op {spec.op!r} does not accept inputs {unexpected}"
            )
        unknown_params = [
            name for name in spec.params if name not in op_def.allowed_params
        ]
        if unknown_params:
            errors.append(
                f"step {spec.id}: op {spec.op!r} does not accept params {unknown_params}"
            )
        missing_params = [
            name for name in op_def.required_params if name not in spec.params
        ]
        if missing_params:
            errors.append(
                f"step {spec.id}: op {spec.op!r} requires params {missing_params}"
            )

        if (
            any(code != 0 for code in op_def.accepted_returncodes)
            and spec.gate is None
        ):
            errors.append(
                f"step {spec.id}: op {spec.op!r} accepts a nonzero blocked "
                "return code and therefore requires a structured gate"
            )
        if (
            op_def.required_gate_profile is not None
            and (
                spec.gate is None
                or spec.gate.profile != op_def.required_gate_profile
            )
        ):
            errors.append(
                f"step {spec.id}: op {spec.op!r} requires gate profile "
                f"{op_def.required_gate_profile!r}"
            )

        if spec.gate is not None:
            if spec.gate.profile not in GATE_PROFILES:
                errors.append(
                    f"step {spec.id}: unknown gate profile {spec.gate.profile!r}"
                )
            else:
                try:
                    resolve_gate_thresholds(spec.gate.profile, spec.gate.thresholds)
                except KeyError as error:
                    errors.append(f"step {spec.id}: {error.args[0]}")

        seen[spec.id] = spec

    if recipe.promotion is not None:
        artifact_step = seen.get(recipe.promotion.artifact_step)
        if artifact_step is None:
            errors.append(
                f"promotion.artifact_step {recipe.promotion.artifact_step!r} is not a declared step"
            )
        elif artifact_step.step_class != "promotable":
            errors.append(
                f"promotion.artifact_step {artifact_step.id!r} must be promotable, "
                f"got class {artifact_step.step_class!r}"
            )
        for gate in recipe.promotion.gates:
            evidence_step = seen.get(gate.evidence_step)
            if evidence_step is None:
                errors.append(
                    f"promotion gate references unknown step {gate.evidence_step!r}"
                )
            elif evidence_step.step_class != "verify":
                errors.append(
                    f"promotion gate step {gate.evidence_step!r} must be class verify, "
                    f"got {evidence_step.step_class!r}"
                )
            if gate.profile not in GATE_PROFILES:
                errors.append(
                    f"promotion gate on {gate.evidence_step!r}: unknown profile {gate.profile!r}"
                )
            else:
                try:
                    resolve_gate_thresholds(gate.profile, gate.thresholds)
                except KeyError as error:
                    errors.append(
                        f"promotion gate on {gate.evidence_step!r}: {error.args[0]}"
                    )
    return errors
