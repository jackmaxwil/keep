"""``keep`` CLI: validate and build declarative recipes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from keep.build.ops import REGISTRY
from keep.build.recipe import (
    Recipe,
    RecipeError,
    load_recipe,
    parse_recipe,
    validate_recipe,
)
from keep.build.runner import plan_recipe, render_dry_run


def _run_optional_rc_diff(
    *,
    build_root: Path,
    accepted_rc_summary: Path | None,
    rc_diff_out: Path | None,
) -> int:
    if accepted_rc_summary is None:
        return 0

    import json

    from keep.build.rc_diff import compare_promotion_to_rc, load_json

    promotion_path = build_root / "promotion-result.json"
    try:
        promotion = load_json(promotion_path)
        accepted = load_json(accepted_rc_summary)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: rc diff failed: {error}", file=sys.stderr)
        return 1

    diff = compare_promotion_to_rc(promotion, accepted)
    output_path = rc_diff_out or (build_root / "rc-diff.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diff, indent=2, sort_keys=True) + "\n")
    print(f"rc diff written to {output_path}")
    if not diff["passed"]:
        print("rc diff: mismatch", file=sys.stderr)
        for mismatch in diff["mismatches"]:
            print(
                "  {path}: promotion={promotion!r} accepted={accepted!r}".format(
                    **mismatch
                ),
                file=sys.stderr,
            )
        return 1
    print("rc diff: pass")
    return 0


def _resolve_repair_from_step(recipe: Recipe, step_id: str) -> str:
    try:
        requested = recipe.step(step_id)
    except KeyError as error:
        raise ValueError(f"unknown repair source step {step_id!r}") from error
    if requested.op == "eval-repair":
        return requested.id

    matches: list[str] = []
    for spec in recipe.steps:
        if spec.op != "eval-repair":
            continue
        existing = spec.inputs.get("existing_evidence")
        if (
            existing is not None
            and existing.kind == "step"
            and existing.name == step_id
            and existing.output == "evidence"
        ):
            matches.append(spec.id)
    if not matches:
        raise ValueError(f"no eval-repair step consumes evidence from {step_id!r}")
    if len(matches) > 1:
        raise ValueError(
            f"multiple eval-repair steps consume evidence from {step_id!r}: {matches}"
        )
    return matches[0]


def _load_cli_recipe(path: str | Path) -> Recipe:
    from keep.build.highlevel import compile_high_level, is_high_level_recipe

    if is_high_level_recipe(path):
        return parse_recipe(compile_high_level(path), source_path=str(path))
    return load_recipe(path)


def _load_build_recipe(path: str | Path) -> tuple[Recipe, str | None]:
    from keep.build.highlevel import compile_high_level, is_high_level_recipe

    if is_high_level_recipe(path):
        raw = compile_high_level(path)
        build_root = raw.get("build_root")
        return parse_recipe(raw, source_path=str(path)), (
            str(build_root) if isinstance(build_root, str) and build_root else None
        )
    from keep.build.plan_next import load_merged_recipe

    return load_merged_recipe(path), None


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        recipe = _load_cli_recipe(args.recipe)
    except RecipeError as error:
        for message in error.errors:
            print(f"error: {message}", file=sys.stderr)
        return 1
    errors = validate_recipe(recipe, REGISTRY)
    if errors:
        for message in errors:
            print(f"error: {message}", file=sys.stderr)
        return 1
    print(f"ok: {recipe.name} ({len(recipe.steps)} steps)")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    if args.auto_extend:
        from keep.build.plan_next import auto_extend

        accepted = auto_extend(
            Path(args.recipe),
            iterations=args.auto_extend,
            build_root=Path(args.build_root) if args.build_root else None,
        )
        print(f"auto-extend: {accepted}/{args.auto_extend} proposals accepted")
        return 0

    try:
        recipe, default_build_root = _load_build_recipe(args.recipe)
        plan = plan_recipe(
            recipe,
            build_root=Path(args.build_root or default_build_root)
            if (args.build_root or default_build_root)
            else None,
            no_hash=args.no_hash,
            strict_inputs=args.strict_inputs,
        )
    except RecipeError as error:
        for message in error.errors:
            print(f"error: {message}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(render_dry_run(plan))
        return 0
    from keep.build.executor import execute_plan

    from_step = args.from_step
    if args.repair_from:
        if from_step:
            print("error: --from and --repair-from are mutually exclusive", file=sys.stderr)
            return 1
        try:
            from_step = _resolve_repair_from_step(recipe, args.repair_from)
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"repair-from: {args.repair_from} -> {from_step}")

    build_rc = execute_plan(
        plan,
        from_step=from_step,
        until_step=args.until_step,
        regate=args.regate,
        continue_on_gate_fail=args.continue_on_gate_fail,
    )
    diff_rc = _run_optional_rc_diff(
        build_root=plan.build_root,
        accepted_rc_summary=Path(args.accepted_rc_summary)
        if args.accepted_rc_summary
        else None,
        rc_diff_out=Path(args.rc_diff_out) if args.rc_diff_out else None,
    )
    return build_rc or diff_rc


def _cmd_compile(args: argparse.Namespace) -> int:
    from keep.build.highlevel import compile_high_level, is_high_level_recipe

    try:
        if is_high_level_recipe(args.recipe):
            raw = compile_high_level(args.recipe)
        else:
            raw = yaml.safe_load(Path(args.recipe).read_text())
            if not isinstance(raw, dict):
                raise RecipeError([f"{args.recipe}: recipe must be a YAML mapping"])
    except (OSError, RecipeError) as error:
        if isinstance(error, RecipeError):
            messages = error.errors
        else:
            messages = [str(error)]
        for message in messages:
            print(f"error: {message}", file=sys.stderr)
        return 1

    text = yaml.safe_dump(raw, sort_keys=False)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        print(f"compiled recipe written to {output}")
    else:
        print(text, end="")
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    from keep.build.promote import PromotionError, promote_step

    try:
        recipe = load_recipe(args.recipe)
        plan = plan_recipe(
            recipe,
            build_root=Path(args.build_root) if args.build_root else None,
        )
        target = promote_step(
            plan,
            step_id=args.step,
            published_name=args.as_name,
            artifacts_root=Path(args.artifacts_root),
        )
    except (RecipeError, PromotionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"promoted {args.step} -> {target}")
    return 0


def _format_plan_next_row_summary(rows: list[dict]) -> str:
    formatted: list[str] = []
    for row in rows:
        if "domain" in row and "row_index" in row:
            formatted.append(f"{row['domain']}:{row['row_index']}")
            continue
        if "target_key" in row:
            parts = [str(row["target_key"])]
            if row.get("layer") is not None:
                parts.append(f"layer{row['layer']}")
            if row.get("projection") is not None:
                parts.append(str(row["projection"]))
            if row.get("expert") is not None:
                parts.append(f"expert{row['expert']}")
            if row.get("route_rank") is not None:
                parts.append(f"route{row['route_rank']}")
            formatted.append(" ".join(parts))
            continue
        formatted.append(str(row))
    return ", ".join(formatted)


def _cmd_plan_next(args: argparse.Namespace) -> int:
    from keep.build.plan_next import PlanNextError, propose_next_step

    try:
        recipe = load_recipe(args.recipe)
        proposal = propose_next_step(
            recipe,
            build_root=Path(args.build_root) if args.build_root else None,
            no_hash=args.no_hash,
        )
    except (RecipeError, PlanNextError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    lines = [
        f"# proposed from evidence of step {proposal.base_step_id}",
        "# selected rows: " + _format_plan_next_row_summary(proposal.train_rows),
        "# resolved train argv:",
        "#   " + " ".join(proposal.resolved_argv),
        "# append the steps below to the recipe (or an .autogen.yaml overlay):",
        proposal.yaml_fragment(),
    ]
    output = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(output + "\n")
        print(f"proposal written to {args.out}")
    else:
        print(output)
    return 0


def _cmd_rc_diff(args: argparse.Namespace) -> int:
    import json

    from keep.build.rc_diff import compare_promotion_to_rc, load_json

    try:
        promotion = load_json(args.promotion_result)
        accepted = load_json(args.accepted_rc_summary)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    diff = compare_promotion_to_rc(promotion, accepted)
    output = json.dumps(diff, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(output)
        print(f"rc diff written to {args.out}")
    else:
        print(output, end="")
    return 0 if diff["passed"] else 1


def _cmd_recipe_lineage(args: argparse.Namespace) -> int:
    import yaml

    from keep.build.lineage import LineageError, build_legacy_lineage_recipe

    try:
        raw = build_legacy_lineage_recipe(
            Path(args.artifact_dir),
            name=args.name,
            teacher_report=Path(args.teacher_report) if args.teacher_report else None,
            teacher_selection=Path(args.teacher_selection)
            if args.teacher_selection
            else None,
            base_revision=args.base_revision,
            include_cache_preflight=args.include_cache_preflight,
            max_depth=args.max_depth,
        )
    except (OSError, ValueError, LineageError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    text = yaml.safe_dump(raw, sort_keys=False)
    for warning in raw.get("lineage", {}).get("warnings", []):
        print(f"warning: {warning}", file=sys.stderr)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        print(f"lineage recipe written to {output}")
    else:
        print(text, end="")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from ramp.models.profiles import (
        ProfileError,
        get_profile,
        list_profiles,
        profile_to_json,
    )

    if args.models_command == "show":
        try:
            print(profile_to_json(get_profile(args.name)), end="")
        except ProfileError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        return 0

    try:
        names = list_profiles()
        profiles = [get_profile(name) for name in names]
    except ProfileError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for profile in profiles:
        revision = profile.revision or "main"
        print(f"{profile.name}\t{profile.architecture}\t{profile.hf_model_id}@{revision}")
    return 0


def _cmd_get(args: argparse.Namespace) -> int:
    from keep.build import get_model

    if args.download_workers <= 0:
        print("error: --download-workers must be positive", file=sys.stderr)
        return 1
    request = get_model.resolve_model_request(
        args.model,
        revision=args.revision,
        check_only=args.check_only,
        download_workers=args.download_workers,
    )
    readiness = get_model.stage_model(request)
    print(get_model.format_readiness(readiness), end="")
    if args.check_only and not readiness.ready:
        return 1
    return 0 if readiness.ready else 1


def _cmd_doctor(args: argparse.Namespace) -> int:
    from keep.build import doctor

    return doctor.run(model=args.model, as_json=args.json)


def _cmd_report(args: argparse.Namespace) -> int:
    from keep.build import report

    if args.html is not None:
        args.format = "html"
        args.output = args.html
    else:
        args.format = "terminal"
        args.output = None
    return report.run(args)


def _cmd_model_card(args: argparse.Namespace) -> int:
    from keep.build import modelcard

    return modelcard.run(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keep",
        description="KEEP declarative recipe builder",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    report = subparsers.add_parser("report", help="render a build report")
    report.add_argument("source", type=Path, help="build run directory or evidence JSON file")
    report.add_argument("--html", type=Path, help="write a self-contained HTML report")
    report.set_defaults(func=_cmd_report)

    model_card = subparsers.add_parser("model-card", help="render an evidence-bound model card")
    from keep.build import modelcard

    modelcard.configure_parser(model_card)
    model_card.set_defaults(func=_cmd_model_card)

    validate = subparsers.add_parser("validate", help="validate a recipe file")
    validate.add_argument("recipe")
    validate.set_defaults(func=_cmd_validate)

    compile_cmd = subparsers.add_parser(
        "compile", help="compile a high-level recipe to low-level YAML"
    )
    compile_cmd.add_argument("recipe")
    compile_cmd.add_argument("-o", "--out", default=None)
    compile_cmd.set_defaults(func=_cmd_compile)

    build = subparsers.add_parser("build", help="build a recipe")
    build.add_argument("recipe")
    build.add_argument("--dry-run", action="store_true")
    build.add_argument(
        "--no-hash",
        action="store_true",
        help="skip external input hashing (dry-run without artifacts present)",
    )
    build.add_argument("--strict-inputs", action="store_true")
    build.add_argument("--from", dest="from_step", default=None, metavar="STEP")
    build.add_argument(
        "--repair-from",
        default=None,
        metavar="STEP",
        help=(
            "force the eval-repair step that consumes STEP/evidence, preserving "
            "the upstream raw split evidence for row-scoped retry"
        ),
    )
    build.add_argument("--until", dest="until_step", default=None, metavar="STEP")
    build.add_argument("--regate", action="store_true")
    build.add_argument(
        "--continue-on-gate-fail",
        action="store_true",
        help=(
            "keep executing later steps after a gate failure so a supervised "
            "run can collect the full evidence packet; exits nonzero if any "
            "gate failed"
        ),
    )
    build.add_argument(
        "--accepted-rc-summary",
        default=None,
        help="after the build, diff promotion-result.json against this rc-summary.json",
    )
    build.add_argument(
        "--rc-diff-out",
        default=None,
        help="where to write the build-time RC diff (default: <build-root>/rc-diff.json)",
    )
    build.add_argument(
        "--auto-extend",
        type=int,
        default=0,
        metavar="N",
        help="propose, execute, and gate up to N recipe extensions "
        "(accepted steps land in <recipe>.autogen.yaml)",
    )
    build.add_argument("--build-root", default=None)
    build.set_defaults(func=_cmd_build)

    promote = subparsers.add_parser(
        "promote", help="assign a docs/naming.md immutable name to a completed step"
    )
    promote.add_argument("recipe")
    promote.add_argument("--step", required=True)
    promote.add_argument("--as", dest="as_name", required=True, metavar="NAME")
    promote.add_argument("--artifacts-root", default="artifacts")
    promote.add_argument("--build-root", default=None)
    promote.set_defaults(func=_cmd_promote)

    plan_next = subparsers.add_parser(
        "plan-next",
        help="propose the next recipe step from recorded eval evidence",
    )
    plan_next.add_argument("recipe")
    plan_next.add_argument("--out", default=None)
    plan_next.add_argument("--no-hash", action="store_true")
    plan_next.add_argument("--build-root", default=None)
    plan_next.set_defaults(func=_cmd_plan_next)

    rc_diff = subparsers.add_parser(
        "rc-diff",
        help="compare recipe promotion-result.json with an accepted rc-summary.json",
    )
    rc_diff.add_argument("promotion_result")
    rc_diff.add_argument("--accepted-rc-summary", required=True)
    rc_diff.add_argument("--out", default=None)
    rc_diff.set_defaults(func=_cmd_rc_diff)

    recipe_lineage = subparsers.add_parser(
        "recipe-lineage",
        help="emit a recipe from legacy continuous artifact manifests",
    )
    recipe_lineage.add_argument("artifact_dir")
    recipe_lineage.add_argument("--name", default=None)
    recipe_lineage.add_argument("--teacher-report", default=None)
    recipe_lineage.add_argument("--teacher-selection", default=None)
    recipe_lineage.add_argument("--base-revision", default=None)
    recipe_lineage.add_argument(
        "--include-cache-preflight",
        action="store_true",
        help=(
            "prepend diagnostic source-view, JACCL hostfile, and cleanroom "
            "teacher-cache preflight steps to the generated lineage recipe"
        ),
    )
    recipe_lineage.add_argument("--max-depth", type=int, default=128)
    recipe_lineage.add_argument("--out", default=None)
    recipe_lineage.set_defaults(func=_cmd_recipe_lineage)

    models = subparsers.add_parser("models", help="list model profiles")
    models_subparsers = models.add_subparsers(dest="models_command")
    show = models_subparsers.add_parser("show", help="dump a resolved model profile")
    show.add_argument("name")
    show.set_defaults(func=_cmd_models)
    models.set_defaults(func=_cmd_models, models_command="list")

    get = subparsers.add_parser(
        "get",
        help="stage a HuggingFace model snapshot for the KEEP pipeline",
    )
    get.add_argument("model", help="model profile name or raw HuggingFace model id")
    get.add_argument(
        "--revision",
        default=None,
        help="HuggingFace revision for raw ids; profile revisions are used by default",
    )
    get.add_argument(
        "--check-only",
        action="store_true",
        help="inspect the local HF cache without downloading missing files",
    )
    get.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="parallel workers for missing shard downloads",
    )
    get.set_defaults(func=_cmd_get)

    doctor = subparsers.add_parser(
        "doctor",
        help="run an offline environment, memory, and model-cache preflight",
    )
    doctor.add_argument(
        "--model",
        default=None,
        help="model profile name or raw HuggingFace model id to inspect offline",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable report",
    )
    doctor.set_defaults(func=_cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "no_hash", False) and not getattr(args, "dry_run", False):
        parser.error("--no-hash is only valid with --dry-run")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
