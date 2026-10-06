from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from mlx_vq.quality.glm52_adapter_training import (
    AdapterTrainingConfig,
    SUPPORTED_PROJECTIONS,
    ValidatedGLM52TrainingBaseline,
    attest_glm52_projection_inventory,
    prepare_glm52_teacher_rows,
    run_low_rank_training,
    save_authenticated_glm52_adapter,
    target_glm52_projection,
)


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list") from error
    if not result:
        raise argparse.ArgumentTypeError("list must not be empty")
    return result


def _csv_projections(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(result) - set(SUPPORTED_PROJECTIONS))
    if not result or unknown:
        raise argparse.ArgumentTypeError(
            "projections must be a comma-separated subset of gate_proj,up_proj,down_proj"
        )
    return result


def _load_callable(spec: str) -> Callable[[argparse.Namespace], ValidatedGLM52TrainingBaseline]:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--validated-model-loader must use module.path:function syntax")
    function = getattr(importlib.import_module(module_name), attribute)
    if not callable(function):
        raise TypeError(f"validated model loader {spec!r} is not callable")
    return function


def _load_validated_baseline(
    loader: Callable[[argparse.Namespace], ValidatedGLM52TrainingBaseline],
    args: argparse.Namespace,
) -> ValidatedGLM52TrainingBaseline:
    baseline = loader(args)
    if not isinstance(baseline, ValidatedGLM52TrainingBaseline):
        raise TypeError(
            "validated model loader must return ValidatedGLM52TrainingBaseline"
        )
    return baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train bounded GLM-5.2 routed-projection low-rank adapters from an authenticated teacher cache."
    )
    parser.add_argument("--teacher-cache-dir", type=Path, required=True)
    parser.add_argument("--prompt-pack", type=Path, required=True)
    parser.add_argument("--expected-teacher-manifest-body-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validated-model-loader",
        required=True,
        help=(
            "module:function returning ValidatedGLM52TrainingBaseline; the provider owns "
            "baseline/recovery validation and binds the model to its candidate identity"
        ),
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        required=True,
        help="mandatory acknowledgement that report and holdout rows are forbidden for tuning",
    )
    parser.add_argument("--layers", type=_csv_ints, required=True)
    parser.add_argument("--projections", type=_csv_projections, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument(
        "--lr",
        type=float,
        default=0.2,
        help="empirically validated value; much smaller values can produce a no-op adapter",
    )
    parser.add_argument(
        "--low-rank-init-scale",
        type=float,
        default=2.0e-2,
        help="empirically validated value; much smaller values can produce a no-op adapter",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--target-nll-weight", type=float, default=0.0)
    parser.add_argument("--teacher-top1-margin-weight", type=float, default=0.0)
    parser.add_argument("--teacher-top1-margin", type=float, default=0.0)
    parser.add_argument("--tail-kld-weight", type=float, default=0.0)
    parser.add_argument(
        "--topk",
        type=int,
        help="enable cached top-K KL; production K must be 2048..8192 and match cache metadata",
    )
    parser.add_argument("--cka-enabled", action="store_true")
    parser.add_argument("--cka-layers", type=_csv_ints, default=())
    parser.add_argument("--router-kl-weight", type=float, default=0.0)
    parser.add_argument("--router-entropy-beta", type=float, default=0.0)
    parser.add_argument(
        "--mc-expert-explore",
        action="store_true",
        help="enable seeded full-expert Monte-Carlo exploration",
    )
    parser.add_argument("--surrogate-output-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--row-indices", type=_csv_ints)
    parser.add_argument("--allow-non-release-teacher-cache", action="store_true")
    parser.add_argument("--ledger-jsonl", type=Path)
    return parser


def _append_ledger(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = prepare_glm52_teacher_rows(
        teacher_cache_dir=args.teacher_cache_dir,
        prompt_pack_path=args.prompt_pack,
        split="selection",
        max_rows=args.max_rows,
        max_positions=args.max_positions,
        row_indices=args.row_indices,
        allow_non_release_teacher_cache=args.allow_non_release_teacher_cache,
        expected_manifest_body_sha256=args.expected_teacher_manifest_body_sha256,
    )
    baseline = _load_validated_baseline(
        _load_callable(args.validated_model_loader),
        args,
    )
    model = baseline.model
    projection_map = {
        (layer, projection): target_glm52_projection(model, layer=layer, projection=projection)
        for layer in args.layers
        for projection in args.projections
    }
    projection_inventory_attestation = attest_glm52_projection_inventory(projection_map)
    config = AdapterTrainingConfig(
        layers=args.layers,
        projections=args.projections,
        rank=args.rank,
        steps=args.steps,
        learning_rate=args.lr,
        low_rank_init_scale=args.low_rank_init_scale,
        grad_clip_norm=args.grad_clip_norm,
        target_nll_weight=args.target_nll_weight,
        teacher_top1_margin_weight=args.teacher_top1_margin_weight,
        teacher_top1_margin=args.teacher_top1_margin,
        tail_kld_weight=args.tail_kld_weight,
        seed=args.seed,
        surrogate_output_chunk_size=args.surrogate_output_chunk_size,
        topk=args.topk,
        cka_enabled=args.cka_enabled,
        cka_layers=args.cka_layers,
        router_kl_weight=args.router_kl_weight,
        router_entropy_beta=args.router_entropy_beta,
        mc_expert_explore=args.mc_expert_explore,
    )
    result = run_low_rank_training(
        model=model,
        rows=rows,
        projection_map=projection_map,
        config=config,
    )
    mx.eval(result.losses)
    saved = save_authenticated_glm52_adapter(
        args.output_dir,
        params=result.params,
        projection_map=projection_map,
        parent_candidate_identity_sha256=baseline.candidate_identity_sha256,
        teacher_manifest_body_sha256=rows[0].teacher_manifest_body_sha256,
        training_config=config,
        selection_rows=rows,
        release_eligible=all(not row.non_release_waiver for row in rows),
        projection_inventory_attestation=projection_inventory_attestation,
    )
    record = {
        "record_type": "glm52_low_rank_adapter_training",
        "selection_only": True,
        "steps": config.steps,
        "layers": list(config.layers),
        "projections": list(config.projections),
        "topk": config.topk,
        "cka_enabled": config.cka_enabled,
        "cka_layers": list(config.cka_layers),
        "router_kl_weight": config.router_kl_weight,
        "router_entropy_beta": config.router_entropy_beta,
        "mc_expert_explore": config.mc_expert_explore,
        "seed": config.seed,
        "final_loss": float(result.losses[-1].item()),
        "adapter_manifest_body_sha256": saved.manifest_body_sha256,
        "adapter_candidate_identity_sha256": saved.candidate_identity_sha256,
        "release_eligible": saved.release_eligible,
    }
    _append_ledger(args.ledger_jsonl, record)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
