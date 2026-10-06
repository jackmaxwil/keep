"""Run the canonical checkpointable schema-v3 Teich adapter candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_callable(spec: str):  # noqa: ANN202
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("baseline loader must use module:function syntax")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise TypeError("baseline loader is not callable")
    return function


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache-dir", type=Path, required=True)
    parser.add_argument("--prompt-pack", type=Path, required=True)
    parser.add_argument("--frozen-prompt-pack", type=Path, required=True)
    parser.add_argument("--expected-teacher-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-s3-prefix")
    parser.add_argument("--boundary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument(
        "--validated-model-loader",
        default="mlx_vq.quality.glm52_training_baseline:accepted_baseline_provider",
    )
    parser.add_argument("--validation-window-count", type=int, default=32)
    parser.add_argument("--layer", type=int, default=77)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--preflight-steps",
        type=int,
        choices=range(1, 21),
        default=20,
    )
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()

    from mlx_vq.quality.glm52_adapter_training import (
        AdapterTrainingConfig,
        GLM52FixedValidationRunner,
        ValidatedGLM52TrainingBaseline,
        attest_glm52_projection_inventory,
        prepare_glm52_teich_teacher_rows,
        run_low_rank_training,
        save_authenticated_glm52_adapter,
        target_glm52_projection,
    )
    from mlx_vq.quality.glm52_teich_training_cache import MANIFEST_FILENAME
    from mlx_vq.quality.glm52_teich_training_campaign import (
        TrainingCheckpointConfig,
        TrainingCheckpointStore,
        TrainingIdentity,
        build_fixed_split_windows,
        build_training_schedule,
    )

    manifest = json.loads(
        (args.teacher_cache_dir / MANIFEST_FILENAME).read_text()
    )
    entries = manifest["shards"]
    schedule = build_training_schedule(entries, window_size=64, seed=20260712)
    train_rows = prepare_glm52_teich_teacher_rows(
        teacher_cache_dir=args.teacher_cache_dir,
        prompt_pack_path=args.prompt_pack,
        frozen_prompt_pack_path=args.frozen_prompt_pack,
        split="train",
        expected_manifest_sha256=args.expected_teacher_manifest_sha256,
        allow_non_release_teacher_cache=True,
        lazy=True,
    )
    validation_rows = (
        []
        if args.preflight_only
        else prepare_glm52_teich_teacher_rows(
            teacher_cache_dir=args.teacher_cache_dir,
            prompt_pack_path=args.prompt_pack,
            frozen_prompt_pack_path=args.frozen_prompt_pack,
            split="validation",
            expected_manifest_sha256=args.expected_teacher_manifest_sha256,
            allow_non_release_teacher_cache=True,
            lazy=True,
        )
    )
    baseline = _load_callable(args.validated_model_loader)(args)
    if not isinstance(baseline, ValidatedGLM52TrainingBaseline):
        raise TypeError("baseline loader did not return ValidatedGLM52TrainingBaseline")
    model = baseline.model
    config = AdapterTrainingConfig(
        layers=(args.layer,),
        projections=("gate_proj", "up_proj", "down_proj"),
        rank=4,
        steps=(
            min(args.preflight_steps, len(schedule.windows))
            if args.preflight_only
            else len(schedule.windows)
        ),
        learning_rate=0.2,
        low_rank_init_scale=0.02,
        grad_clip_norm=1.0,
        target_nll_weight=0.0,
        teacher_top1_margin_weight=0.0,
        tail_kld_weight=0.0,
        seed=20260712,
        topk=2048,
        cka_enabled=True,
        cka_layers=(77,),
        router_kl_weight=1.0,
        router_entropy_beta=0.01,
        mc_expert_explore=True,
    )
    projection_map = {
        (layer, projection): target_glm52_projection(
            model, layer=layer, projection=projection
        )
        for layer in config.layers
        for projection in config.projections
    }
    checkpoint_config = TrainingCheckpointConfig(
        local_checkpoint_dir=args.checkpoint_dir,
        s3_checkpoint_prefix=args.checkpoint_s3_prefix,
        stop_file=args.stop_file,
    )
    training_identity = TrainingIdentity(
        run_id=args.run_id,
        baseline_sha256=baseline.candidate_identity_sha256,
        teacher_manifest_sha256=args.expected_teacher_manifest_sha256,
        schedule_sha256=schedule.sha256,
        training_config_sha256=_sha(asdict(config)),
    )
    store = TrainingCheckpointStore(
        args.checkpoint_dir, identity=training_identity
    )
    validation_runner = None
    if not args.preflight_only:
        validation_windows = build_fixed_split_windows(
            entries,
            split="validation",
            seed=20260712,
            max_windows=args.validation_window_count,
        )
        validation_runner = GLM52FixedValidationRunner(
            model=model,
            rows=validation_rows,
            windows=validation_windows,
            config=config,
            boundary_dir=args.boundary_dir / "validation",
        )

    if args.stop_file is not None:
        def request_stop(_signum, _frame):  # noqa: ANN001
            args.stop_file.parent.mkdir(parents=True, exist_ok=True)
            args.stop_file.touch()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

    def sync_checkpoints(root: Path, _step: int, _force: bool) -> None:
        if not args.checkpoint_s3_prefix:
            return
        subprocess.run(
            [
                str(REPO_ROOT / "aws/glm52-gpu/scripts/sync_checkpoint_tree.sh"),
                str(root),
                args.checkpoint_s3_prefix,
            ],
            check=True,
        )

    os.environ["GLM52_TRAIN_BOUNDARY_DISK"] = "1"
    reset_peak = getattr(mx, "reset_peak_memory", None)
    if callable(reset_peak):
        reset_peak()
    started = time.monotonic()
    result = run_low_rank_training(
        model=model,
        rows=train_rows,
        projection_map=projection_map,
        config=config,
        training_schedule=schedule,
        checkpoint_store=store,
        checkpoint_config=checkpoint_config,
        resume=args.resume,
        persistent_boundary_dir=args.boundary_dir / "train",
        validation_fn=validation_runner,
        checkpoint_sync=sync_checkpoints,
    )
    elapsed = time.monotonic() - started
    mx.eval(result.losses, *result.params.values())
    if args.preflight_only:
        peak_getter = getattr(mx, "get_peak_memory", None)
        peak_gib = (
            float(peak_getter()) / 1024**3 if callable(peak_getter) else float("inf")
        )
        payload = {
            "record_type": "glm52_teich_real_gpu_training_preflight_v1",
            "sample_steps": len(result.losses),
            "sample_seconds": elapsed,
            "peak_gpu_gib": peak_gib,
            "memory_gate_pass": peak_gib < 70.0,
        }
        if args.preflight_output is not None:
            args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
            args.preflight_output.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            )
        print(json.dumps(payload, sort_keys=True))
        return (
            0
            if peak_gib < 70.0
            and len(result.losses) == args.preflight_steps
            else 2
        )
    if len(result.losses) < len(schedule.windows):
        print(
            json.dumps(
                {
                    "status": "checkpointed-stop",
                    "completed_schedule_index": len(result.losses),
                    "schedule_windows": len(schedule.windows),
                },
                sort_keys=True,
            )
        )
        return 75
    best_params = {
        name: mx.array(value, dtype=mx.float32)
        for name, value in store.load_best_params().items()
    }
    mx.eval(*best_params.values())
    result = replace(result, params=best_params)
    saved = save_authenticated_glm52_adapter(
        args.output_dir,
        params=result.params,
        projection_map=projection_map,
        parent_candidate_identity_sha256=baseline.candidate_identity_sha256,
        teacher_manifest_body_sha256=manifest["manifest_body_sha256"],
        training_config=config,
        selection_rows=train_rows,
        release_eligible=False,
        projection_inventory_attestation=attest_glm52_projection_inventory(
            projection_map
        ),
    )
    print(
        json.dumps(
            {
                "status": "candidate-ready-for-evaluation",
                "candidate_identity_sha256": saved.candidate_identity_sha256,
                "release_eligible": False,
                "automatically_promoted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
