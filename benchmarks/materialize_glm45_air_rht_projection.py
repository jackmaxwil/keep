from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download

from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID, fetch_hf_config
from mlx_vq.quality.rht_materialization import materialize_rht_projection_candidate


def _load_config(*, model_id: str, revision: str, config_path: str | None) -> dict[str, Any]:
    if config_path is not None:
        return json.loads(Path(config_path).read_text(encoding="utf-8"))
    try:
        cached_config = hf_hub_download(
            model_id,
            "config.json",
            revision=revision,
            local_files_only=True,
        )
        return json.loads(Path(cached_config).read_text(encoding="utf-8"))
    except Exception:
        return fetch_hf_config(model_id, revision=revision)


def _resolve_source_dir(*, model_id: str, revision: str, source_dir: str | None) -> Path:
    if source_dir is not None:
        return Path(source_dir)
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=["model.safetensors.index.json"],
            local_files_only=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize one linked GLM-4.5-Air RHT projection candidate without mutating the seed artifact."
    )
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--config-path")
    parser.add_argument("--source-dir")
    parser.add_argument("--index-path")
    parser.add_argument("--seed-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--projection", choices=("gate_proj", "up_proj", "down_proj"), required=True)
    parser.add_argument("--rht-seed", required=True)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--code-bits", type=int, choices=(8, 16), default=8)
    parser.add_argument("--scale-estimator", choices=("max_abs", "percentile_99"), default="max_abs")
    parser.add_argument("--expert-workers", type=int, default=1)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--no-strict-config", action="store_true")
    args = parser.parse_args()

    source_dir = _resolve_source_dir(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
    )
    index_path = Path(args.index_path) if args.index_path is not None else source_dir / "model.safetensors.index.json"
    config = _load_config(model_id=args.model_id, revision=args.revision, config_path=args.config_path)
    manifest = materialize_rht_projection_candidate(
        config,
        source_dir=source_dir,
        index_path=index_path,
        seed_artifact_dir=args.seed_artifact_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        layer=args.layer,
        projection=args.projection,
        rht_seed=args.rht_seed,
        group_size=args.group_size,
        code_bits=args.code_bits,
        scale_estimator=args.scale_estimator,
        expert_workers=args.expert_workers,
        strict_config=not args.no_strict_config,
        allow_existing=args.allow_existing,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
