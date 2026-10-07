from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from keep.convert.inspect_hf import GLM52_MODEL_ID, fetch_hf_config
from keep.validate.glm52_vq import validate_glm52_vq


def _load_config(*, model_id: str, revision: str, config_path: str | None) -> dict:
    if config_path is not None:
        return json.loads(Path(config_path).read_text())
    try:
        cached_config = hf_hub_download(
            model_id,
            "config.json",
            revision=revision,
            local_files_only=True,
        )
        return json.loads(Path(cached_config).read_text())
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
    parser = argparse.ArgumentParser(description="Validate the GLM-5.2 routed VQ artifact against source tensors.")
    parser.add_argument("--model-id", default=GLM52_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--source-dir")
    parser.add_argument("--artifact-dir", default="artifacts/glm-5.2-vq")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--strict-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-artifact-cosine", type=float, default=0.999)
    parser.add_argument("--min-source-weighted-cosine", type=float)
    args = parser.parse_args()

    source_dir = _resolve_source_dir(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
    )
    index_path = Path(args.index_path) if args.index_path is not None else source_dir / "model.safetensors.index.json"
    config = _load_config(
        model_id=args.model_id,
        revision=args.revision,
        config_path=args.config_path,
    )

    result = validate_glm52_vq(
        config,
        source_dir=source_dir,
        index_path=index_path,
        artifact_dir=args.artifact_dir,
        model_id=args.model_id,
        layer=args.layer,
        tokens=args.tokens,
        seed=args.seed,
        input_scale=args.input_scale,
        strict_config=args.strict_config,
    )
    payload = result.to_dict()
    payload["thresholds"] = {
        "min_artifact_cosine": args.min_artifact_cosine,
        "min_source_weighted_cosine": args.min_source_weighted_cosine,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    failures = []
    for metric_name in (
        "artifact_gate_proj",
        "artifact_up_proj",
        "artifact_routed_glu",
        "artifact_weighted_routed",
    ):
        cosine = result.metrics[metric_name].cosine
        if cosine < args.min_artifact_cosine:
            failures.append(f"{metric_name} cosine {cosine:.6f} < {args.min_artifact_cosine:.6f}")
    if args.min_source_weighted_cosine is not None:
        cosine = result.metrics["source_weighted_routed"].cosine
        if cosine < args.min_source_weighted_cosine:
            failures.append(
                "source_weighted_routed cosine "
                f"{cosine:.6f} < {args.min_source_weighted_cosine:.6f}"
            )
    if failures:
        raise SystemExit("validation threshold failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
