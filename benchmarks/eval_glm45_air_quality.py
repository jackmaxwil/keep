from __future__ import annotations

import argparse
import json

from keep.convert.inspect_hf import GLM45_AIR_MODEL_ID
from keep.quality.glm45_air import run_quality_suite
from keep.quality.prompts import get_quality_prompts

DEFAULT_VQ_ARTIFACT_DIR = "artifacts/glm-4.5-air-vq"
DEFAULT_Q2_ARTIFACT_DIR = "artifacts/glm-4.5-air-mlx-q2-routed-g128"


def main() -> None:
    prompt_ids = [prompt.prompt_id for prompt in get_quality_prompts()]
    parser = argparse.ArgumentParser(description="Evaluate resident GLM-4.5-Air VQ quality prompts.")
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument(
        "--artifact-dir",
        help=(
            "Quantized routed artifact directory. Defaults to the VQ artifact for VQ engines "
            "and the routed q2 artifact for --engine mlx_q2_routed_g128."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=[
            "vq_e1_routed",
            "vq_e1_routed_vq_metal",
            "vq_e1_routed_nax_e8",
            "vq_e1_routed_nax_e8p",
            "mlx_q2_routed_g128",
        ],
        default="vq_e1_routed",
    )
    parser.add_argument("--append-jsonl", default="artifacts/quality/glm45-air-baseline.jsonl")
    parser.add_argument("--prompt-id", action="append", choices=prompt_ids)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--nll-max-tokens", type=int, default=128)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir or (
        DEFAULT_Q2_ARTIFACT_DIR
        if args.engine == "mlx_q2_routed_g128"
        else DEFAULT_VQ_ARTIFACT_DIR
    )
    selected = set(args.prompt_id) if args.prompt_id else None
    records = run_quality_suite(
        prompt_ids=selected,
        append_path=args.append_jsonl,
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
        config_path=args.config_path,
        index_path=args.index_path,
        artifact_dir=artifact_dir,
        engine=args.engine,
        top_k=args.top_k,
        nll_max_tokens=args.nll_max_tokens,
    )
    print(
        json.dumps(
            {
                "append_jsonl": args.append_jsonl,
                "records": len(records),
                "prompt_ids": [record["prompt_id"] for record in records],
                "generated_text": {
                    record["prompt_id"]: record["generated_text"] for record in records
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
