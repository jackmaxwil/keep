from __future__ import annotations

import argparse
import json
from pathlib import Path

from keep.convert.inspect_hf import GLM45_AIR_MODEL_ID
from keep.quality.calibration import run_activation_calibration, summarize_route_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GLM-4.5-Air routed MoE activation calibration stats.")
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--artifact-dir", default="artifacts/glm-4.5-air-vq")
    parser.add_argument("--prompt-set", default="base")
    parser.add_argument("--prompt-id", action="append", dest="prompt_ids")
    parser.add_argument("--layer", action="append", type=int, dest="layers")
    parser.add_argument("--append-jsonl", default="artifacts/quality/glm45-air-q4-activation-stats.jsonl")
    parser.add_argument("--no-down-inputs", action="store_true")
    args = parser.parse_args()

    append_path = Path(args.append_jsonl)
    layers = tuple(args.layers or [1])
    records = run_activation_calibration(
        prompt_ids=set(args.prompt_ids) if args.prompt_ids else None,
        layers=layers,
        append_path=append_path,
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
        config_path=args.config_path,
        index_path=args.index_path,
        artifact_dir=args.artifact_dir,
        collect_down_inputs=not args.no_down_inputs,
        prompt_set=args.prompt_set,
    )
    summary = {
        "append_jsonl": str(append_path),
        "layers": list(layers),
        "records": len(records),
        "prompt_ids": sorted({record["prompt_id"] for record in records}),
        "selected_expert_counts": {
            f"{record['prompt_id']}:layer{record['layer']}": record["selected_expert_count"]
            for record in records
        },
        "route_summary": summarize_route_records(records),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
