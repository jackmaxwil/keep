from __future__ import annotations

import argparse
import json

from mlx_vq.benchmark.glm45_air import (
    append_jsonl,
    enforce_context_caps,
    format_markdown_summary,
    run_generation_benchmark,
    run_moe_kernel_benchmark,
    run_sparse_residual_microbenchmark,
    scenario_defaults,
    validate_context_gate,
)
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark resident GLM-4.5-Air VQ inference.")
    parser.add_argument(
        "--scenario",
        choices=[
            "short_decode",
            "decode_128",
            "prefill_1k",
            "prefill_4k",
            "prefill",
            "moe_kernel",
            "sparse_residual_micro",
        ],
        default="short_decode",
    )
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--artifact-dir", default="artifacts/glm-4.5-air-vq")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prompt-id")
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--append-jsonl")
    parser.add_argument("--print-markdown", action="store_true")
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument("--use-gather-switch", action="store_true")
    parser.add_argument("--disable-gather-switch", action="store_true")
    parser.add_argument("--allow-larger-context", action="store_true")
    parser.add_argument("--max-mlx-peak-bytes", type=int)
    parser.add_argument("--max-rss-bytes", type=int)
    parser.add_argument("--max-prefill-seconds", type=float)
    parser.add_argument("--moe-tokens", type=int, default=8)
    parser.add_argument("--moe-top-k", type=int, default=8)
    parser.add_argument("--moe-experts", type=int, default=128)
    parser.add_argument("--moe-input-dims", type=int, default=4096)
    parser.add_argument("--moe-output-dims", type=int, default=1408)
    parser.add_argument("--moe-group-size", type=int, default=512)
    parser.add_argument("--sparse-residual-rows", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    defaults = scenario_defaults(args.scenario)
    context_tokens = args.context_tokens
    if context_tokens is None:
        default_context = defaults["context_tokens"]
        context_tokens = None if default_context is None else int(default_context)
    if context_tokens is not None:
        validate_context_gate(
            context_tokens=context_tokens,
            allow_larger_context=args.allow_larger_context,
            max_mlx_peak_bytes=args.max_mlx_peak_bytes,
            max_rss_bytes=args.max_rss_bytes,
            max_prefill_seconds=args.max_prefill_seconds,
        )

    use_gather_switch = True
    if args.disable_gather_switch:
        use_gather_switch = False
    if args.use_gather_switch:
        use_gather_switch = True

    if args.scenario == "moe_kernel":
        record = run_moe_kernel_benchmark(
            model_id=args.model_id,
            artifact_dir=args.artifact_dir,
            tokens=args.moe_tokens,
            top_k=args.moe_top_k,
            experts=args.moe_experts,
            input_dims=args.moe_input_dims,
            output_dims=args.moe_output_dims,
            group_size=args.moe_group_size,
            iterations=args.iterations,
            warmup=args.warmup,
        )
    elif args.scenario == "sparse_residual_micro":
        record = run_sparse_residual_microbenchmark(
            model_id=args.model_id,
            artifact_dir=args.artifact_dir,
            tokens=args.moe_tokens,
            top_k=args.moe_top_k,
            experts=args.moe_experts,
            input_dims=args.moe_input_dims,
            output_dims=args.moe_output_dims,
            group_size=args.moe_group_size,
            sparse_rows=args.sparse_residual_rows,
            iterations=args.iterations,
            warmup=args.warmup,
        )
    else:
        record = run_generation_benchmark(
            scenario=args.scenario,
            model_id=args.model_id,
            revision=args.revision,
            source_dir=args.source_dir,
            config_path=args.config_path,
            index_path=args.index_path,
            artifact_dir=args.artifact_dir,
            prompt=args.prompt,
            prompt_id=args.prompt_id,
            context_tokens=context_tokens,
            max_new_tokens=args.max_new_tokens,
            profile_components=args.profile_components,
            use_gather_switch=use_gather_switch,
        )
        enforce_context_caps(
            record,
            max_mlx_peak_bytes=args.max_mlx_peak_bytes,
            max_rss_bytes=args.max_rss_bytes,
            max_prefill_seconds=args.max_prefill_seconds,
        )

    if args.append_jsonl:
        append_jsonl(args.append_jsonl, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    if args.print_markdown:
        print()
        print(format_markdown_summary(record))


if __name__ == "__main__":
    main()
