from __future__ import annotations

import argparse
import json

from mlx_vq.validate.qwen_vq import validate_qwen_vq


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Qwen generation through QuantizedVQLinear.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--target", default="model.layers.0.self_attn.q_proj")
    parser.add_argument("--max-linears", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()

    result = validate_qwen_vq(
        args.model,
        prompt=args.prompt,
        target=args.target,
        max_linears=args.max_linears,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
