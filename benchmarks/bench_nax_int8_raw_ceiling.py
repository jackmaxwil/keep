from __future__ import annotations

import argparse
import json

from mlx_vq.benchmark.glm45_air import append_jsonl
from mlx_vq.benchmark.projection_kernels import run_nax_int8_raw_ceiling


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate INT8-NAX work with a 64x64x64 native E8 INT8 vs FP16 ceiling probe."
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    record = run_nax_int8_raw_ceiling(
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
    )
    if args.append_jsonl:
        append_jsonl(args.append_jsonl, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
