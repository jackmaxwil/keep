from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Any

import mlx.core as mx


DTYPES = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}


def _dtype_from_name(name: str) -> mx.Dtype:
    try:
        return DTYPES[name]
    except KeyError as error:
        choices = ", ".join(sorted(DTYPES))
        raise ValueError(f"unsupported dtype {name!r}; choices: {choices}") from error


def _payload_bytes(tokens: int, hidden_size: int, dtype_name: str) -> int:
    bytes_per_element = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }[dtype_name]
    return int(tokens) * int(hidden_size) * bytes_per_element


def _scalar_value(array: mx.array) -> float:
    mx.eval(array)
    return float(array.item())


def _send_recv_roundtrip(
    *,
    rank: int,
    tokens: int,
    hidden_size: int,
    dtype_name: str,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    dtype = _dtype_from_name(dtype_name)
    payload_shape = (tokens, hidden_size)
    payload_bytes = _payload_bytes(tokens, hidden_size, dtype_name)
    timings: list[float] = []
    checksums: list[float] = []

    total_iterations = warmup + iterations
    for iteration in range(total_iterations):
        start = time.perf_counter()
        if rank == 0:
            payload = mx.full(payload_shape, iteration + 1, dtype=dtype)
            sent = mx.distributed.send(payload, dst=1)
            mx.eval(sent)
            ack = mx.distributed.recv((1,), mx.float32, src=1)
            checksum = _scalar_value(ack)
        elif rank == 1:
            received = mx.distributed.recv(payload_shape, dtype, src=0)
            mx.eval(received)
            checksum_array = mx.array([mx.sum(received)], dtype=mx.float32)
            sent = mx.distributed.send(checksum_array, dst=0)
            mx.eval(sent)
            checksum = _scalar_value(checksum_array)
        else:
            checksum = 0.0
        elapsed = time.perf_counter() - start
        if iteration >= warmup and rank in (0, 1):
            timings.append(elapsed)
            checksums.append(checksum)

    if not timings:
        return {
            "role": "idle",
            "payload_bytes": payload_bytes,
            "iteration_count": 0,
        }

    total_seconds = float(sum(timings))
    total_payload_bytes = payload_bytes * len(timings) * 2
    return {
        "role": "sender" if rank == 0 else "receiver",
        "payload_shape": list(payload_shape),
        "payload_bytes": payload_bytes,
        "iteration_count": len(timings),
        "warmup_count": warmup,
        "total_seconds": total_seconds,
        "mean_roundtrip_seconds": total_seconds / len(timings),
        "roundtrip_gbps": (total_payload_bytes * 8.0) / total_seconds / 1e9
        if total_seconds > 0
        else None,
        "first_checksum": checksums[0],
        "last_checksum": checksums[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the JACCL point-to-point tensor path needed by a "
            "rank-split GLM-4.5-Air teacher-cache service."
        )
    )
    parser.add_argument("--backend", default="jaccl", choices=["any", "ring", "jaccl"])
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--require-two-ranks",
        action="store_true",
        help="Exit nonzero unless the launcher created at least two ranks.",
    )
    args = parser.parse_args()

    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    group = mx.distributed.init(strict=args.require_two_ranks, backend=args.backend)
    rank = int(group.rank())
    size = int(group.size())
    if args.require_two_ranks and size < 2:
        raise SystemExit("--require-two-ranks needs at least two distributed ranks")

    result: dict[str, Any] = {
        "backend": args.backend,
        "host": socket.gethostname(),
        "rank": rank,
        "size": size,
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "dtype": args.dtype,
        "status": "ok",
    }

    if size < 2:
        result.update(
            {
                "status": "singleton",
                "payload_bytes": _payload_bytes(args.tokens, args.hidden_size, args.dtype),
                "note": "launcher created one rank; point-to-point send/recv was skipped",
            }
        )
    else:
        result.update(
            _send_recv_roundtrip(
                rank=rank,
                tokens=args.tokens,
                hidden_size=args.hidden_size,
                dtype_name=args.dtype,
                iterations=args.iterations,
                warmup=args.warmup,
            )
        )

    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
