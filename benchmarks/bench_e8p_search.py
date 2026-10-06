from __future__ import annotations

import argparse
import json
import time

import numpy as np


def project_worst8_hours(
    codewords_per_second: float,
    *,
    experts: int = 168,
    out_dim: int = 2048,
    in_dim: int = 6144,
    passes: int = 4,
    groups: int = 24,
) -> float:
    cw_per_group = out_dim * (in_dim // 8) * experts * passes
    total = cw_per_group * groups
    return total / codewords_per_second / 3600.0


def acceptance_gate(hours: float, *, max_hours: float = 12.0) -> bool:
    return hours <= max_hours


def measure_backend(backend: str, *, n_codewords: int, seed: int) -> dict:  # pragma: no cover - host timing
    from mlx_vq.quant.rtn import nearest_e8p_codes_diagonal_hessian

    rng = np.random.default_rng(seed)
    vecs = (rng.standard_normal((n_codewords, 8)) * 0.6).astype(np.float32)
    diag = np.abs(rng.standard_normal(8)).astype(np.float32) + 0.05
    nearest_e8p_codes_diagonal_hessian(vecs[:64], diag, backend=backend)  # warm-up
    start = time.perf_counter()
    nearest_e8p_codes_diagonal_hessian(vecs, diag, backend=backend)
    seconds = time.perf_counter() - start
    return {"backend": backend, "seconds": seconds, "codewords_per_second": n_codewords / seconds}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - host path
    parser = argparse.ArgumentParser(prog="bench_e8p_search")
    parser.add_argument("--backend", default="metal", choices=["metal", "numpy"])
    parser.add_argument("--codewords", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    result = measure_backend(args.backend, n_codewords=args.codewords, seed=20260711)
    hours = project_worst8_hours(result["codewords_per_second"])
    result["projected_worst8_hours"] = hours
    result["acceptance_gate_pass"] = acceptance_gate(hours)
    print(json.dumps(result, indent=2))
    return 0 if result["acceptance_gate_pass"] else 1
