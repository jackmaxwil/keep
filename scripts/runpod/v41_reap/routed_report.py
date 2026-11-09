"""Costs and quality of routed-precision variants (``local_eval.py --dynamic``), per variant:

- KL / top-1 / NLL change: domain means from ``summary.json``.
- resident expert memory: every base (tier 1) plus K residuals per layer when cached, otherwise
  every residual too (nothing streamed), in units per layer (1 unit = one 1-bit expert) and GB.
- SSD traffic: residual admissions per token, in MB/token and GB/s at ``--tok-s``.
- unified-memory bytes read per token by routed experts (tier 1 = base, tier 2 = base + residual).
  ``tok_s_bound`` divides ``--bandwidth-gbs`` by those bytes alone; routed experts were about 11%
  of decode traffic on V4-Flash, so it is an upper bound on what expert tiers can buy, not a speed
  estimate.

    python routed_report.py --eval ~/Models/v41-reap-run/evals/step4-routed [--eval ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DIM, INTER = 5120, 2304
# 1-bit E8 code + fp16 group scales: gate/up group 512 (1.031 bpw), down group 256 (1.0625 bpw)
UNIT_BYTES = (2 * DIM * INTER * (1 + 16 / 512) + DIM * INTER * (1 + 16 / 256)) / 8  # ~4.6 MB


def rows(eval_dir: Path, tok_s: float, bandwidth_gbs: float) -> list[dict]:
    summary = json.loads((eval_dir / "summary.json").read_text())
    out = []
    for name, per in summary["variants"].items():
        tiers = per.get("_tiers")
        row = {"variant": f"{eval_dir.name}:{name}", **{k: round(v, 4) for k, v in per["_domain_mean"].items()}}
        if tiers is None:
            out.append(row)
            continue
        c = np.load(eval_dir / name / "tier-counts.npy")  # [L, E, 7]
        n_layers, n_exp = c.shape[:2]
        used = c[:, :, :2].sum() + c[:, :, 5].sum()  # every routed slot
        tokens = used / 6 / n_layers
        k = tiers.get("cache_k")
        units = n_exp + (k if k is not None else n_exp)
        read = (c[:, :, 0].sum() * UNIT_BYTES + c[:, :, 1].sum() * 2 * UNIT_BYTES) / tokens
        ssd = c[:, :, 4].sum() * UNIT_BYTES / tokens if k is not None else 0.0
        row |= {
            "tau": tiers["tau"], "tau_drop": tiers.get("tau_drop", 0.0), "cache_k": k,
            "hi_slots": round(tiers["hi_slot_share"], 3), "drop_slots": round(tiers.get("drop_slot_share", 0.0), 3),
            "units_per_layer": units, "resident_gb": round(units * n_layers * UNIT_BYTES / 1e9, 1),
            "ssd_mb_per_tok": round(ssd / 1e6, 1), "ssd_gbs": round(ssd * tok_s / 1e9, 2),
            "read_gb_per_tok": round(read / 1e9, 2), "tok_s_bound": round(bandwidth_gbs * 1e9 / read, 1),
        }
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", action="append", required=True, type=Path)
    p.add_argument("--tok-s", type=float, default=30.0)
    p.add_argument("--bandwidth-gbs", type=float, default=546.0)
    args = p.parse_args(argv)
    for d in args.eval:
        for row in rows(d.expanduser(), args.tok_s, args.bandwidth_gbs):
            print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
