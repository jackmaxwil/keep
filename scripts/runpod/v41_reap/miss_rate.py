"""How often a token would be replayed under "replay on a strong miss", from a stored demand trace.

Replays the cache policy of ``local_eval.py`` (``residual_cache``: K residents warmed from
calibration mass, admission at tau, least recently demanded evicted) over the router's demand
(``demand.npz`` of a ``--demand`` variant) and counts, per token, the layers in which a slot with
gate weight >= WAIT was not resident. A token with at least one such layer would be rerun from the
first of them.

    python miss_rate.py DEMAND.npz --warm reap-stats.npz [--k 96 144 200] [--tau 0.3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from local_eval import residual_cache

WAITS = (0.15, 0.3, 0.5, 0.7)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("demand", type=Path)
    p.add_argument("--warm", required=True, type=Path)
    p.add_argument("--k", type=int, nargs="+", default=[96, 144, 200])
    p.add_argument("--tau", type=float, default=0.3)
    args = p.parse_args()
    z = np.load(args.demand)
    ids, gates, bounds = z["ids"].astype(np.int64), z["gates"].astype(np.float64), z["tokens"].tolist()
    warm = np.argsort(-np.load(args.warm)["reap__gate_norm_sum"], axis=1, kind="stable")
    n_layers, n_tokens, _ = ids.shape
    print(f"{n_tokens} tokens in {len(bounds)} records, {n_layers} layers, tau {args.tau}")
    print("K | gate weight resident | " + " | ".join(f"wait {w}: tokens replayed, layers rerun" for w in WAITS)
          + " | first choice missed: tokens, layers rerun")
    for k in args.k:
        hit = np.stack([residual_cache(ids[l], gates[l], k, args.tau, bounds, warm[l, :k], np.zeros((warm.shape[1], 5)))
                        for l in range(n_layers)])  # [L, T, top_k]
        miss = ~hit
        cells = [f"{k}", f"{(gates * hit).sum() / gates.sum():.3f}"]
        first = gates.argmax(-1)[..., None]
        rules = [miss & (gates >= w) for w in WAITS] + [np.take_along_axis(miss, first, -1)]
        for rule in rules:
            layer_miss = rule.any(-1)  # [L, T]
            replayed = layer_miss.any(0)
            start = np.where(replayed, layer_miss.argmax(0), n_layers)  # first layer that missed
            cells.append(f"{replayed.mean():.3f}, {(n_layers - start).mean() / n_layers:.3f}")
        print(" | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
