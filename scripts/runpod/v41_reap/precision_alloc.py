"""Static per-expert precision allocation inside REAP-50 (step 2c): rungs 0 (dropped), 1 and 2 bits.

Within each layer, under a budget of B units (1-bit expert = 1 unit, 2-bit = 2), greedily buy the
upgrade with the largest estimated error reduction per unit:

    cost(e, r) = m_e * eps_r       m_e = expert e's share of the layer's routed output mass
                                   (sum of g * ||f|| on the pruned model, ``--capture-imatrix``)
                                   eps_r = relative output error at rung r (eps_0 = 1, dropped)

eps_1 and eps_2 are the layer means of KEEP's weighted relative MSE (gate/up averaged, plus down)
from ``keep_quant.py``'s ``metrics.jsonl``. Every layer gets the same budget, because moving
budget between layers by mass share lost to uniform in step 1.

Writes ``keep.json`` (experts at rung >= 1, original ids, for ``local_eval.py --variant``) and
``hot.json`` (rung 2, for ``--expert-override HOT@hot.json+REST``), plus ``alloc.json``.
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np


def layer_eps(metrics: Path) -> dict[int, float]:
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    return {r["layer"]: (r["w1_wrel_mse"] + r["w3_wrel_mse"]) / 2 + r["w2_wrel_mse"] for r in rows}


def allocate(mass: np.ndarray, eps: tuple[float, float, float], budget: int) -> np.ndarray:
    """Rung per expert (0/1/2) maximizing the estimated error reduction under ``budget`` units."""

    rung = np.zeros(len(mass), dtype=int)
    heap = [(-m * (eps[0] - eps[1]), e) for e, m in enumerate(mass)]
    heapq.heapify(heap)
    left = budget
    while heap and left > 0:
        _, e = heapq.heappop(heap)
        rung[e] += 1
        left -= 1
        if rung[e] == 1:
            heapq.heappush(heap, (-mass[e] * (eps[1] - eps[2]), e))
    return rung


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-survivors", required=True, type=Path)
    p.add_argument("--imatrix", required=True, type=Path, help="pruned-model capture npz (reap__gate_norm_sum)")
    p.add_argument("--metrics-1bit", required=True, type=Path)
    p.add_argument("--metrics-2bit", required=True, type=Path)
    p.add_argument("--budget", required=True, type=int, help="units per layer (1-bit expert = 1, 2-bit = 2)")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    survivors = json.loads(args.checkpoint_survivors.read_text())["survivors"]
    gate_mass = np.load(args.imatrix)["reap__gate_norm_sum"].astype(np.float64)  # [L, E_local]
    e1, e2 = layer_eps(args.metrics_1bit), layer_eps(args.metrics_2bit)
    keep, hot, rows = {}, {}, []
    for layer in range(len(survivors)):
        ids = sorted(survivors[str(layer)])
        mass = gate_mass[layer] / gate_mass[layer].sum()
        eps = (1.0, e1[layer], e2[layer])
        rung = allocate(mass, eps, args.budget)
        keep[str(layer)] = [ids[i] for i in np.flatnonzero(rung >= 1)]
        hot[str(layer)] = [ids[i] for i in np.flatnonzero(rung == 2)]
        rows.append({"layer": layer, "dropped": int((rung == 0).sum()), "rung1": int((rung == 1).sum()),
                     "rung2": int((rung == 2).sum()), "eps1": round(eps[1], 4), "eps2": round(eps[2], 4),
                     "est_cost": float((mass * np.asarray(eps)[rung]).sum())})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "keep.json").write_text(json.dumps({"survivors": keep}))
    (args.out / "hot.json").write_text(json.dumps({"survivors": hot}))
    (args.out / "alloc.json").write_text(json.dumps({"budget": args.budget, "layers": rows}, indent=1))
    print(json.dumps({"budget": args.budget, "kept_mean": float(np.mean([len(v) for v in keep.values()])),
                      "hot_mean": float(np.mean([len(v) for v in hot.values()])),
                      "est_cost_mean": float(np.mean([r["est_cost"] for r in rows]))}))
    return 0


def _check() -> None:
    eps = (1.0, 0.3, 0.1)
    # flat mass: a 1-bit floor for everyone before any upgrade (0.175 per unit vs 0.05)
    assert allocate(np.full(4, 0.25), eps, 4).tolist() == [1, 1, 1, 1]
    m = np.array([0.5, 0.3, 0.15, 0.05])
    assert allocate(m, eps, 3).tolist() == [1, 1, 1, 0]
    # upgrading the 0.5 expert (0.10) beats keeping the 0.05 one (0.035)
    assert allocate(m, eps, 4).tolist() == [2, 1, 1, 0]
    assert allocate(m, eps, 5).tolist() == [2, 2, 1, 0]


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--check"]:
        _check()
        print("ok")
        sys.exit(0)
    sys.exit(main())
