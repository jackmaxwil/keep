"""Per-expert admission thresholds for the routed cache, fitted on stored calibration routes.

``local_eval.residual_cache`` admits an expert when a slot's gate weight reaches one threshold,
the same for every expert. Here each expert outside the starting resident set gets its own, from
the teacher's routes of the calibration sessions (``collect.py`` stores the first 2,048 of each):

- An admitted expert almost always stays to the end of its record (about 100 admissions per
  layer per record against 144 slots), so under threshold ``g`` expert ``e`` misses every slot up
  to and including its first slot with gate >= ``g``, and costs one install if there is one.
- A miss is counted as ``max(gate - 0.25, 0)``, the strong part of the slot: with re-routing a weak
  slot is replaced at little cost, and over the measured variants at 144 residents this quantity
  orders them as their KL does, where missed gate mass does not (tau 0.2 against 0.3).
- ``tau[e]`` minimizes missed strong mass + ``--lam`` installs, summed over records, plus
  ``--prior`` times the layer's mean cost curve (an expert the routes never show gets the
  layer's best single threshold). The starting residents keep ``--default``.

Prints the frontier on held-out sessions (3 folds) for single thresholds and for the fit, through
``residual_cache`` itself, then fits on every session and writes ``tau`` [layers, experts].

Result (2026-11-15): on held-out routes the fit misses 9% less strong mass than one threshold at
equal installs, and in KL it loses (0.476-0.480 against 0.454 for tau 0.3 at 144 residents, 22
eval sessions). Missed strong mass ranks single thresholds, not policies that change which
experts are missed. Kept as the record of that result; serving uses one threshold.

    python fit_tau.py --routes ~/Models/v41-reap-run/stats/calibration/sessions \\
        --pack ~/Models/v41-reap-run/pack/v41-pack.jsonl \\
        --cache-warm ~/Models/v41-reap-run/stats/calibration/reap-stats.npz \\
        --cache-k 144 --lam 0.4 --out ~/Models/v41-reap-run/tau/k144-lam40.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from local_eval import residual_cache

STRONG = 0.25
GRID = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1e9])  # 1e9: never admitted


def strong(w):
    return np.maximum(w - STRONG, 0)


def load_routes(root: Path, pack: Path) -> list[tuple]:
    """One (ids [T, layers, top_k], gates, record lengths) per session file of ``root``."""
    lengths: dict[str, list[int]] = {}
    for line in pack.open():
        r = json.loads(line)
        lengths.setdefault(r["session"], []).append(len(r["token_ids"]))
    out = []
    for path in sorted(root.glob("*.npz")):
        z = np.load(path)
        ids, gates = z["routes__expert_ids"].astype(np.int64), z["routes__gate_weights"].astype(np.float32)
        bounds, left = [], ids.shape[0]
        for n in lengths[path.stem]:  # the routes are the session's first tokens, record after record
            if left <= 0:
                break
            bounds.append(min(n, left))
            left -= n
        assert sum(bounds) == ids.shape[0], path
        out.append((ids, gates, bounds))
    return out


def joined(sessions: list[tuple]) -> tuple:
    return (np.concatenate([s[0] for s in sessions]), np.concatenate([s[1] for s in sessions]),
            [n for s in sessions for n in s[2]])


def cost_tables(ids, gates, bounds, warm, k: int, experts: int) -> tuple[np.ndarray, np.ndarray]:
    """(missed strong mass, installs) [layers, experts, len(GRID)] over the records, for the experts
    outside each layer's starting set ``warm[layer, :k]``."""
    miss, inst = (np.zeros((ids.shape[1], experts, len(GRID))) for _ in range(2))
    for layer in range(ids.shape[1]):
        cold = np.ones(experts, bool)
        cold[warm[layer, :k]] = False
        start = 0
        for n in bounds:
            e, w = ids[start:start + n, layer].ravel(), gates[start:start + n, layer].ravel()  # in time order
            start += n
            e, w = e[cold[e]], w[cold[e]]
            order = np.argsort(e, kind="stable")
            e, w = e[order], w[order]
            cuts = np.flatnonzero(np.diff(e)) + 1
            for ee, ww in zip(np.split(e, cuts), np.split(w, cuts)):
                if not ee.size:
                    continue
                lost = np.cumsum(strong(ww))
                for g, threshold in enumerate(GRID):
                    hit = np.flatnonzero(ww >= threshold)
                    miss[layer, ee[0], g] += lost[hit[0]] if hit.size else lost[-1]
                    inst[layer, ee[0], g] += bool(hit.size)
    return miss, inst


def fit(miss, inst, warm, k: int, lam: float, prior: float, default: float) -> np.ndarray:
    cost = miss + lam * inst
    pooled = cost.mean(axis=1, keepdims=True)
    tau = GRID[np.argmin(cost + max(prior, 1e-6) * pooled, axis=-1)]
    for layer in range(tau.shape[0]):
        tau[layer, warm[layer, :k]] = default
    return tau


def replay(ids, gates, bounds, warm, k: int, tau, experts: int) -> tuple[float, float]:
    """(share of strong mass missed, installs per token over all layers) under ``residual_cache``."""
    miss = total = installs = 0.0
    for layer in range(ids.shape[1]):
        counts = np.zeros((experts, 7))
        resident = residual_cache(ids[:, layer], gates[:, layer], k, tau[layer] if np.ndim(tau) else tau, bounds,
                                  warm[layer, :k], counts)
        s = strong(gates[:, layer])
        miss, total, installs = miss + float(s[~resident].sum()), total + float(s.sum()), installs + counts[:, 4].sum()
    return miss / total, installs / ids.shape[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--routes", type=Path, required=True, help="collect.py session files of the calibration sweep")
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--cache-warm", type=Path, required=True, help="as local_eval.py --cache-warm")
    p.add_argument("--cache-k", type=int, required=True)
    p.add_argument("--lam", type=float, nargs="+", required=True, help="strong mass one install is worth; the first is written")
    p.add_argument("--prior", type=float, default=5.0)
    p.add_argument("--default", type=float, default=0.3)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    stats = np.load(args.cache_warm.expanduser())["reap__gate_norm_sum"]
    warm, experts, k = np.argsort(-stats, axis=1, kind="stable"), stats.shape[1], args.cache_k
    sessions = load_routes(args.routes.expanduser(), args.pack.expanduser())
    folds = [sessions[i::3] for i in range(3)]
    tables = [cost_tables(*joined(f), warm, k, experts) for f in folds]

    def held_out(make_tau) -> tuple[float, float]:
        scores = [replay(*joined(folds[i]), warm, k, make_tau(*(sum(tables[j][t] for j in range(3) if j != i) for t in (0, 1))),
                         experts) for i in range(3)]
        return tuple(np.mean(scores, axis=0))

    print(f"{len(sessions)} sessions, {sum(s[0].shape[0] for s in sessions)} tokens, {k} residents; held-out, 3 folds")
    print("| Policy | Strong mass missed | Installs per token |\n|---|---:|---:|")
    for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        print("| tau {} | {:.4f} | {:.2f} |".format(t, *held_out(lambda *_, t=t: t)), flush=True)
    for lam in args.lam:
        print("| per expert, lam {} | {:.4f} | {:.2f} |".format(
            lam, *held_out(lambda miss, inst, lam=lam: fit(miss, inst, warm, k, lam, args.prior, args.default))), flush=True)
    tau = fit(*(sum(t[i] for t in tables) for i in (0, 1)), warm, k, args.lam[0], args.prior, args.default)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, tau=tau, cache_k=k, lam=args.lam[0], prior=args.prior, default=args.default)
    cold = np.ones(tau.shape, bool)
    np.put_along_axis(cold, warm[:, :k], False, axis=1)
    shares = {f"{g:g}" if g < 1 else "never": round(float(np.mean(tau[cold] == g)), 3) for g in GRID}
    print(f"wrote {args.out}: thresholds of the experts outside the starting sets {shares}")


if __name__ == "__main__":
    main()
