"""How much demand a starting resident set serves, by how it is chosen, from a stored demand trace.

No admissions: each record gets one set of K experts per layer and the score is the share of its
gate weight, from token ``--after`` on, that the set holds. Ways to choose the set:

- ``static``      the K with the most calibration mass (``--warm``), the same for every record
- ``prompt:P``    the record's own top K over its first P tokens, padded with ``static``
- ``profiles:D``  D profiles from k-means over the mass of records from other sessions; the record
                  takes the profile nearest to the mass of its first ``--see`` tokens
- ``own``         the record's own top K over all its tokens (hindsight)

Two rows do admit, for comparison: ``lru`` is the cache of ``local_eval.py`` at ``--tau`` from the
static start, and ``prompt:P+lru`` the same cache after the prompt warm start.

    python start_sets.py DEMAND.npz --warm reap-stats.npz [--k 96 144] [--after 256] [--see 32]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from local_eval import residual_cache


def top_k(mass: np.ndarray, k: int, pad: np.ndarray | None = None) -> np.ndarray:
    """[L, k] experts by mass per layer; experts with no mass give way to ``pad``'s order."""
    if pad is not None:
        mass = mass + 1e-9 * (pad.shape[1] - np.argsort(np.argsort(-pad, axis=1), axis=1)) * (mass.max() + 1)
    return np.argsort(-mass, axis=1, kind="stable")[:, :k]


def kmeans(x: np.ndarray, d: int, rng: np.random.Generator, rounds: int = 20) -> np.ndarray:
    centers = x[rng.choice(len(x), d, replace=False)]
    for _ in range(rounds):
        label = (x @ centers.T).argmax(1)  # rows are unit norm
        for c in range(d):
            if (label == c).any():
                centers[c] = x[label == c].mean(0)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    return centers


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("demand", type=Path)
    p.add_argument("--warm", required=True, type=Path)
    p.add_argument("--k", type=int, nargs="+", default=[96, 144])
    p.add_argument("--after", type=int, default=256)
    p.add_argument("--see", type=int, default=32)
    p.add_argument("--tau", type=float, default=0.3)
    args = p.parse_args()
    z = np.load(args.demand)
    ids, gates, bounds = z["ids"].astype(np.int64), z["gates"].astype(np.float64), z["tokens"].tolist()
    static = np.load(args.warm)["reap__gate_norm_sum"].astype(np.float64)
    n_layers, n = static.shape
    starts = np.concatenate([[0], np.cumsum(bounds)])

    def mass(lo: int, hi: int) -> np.ndarray:  # [L, n] gate weight per expert over tokens lo..hi
        out = np.zeros((n_layers, n))
        for layer in range(n_layers):
            out[layer] = np.bincount(ids[layer, lo:hi].ravel(), weights=gates[layer, lo:hi].ravel(), minlength=n)
        return out

    full = [mass(starts[r], starts[r + 1]) for r in range(len(bounds))]
    seen = {P: [mass(starts[r], starts[r] + P) for r in range(len(bounds))] for P in sorted({args.see, 64, 256} - {0})}
    tail = [mass(starts[r] + args.after, starts[r + 1]) for r in range(len(bounds))]
    unit = np.stack([m.ravel() / (np.linalg.norm(m) + 1e-12) for m in full])
    rng = np.random.default_rng(0)
    session = np.array([str(name).partition("#")[0] for name in z["records"]])
    warm = np.argsort(-static, axis=1, kind="stable")
    in_tail = np.concatenate([np.arange(b) >= args.after for b in bounds])

    def cached(k: int, policy: str) -> float:  # gate weight resident from --after on, over all records
        hit = np.stack([residual_cache(ids[l], gates[l], k, args.tau, bounds, warm[l, :k], np.zeros((n, 5)), policy=policy)
                        for l in range(n_layers)])
        return float((gates * hit)[:, in_tail].sum() / gates[:, in_tail].sum())

    def served(r: int, chosen: np.ndarray) -> float:
        return float(np.take_along_axis(tail[r], chosen, 1).sum() / tail[r].sum())

    print(f"{len(bounds)} records, scored on tokens {args.after}+; profiles see the first {args.see} tokens")
    for k in args.k:
        rows = {"static": [served(r, top_k(static, k)) for r in range(len(bounds))]}
        for P in seen:
            if P <= args.after:
                rows[f"prompt:{P}"] = [served(r, top_k(seen[P][r], k, static)) for r in range(len(bounds))]
        for d in (4, 8, 16):
            scores = []
            for r in range(len(bounds)):
                others = np.flatnonzero(session != session[r])
                centers = kmeans(unit[others].copy(), d, rng)
                look = seen[args.see][r].ravel()
                best = centers[(centers @ look).argmax()].reshape(n_layers, n)
                scores.append(served(r, top_k(best, k, static)))
            rows[f"profiles:{d}"] = scores
        rows["own"] = [served(r, top_k(full[r], k)) for r in range(len(bounds))]
        print(f"K = {k}: " + ", ".join(f"{name} {np.mean(v):.3f}" for name, v in rows.items())
              + f" | lru {cached(k, 'lru'):.3f}, " + ", ".join(f"prompt:{P}+lru {cached(k, f'prompt:{P}'):.3f}" for P in seen if P <= args.after))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
