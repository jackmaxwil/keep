"""What a missed expert costs, measured through the model, for ``fit_tau.py --miss-cost``.

``fit_tau.py`` priced a miss by its gate weight, and the thresholds it fitted lost in KL. Here
the price comes from runs of ``local_eval.py`` on calibration records under thresholds drawn at
random per expert (``draw``), scored against the same codes with every expert resident
(``--reference``): the KL at a token is then what that token's misses cost. ``fit`` models it as

    KL[token] = sum over the token's missed slots of cost[layer, expert]

a ridge fit towards the one cost that fits every miss alike, one layer at a time (an expert with
few misses keeps that cost). The gate of the missed slot and its layer are not in the model: on
held-out records they explained 2 to 7% of the KL per token, the expert 20 to 30%. It prints, on
held-out records (4 folds), how well each price predicts the KL, and above all the difference
between two random policies on the same record, which is where missed gate mass failed.

Result (2026-11-15): the cost predicts held-out KL (r 0.85 over records, 0.63 for the difference,
against 0.44 and 0.41 for gate mass above 0.25), and thresholds fitted on it still lose to tau 0.3
(0.477 against 0.454 at 144 residents). It is additive over independent random misses and not
for a policy chosen by it. Kept as the record of that result.

    python miss_cost.py draw --like ~/Models/v41-reap-run/tau/k144-lam40.npz --seed 1 --out random-1.npz
    python miss_cost.py fit --run ~/Models/v41-reap-run/evals/miss-cost --variants r1 r2 --out miss-cost.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DRAWN = np.array([0.1, 0.2, 0.3, 0.5, 1e9])  # 1e9: never admitted


def load(run: Path, variant: str) -> tuple:
    """(expert ids [layers, N, top_k], gates, missed, KL [N], record of each token) over the scored tokens."""
    d = np.load(run / variant / "demand.npz")
    kl, start = np.full(d["ids"].shape[1], np.nan), 0
    for name, n in zip(d["records"], d["tokens"]):
        z = np.load(run / variant / f"{name}.npz")
        kl[start + z["positions"]] = z["kl"]
        start += n
    keep = ~np.isnan(kl)
    record = np.repeat(np.arange(len(d["tokens"])), d["tokens"])
    return (d["ids"][:, keep].astype(np.int64), d["gates"][:, keep].astype(np.float32), ~d["resident"][:, keep],
            kl[keep], record[keep])


def predict(ids, missed, cost) -> np.ndarray:
    return (cost[np.arange(ids.shape[0])[:, None, None], ids] * missed).sum((0, 2))


def fit_cost(ids, missed, kl, experts: int, prior: float | None, sweeps: int = 12) -> np.ndarray:
    """cost [layers, experts] of one miss; ``prior=None`` gives every miss the same."""
    layers, n, top_k = ids.shape
    count = missed.sum((0, 2))
    flat = count @ kl / (count @ count)
    cost, pred, rows = np.full((layers, experts), flat), flat * count, np.arange(n)[:, None].repeat(top_k, 1)
    for _ in range(sweeps if prior is not None else 0):
        for layer in range(layers):
            X, m = np.zeros((n, experts)), missed[layer]
            X[rows[m], ids[layer][m]] = 1  # an expert holds one slot of a token
            gram = X.T @ X
            alpha = prior * gram.diagonal().mean()  # in misses: ``prior`` times an expert's average count
            new = np.linalg.solve(gram + alpha * np.eye(experts), X.T @ (kl - pred + X @ cost[layer]) + alpha * flat).clip(0)
            pred += X @ (new - cost[layer])
            cost[layer] = new
    return cost


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="one random admission threshold per layer and expert")
    d.add_argument("--like", type=Path, required=True, help="a fit_tau.py file, for the shape")
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("fit", help="the price of a miss from local_eval.py --demand --reference runs")
    f.add_argument("--run", type=Path, required=True)
    f.add_argument("--variants", nargs=2, required=True, help="two random policies over the same records")
    f.add_argument("--prior", type=float, nargs="+", default=[3.0], help="ridge towards one cost for every miss; "
                   "the first is written")
    f.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "draw":
        shape = np.load(args.like.expanduser())["tau"].shape
        np.savez(args.out, tau=np.random.default_rng(args.seed).choice(DRAWN, size=shape), seed=args.seed)
        return

    run = args.run.expanduser()
    parts = [load(run, v) for v in args.variants]
    ids, gates, missed = (np.concatenate([part[i] for part in parts], axis=1) for i in range(3))
    kl, record = (np.concatenate([part[i] for part in parts]) for i in (3, 4))
    variant = np.concatenate([np.full(part[3].size, i) for i, part in enumerate(parts)])
    experts = np.load(run / args.variants[0] / "tier-counts.npy").shape[1]
    print(f"{kl.size} tokens of {record.max() + 1} records under 2 policies; mean KL {kl.mean():.3f}, "
          f"{missed.mean():.1%} of slots missed")

    def by_mass(x):  # one coefficient
        return lambda tr, te: x[te] * (x[tr] @ kl[tr]) / (x[tr] @ x[tr])

    def model(prior):
        return lambda tr, te: predict(ids[:, te], missed[:, te], fit_cost(ids[:, tr], missed[:, tr], kl[tr], experts, prior))

    models = {"gate mass": by_mass((gates * missed).sum((0, 2))),
              "gate mass above 0.25": by_mass((np.maximum(gates - 0.25, 0) * missed).sum((0, 2))),
              "misses": model(None), **{f"misses, cost per expert, prior {q:g}": model(q) for q in args.prior}}
    group = record * 2 + variant
    def mean(x):  # per record and policy
        return np.bincount(group, x, 2 * (record.max() + 1)) / np.bincount(group, minlength=2 * (record.max() + 1))

    print("| Price | R2 per token | r over records | r of the two policies' difference per record |\n|---|---:|---:|---:|")
    for name, fit_predict in models.items():
        pred = np.zeros(kl.size)
        for fold in range(4):
            te = record % 4 == fold
            pred[te] = fit_predict(~te, te)
        r2 = 1 - ((kl - pred) ** 2).sum() / ((kl - kl.mean()) ** 2).sum()
        a, b = mean(kl), mean(pred)
        print(f"| {name} | {r2:.3f} | {np.corrcoef(a, b)[0, 1]:.3f} | {np.corrcoef(a[::2] - a[1::2], b[::2] - b[1::2])[0, 1]:.3f} |",
              flush=True)
    cost = fit_cost(ids, missed, kl, experts, args.prior[0])
    np.savez(args.out, cost=cost, prior=args.prior[0])
    top = np.sort(cost.ravel())[::-1]
    print("cost quantiles (10, 50, 90, 99, 99.9%)", np.round(np.quantile(cost, [0.1, 0.5, 0.9, 0.99, 0.999]), 4).tolist())
    print(f"the costliest 1% of experts hold {top[:top.size // 100].sum() / top.sum():.0%} of the summed cost; "
          f"by layer {np.round(cost.mean(1), 4).tolist()}")


if __name__ == "__main__":
    main()
