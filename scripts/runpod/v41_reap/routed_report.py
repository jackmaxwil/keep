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
    python routed_report.py --compare EVAL_DIR:c96t2 EVAL_DIR:s96
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
        n_layers = c.shape[0]
        used = c[:, :, :2].sum() + c[:, :, 5].sum()  # every routed slot
        tokens = used / 6 / n_layers
        k = tiers.get("cache_k")
        n_bases, hi_units = tiers.get("n_bases", c.shape[1]), tiers.get("hi_units", 1.0)
        units = n_bases + (k if k is not None else n_bases) * hi_units
        hi_read = hi_units if tiers.get("copy_mode") else 1 + hi_units  # a separate copy is read alone
        read = (c[:, :, 0].sum() + c[:, :, 1].sum() * hi_read) * UNIT_BYTES / tokens
        ssd = c[:, :, 4].sum() * hi_units * UNIT_BYTES / tokens if k is not None else 0.0
        if tiers.get("copy_mode") and k is not None and not tiers.get("miss"):
            # A separate-copy cache can instead drop a hot expert's low copy (1 extra unit, not 2) and read
            # it back on demotion: every promote/demote cycle then moves 3 units where a residual moves 1.
            row |= {"units_per_layer_swap": round(n_bases + k, 1),
                    "ssd_gbs_swap": round(c[:, :, 4].sum() * 3 * UNIT_BYTES / tokens * tok_s / 1e9, 2)}
        row |= {
            "tau": tiers["tau"], "tau_drop": tiers.get("tau_drop", 0.0), "cache_k": k, "policy": tiers.get("policy"),
            "miss": tiers.get("miss"),
            "n_bases": n_bases, "hi_units": round(hi_units, 2),
            "hi_slots": round(tiers["hi_slot_share"], 3), "drop_slots": round(tiers.get("drop_slot_share", 0.0), 3),
            "units_per_layer": round(units, 1), "resident_gb": round(units * n_layers * UNIT_BYTES / 1e9, 1),
            "ssd_mb_per_tok": round(ssd / 1e6, 1), "ssd_gbs": round(ssd * tok_s / 1e9, 2),
            "read_gb_per_tok": round(read / 1e9, 2), "tok_s_bound": round(bandwidth_gbs * 1e9 / read, 1),
        }
        out.append(row)
    return out


def session_kl(spec: str) -> dict[str, np.ndarray]:
    """``EVAL_DIR:VARIANT`` -> {session: per-position KL}."""

    d, _, name = spec.rpartition(":")
    return {p.stem: np.load(p)["kl"].astype(np.float64) for p in sorted((Path(d).expanduser() / name).glob("*.npz"))
            if p.stem != "admissions"}


def compare(a: str, b: str, reps: int = 2000, block: int = 64, seed: int = 0) -> dict:
    """Paired block bootstrap of the domain-mean KL difference A - B.

    Positions are resampled in blocks of ``block`` within each session (the same blocks for A and
    B), sessions are pooled by domain as in ``local_eval.py``, and domains are averaged."""

    ka, kb = session_kl(a), session_kl(b)
    da, _, na = a.rpartition(":")
    doms = {s: r["domain"] for s, r in json.loads((Path(da).expanduser() / "summary.json").read_text())["variants"][na].items()
            if not s.startswith("_")}
    assert set(ka) == set(kb) == set(doms), "the two variants cover different sessions"
    rng = np.random.default_rng(seed)

    def stat(pick) -> float:
        num: dict[str, float] = {}
        den: dict[str, int] = {}
        for s in ka:
            i = pick(len(ka[s]))
            num[doms[s]] = num.get(doms[s], 0.0) + float((ka[s][i] - kb[s][i]).sum())
            den[doms[s]] = den.get(doms[s], 0) + len(i)
        return float(np.mean([num[d] / den[d] for d in num]))

    def blocks(n: int) -> np.ndarray:
        starts = rng.integers(0, max(n - block, 0) + 1, size=-(-n // block))
        return (starts[:, None] + np.arange(min(block, n))[None, :]).ravel()[:n]

    point = stat(lambda n: np.arange(n))
    draws = np.sort([stat(blocks) for _ in range(reps)])
    return {"a": a, "b": b, "kl_diff": round(point, 4), "ci95": [round(float(draws[int(0.025 * reps)]), 4),
                                                                   round(float(draws[int(0.975 * reps)]), 4)]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", action="append", default=[], type=Path)
    p.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"),
                   help="EVAL_DIR:VARIANT pairs: paired block-bootstrap 95%% interval of domain-mean KL(A) - KL(B)")
    p.add_argument("--tok-s", type=float, default=30.0)
    p.add_argument("--bandwidth-gbs", type=float, default=546.0)
    args = p.parse_args(argv)
    for d in args.eval:
        for row in rows(d.expanduser(), args.tok_s, args.bandwidth_gbs):
            print(json.dumps(row))
    for a, b in args.compare:
        print(json.dumps(compare(a, b)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
