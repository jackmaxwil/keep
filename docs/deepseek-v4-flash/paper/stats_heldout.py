"""The held-out confirmation (Section "Held-out texts"), from the stored evals.

42 evaluation sessions that no setting was chosen on, each cut to its first 8,192 tokens, scored
in six groups and three passes (``jobs/heldout_mac.sh``). Same metric as ``stats.py``: KL to the
full model per text, tool use excluded, paired differences with a 95% bootstrap interval over
texts. The 36 texts outside tool use are all distinct, so each has weight one.

    python stats_heldout.py [--evals ~/Models/v41-reap-run/evals] > stats-heldout.md
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

EVALS = Path(sys.argv[sys.argv.index("--evals") + 1] if "--evals" in sys.argv else "~/Models/v41-reap-run/evals").expanduser()
RUN: dict = {}
for path in sorted(glob.glob(str(EVALS / "heldout8k-g*-p*/summary.json"))):
    for name, sessions in json.loads(Path(path).read_text())["variants"].items():
        RUN.setdefault(name, {}).update({s: x for s, x in sessions.items() if s.startswith("eval-")})
ALL = sorted(RUN["k144"])
KEEP = [s for s in ALL if RUN["k144"][s]["domain"] != "tool-use"]
LENGTH = np.array([RUN["k144"][s]["positions"] for s in KEEP])


def kl(name: str, key: str = "kl") -> np.ndarray:
    return np.array([RUN[name][s][key] for s in KEEP])


def diff(a: str, b: str, mask: np.ndarray | None = None, reps: int = 10000) -> str:
    d = (kl(a) - kl(b))[slice(None) if mask is None else mask]
    idx = np.random.default_rng(0).integers(0, d.size, (reps, d.size))
    lo, hi = np.quantile(d[idx].mean(1), [0.025, 0.975])
    return f"{d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {int((d < 0).sum())} of {d.size}"


if __name__ == "__main__":
    print(f"{len(ALL)} sessions, {sum(RUN['k144'][s]['positions'] for s in ALL)} scored positions; {len(KEEP)} texts outside "
          f"tool use, median {int(np.median(LENGTH))} positions\n")
    print("| Variant | KL | Top-1 | KL pooled over positions | Tool-use KL |\n|---|---:|---:|---:|---:|")
    for name in ("q384", "s96", "k96", "s144", "k144", "s200", "s250"):
        tool = np.mean([RUN[name][s]["kl"] for s in ALL if s not in KEEP])
        print(f"| {name} | {kl(name).mean():.3f} | {kl(name, 'top1').mean():.3f} | {(kl(name) * LENGTH).sum() / LENGTH.sum():.3f} | {tool:.3f} |")
    print("\n| A | B | KL(A) - KL(B) | 95% interval | Texts where A is better |\n|---|---|---:|---|---|")
    for a, b in (("k96", "s96"), ("k144", "s144"), ("k96", "s200"), ("k144", "s250"), ("k144", "s200"), ("k144", "q384")):
        print(f"| {a} | {b} | {diff(a, b)} |")
    print("\n| Text length | Texts | A | B | KL(A) - KL(B) | 95% interval | Texts where A is better |\n|---|---:|---|---|---:|---|---|")
    for tag, mask in (("under 256", LENGTH < 256), ("256 to 1,023", (LENGTH >= 256) & (LENGTH < 1024)), ("1,024 and over", LENGTH >= 1024)):
        for a, b in (("k144", "s144"), ("k144", "s250"), ("k144", "q384")):
            print(f"| {tag} | {int(mask.sum())} | {a} | {b} | {diff(a, b, mask)} |")
