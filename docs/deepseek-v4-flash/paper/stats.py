"""Every number of the paper's quality tables, from the stored evals (``local_eval.py`` summaries).

Primary metric: KL to the full model, mean over the 10 domains without tool use (20 sessions, 17
distinct texts: four agentic sessions share their first 1,024 tokens and count as one text with
their weight). Differences are paired over texts with a 95% bootstrap interval over the 17 texts.

    python stats.py [--evals ~/Models/v41-reap-run/evals] > stats.md
    python stats.py --domains > domains.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EVALS = Path(sys.argv[sys.argv.index("--evals") + 1] if "--evals" in sys.argv else "~/Models/v41-reap-run/evals").expanduser()
SAME = ["eval-00000", "eval-00001", "eval-00004", "eval-00005"]  # one text, in two domains


def load(name: str) -> dict:
    d, v = name.split(":")
    return json.loads((EVALS / d / "summary.json").read_text())["variants"][v]


def texts(name: str, key: str = "kl") -> tuple[np.ndarray, np.ndarray, float]:
    """(value per distinct text, weight, tool-use mean) of one variant."""
    x = load(name)
    ses = {k: s for k, s in x.items() if k.startswith("eval-")}
    assert len({round(ses[k][key], 6) for k in SAME}) == 1, name
    keep = [k for k, s in ses.items() if s["domain"] != "tool-use" and k not in SAME[1:]]
    w = np.array([len(SAME) if k == SAME[0] else 1 for k in keep], float)
    return np.array([ses[k][key] for k in keep]), w / w.sum(), float(np.mean([s[key] for s in ses.values() if s["domain"] == "tool-use"]))


def value(name: str) -> dict:
    kl, w, tool = texts(name)
    top1, _, _ = texts(name, "top1")
    return {"kl": float(kl @ w), "all": float((10 * (kl @ w) + tool) / 11), "tool": tool, "top1": float(top1 @ w)}


def diff(a: str, b: str, reps: int = 10000) -> tuple[float, float, float]:
    """KL(a) - KL(b) on the primary metric, with a 95% bootstrap interval over texts."""
    (x, w, _), (y, _, _) = texts(a), texts(b)
    d, rng = x - y, np.random.default_rng(0)
    idx = rng.integers(0, d.size, (reps, d.size))
    boot = (d[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return float(d @ w), *np.quantile(boot, [0.025, 0.975]).tolist()


VALUES = ["tau384:q384", "tau384:s144", "tau384:o144", "tau384:t10", "tau384:t20", "tau384:t30", "tau384:t45",
          *(f"c1-residents:k{k}" for k in (32, 48, 64, 96, 144, 200, 250)), *(f"frontier:s{k}" for k in (48, 96, 200, 250)),
          "b1-hot:h16", "b1-hot:h32", "b1-hot:b32", "b1-hot:k177", "b1-hot:q384h32", "b1b-hot:k217h32", "b1b-hot:k183h64",
          "b2-nested:k234n32", "b2-nested:k218n64", "b2-nested:k202n96",
          "bits-budget:k80n32", "bits-budget:k128n32", "bits-budget:k168n64", "bits-budget:s192",
          "frontier:k144d05", "frontier:k144d10", "frontier:k144d20", "frontier:k218n64d05", "frontier:k218n64d10",
          "tau-fit:k144L20", "tau-fit:k144L40", "tau-fit:k96L40", "tau-cost:c14", "tau-cost:c20",
          "step2-fp4-bf16:reap50", "step1:reap50", "step2-e8p-podimp:reap50", "step2b-1bit:reap50", "step2c-selection:prunedmass-96",
          "e1-residency:rs144", "e1-residency:rr144", "e1-residency:rr96", "e1-residency:ro96",
          "step4-routed:all2", "step4-routed:s96", "step4-routed:c96t2", "step4-routed:c96t3",
          *(f"dialkl:{v}" for v in ("k73", "k73w30", "k73w15", "k140w30", "k207w30", "s192", "k192")), "dialkl-ista:ista192",
          *(f"timing:{v}" for v in ("k96L", "k144L", "k200L", "k96LF", "k144LF"))]
C, F, T, B = "c1-residents:", "frontier:", "tau384:", "bits-budget:"
DIFFS = [*((f"{C}k{k}", f"{F}s{k}") for k in (48, 96, 200, 250)), (C + "k144", T + "s144"),
         (C + "k200", C + "k144"), (C + "k250", C + "k200"), (T + "q384", C + "k250"),
         (C + "k48", T + "s144"), (C + "k96", T + "s144"), (C + "k96", F + "s200"), (C + "k144", F + "s250"),
         ("b1-hot:h16", C + "k144"), ("b1-hot:h32", C + "k144"), ("b1-hot:b32", "b1-hot:k177"),
         ("b1b-hot:k217h32", C + "k250"), ("b1b-hot:k183h64", C + "k250"),
         ("b2-nested:k234n32", C + "k250"), ("b2-nested:k218n64", C + "k250"), ("b2-nested:k202n96", C + "k250"),
         ("b2-nested:k218n64", "b1b-hot:k217h32"), ("b2-nested:k218n64", T + "q384"), ("b1-hot:q384h32", T + "q384"),
         (B + "k80n32", C + "k96"), (B + "k128n32", C + "k144"), (B + "k168n64", C + "k200"),
         (B + "s192", "step2-e8p-podimp:reap50"), (T + "q384", "step2-fp4-bf16:reap50"), (F + "s250", "step2-fp4-bf16:reap50"),
         (T + "t20", T + "t30"), (T + "t10", T + "t30"), (T + "t45", T + "t30"), (T + "o144", T + "t30"), (T + "q384", T + "t30"),
         (F + "k144d05", C + "k144"), (F + "k144d10", C + "k144"), (F + "k144d20", C + "k144"), (F + "k144d20", C + "k96"),
         (F + "k218n64d05", "b2-nested:k218n64"), (F + "k218n64d10", "b2-nested:k218n64"),
         ("tau-fit:k144L20", C + "k144"), ("tau-fit:k144L40", C + "k144"), ("tau-fit:k96L40", C + "k96"),
         ("tau-cost:c14", C + "k144"), ("tau-cost:c20", C + "k144"),
         ("e1-residency:rr144", "e1-residency:rs144"), (T + "t20", "e1-residency:rr144"), (C + "k144", "step2-e8p-podimp:reap50"),
         ("step2c-selection:prunedmass-96", "step2b-1bit:reap50"), ("step4-routed:c96t2", "step4-routed:s96"),
         ("step4-routed:c96t3", "step4-routed:s96"), ("step4-routed:c96t2", "e1-residency:rs144"),
         ("step4-routed:c96t3", "e1-residency:rs144"), ("step4-routed:all2", "step2-e8p-podimp:reap50"),
         (B + "k168n64", C + "k250"), (C + "k96", "step2-fp4-bf16:reap50"), (T + "o144", T + "q384"),
         (B + "s192", "step2-fp4-bf16:reap50"),
         ("step2-fp4-bf16:reap50", "step1:reap50"), (C + "k96", "step1:reap50"),
         ("dialkl:k73w30", "dialkl:k73"), ("dialkl:k73w15", "dialkl:k73"), ("dialkl:k73w30", C + "k200"),
         ("dialkl:k73w30", "dialkl:s192"), ("dialkl:k73w15", T + "q384"), ("dialkl:k140w30", C + "k144"),
         ("dialkl:k207w30", C + "k200"), ("dialkl:k207w30", T + "q384"), ("dialkl-ista:ista192", "dialkl:s192"),
         ("dialkl:k73", "dialkl-ista:ista192"), ("dialkl:k73w30", "dialkl-ista:ista192"),
         *((f"timing:k{k}L", f"{C}k{k}") for k in (96, 144, 200)), ("timing:k96LF", C + "k96"), ("timing:k144LF", C + "k144"),
         ("timing:k96LF", "timing:k96L"), ("timing:k144LF", "timing:k144L"), ("timing:k96LF", F + "s96"), ("timing:k144LF", T + "s144")]

def domains(names: list[str]) -> dict:
    """KL per domain (tool use excluded) of each variant, for the per-domain figure."""
    out: dict = {}
    for name in names:
        x = load(name)
        for dom, v in x["_domains"].items():
            if dom != "tool-use":
                out.setdefault(dom, {})[name] = round(v["kl"], 4)
    return out


if __name__ == "__main__":
    if "--domains" in sys.argv:
        print(json.dumps(domains([T + "s144", C + "k144", T + "q384"]), indent=1))
        raise SystemExit
    print("| Variant | KL, 10 domains | KL, 11 domains | Tool-use KL | Top-1, 10 domains |\n|---|---:|---:|---:|---:|")
    for name in VALUES:
        try:
            v = value(name)
        except (FileNotFoundError, KeyError):
            print(f"| {name} | missing | | | |")
            continue
        print(f"| {name} | {v['kl']:.3f} | {v['all']:.3f} | {v['tool']:.2f} | {v['top1']:.3f} |")
    print("\n| A | B | KL(A) - KL(B), 10 domains | 95% interval |\n|---|---|---:|---|")
    for a, b in DIFFS:
        try:
            d, lo, hi = diff(a, b)
        except (FileNotFoundError, KeyError):
            print(f"| {a} | {b} | missing | |")
            continue
        print(f"| {a} | {b} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
