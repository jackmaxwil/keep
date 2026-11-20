"""The second model's numbers (GLM-5.3-Flash, Section "A second model"), from the stored summaries.

Same metric as ``stats.py``: KL to the full model over the sessions outside tool use, one entry per
distinct text (four agentic sessions share theirs), paired differences with a 95% bootstrap interval
over the 17 texts. Run 1 is the pre-registered pass; run 2 is the sensitivity pass.

    python stats_glm.py > stats-glm.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "glm-5.3-flash" / "results"
SAME = ["eval-00000", "eval-00001", "eval-00004", "eval-00005"]  # one text, in two domains
RUN = {k: v for name in ("residency-run2.json", "residency-run1.json") for k, v in json.loads((RESULTS / name).read_text()).items()}
KEEP = [s for s, v in RUN["_teacher"].items() if v["domain"] != "tool-use" and s not in SAME[1:]]
W = np.array([len(SAME) if s == SAME[0] else 1 for s in KEEP], float)
W /= W.sum()


def kl(name: str, key: str = "kl") -> np.ndarray:
    assert len({round(RUN[name][s][key], 6) for s in SAME}) == 1, name
    return np.array([RUN[name][s][key] for s in KEEP])


def diff(a: str, b: str, reps: int = 10000) -> str:
    d, rng = kl(a) - kl(b), np.random.default_rng(0)
    idx = rng.integers(0, d.size, (reps, d.size))
    lo, hi = np.quantile((d[idx] * W[idx]).sum(1) / W[idx].sum(1), [0.025, 0.975])
    return f"{float(d @ W):+.3f} | [{lo:+.3f}, {hi:+.3f}] | {int((d < 0).sum())} of {d.size}"


if __name__ == "__main__":
    teacher = RUN["_teacher"]
    print(f"{len(KEEP)} texts, {sum(v['tokens'] for v in teacher.values())} tokens in {len(teacher)} sessions; "
          f"full model NLL {np.mean([v['nll'] for v in teacher.values()]):.3f}\n")
    print("| Variant | Residents | tau | KL | Top-1 | Gate weight resident | Admissions per token and layer |\n|---|---:|---:|---:|---:|---:|---:|")
    for name in ("all", "s72", "k72", "s108", "k108", "s150", "k150", "k72t20", "k108t20", "k150t20", "k72t10", "k108t10", "k150t10", "o108"):
        p = RUN[name]["_policy"]
        tau = "static" if p["tau"] > 1e6 else ("hindsight" if p["policy"] == "oracle" else p["tau"])
        print(f"| {name} | {p['residents']} | {tau} | {float(kl(name) @ W):.3f} | {float(kl(name, 'top1') @ W):.3f} | "
              f"{p['gate_weight_resident']:.3f} | {p['admissions_per_token_layer']:.3f} |")
    print("\n| A | B | KL(A) - KL(B) | 95% interval | Texts where A is better |\n|---|---|---:|---|---|")
    for a, b in (("k72", "s72"), ("k108", "s108"), ("k150", "s150"), ("k72t20", "s72"), ("k108t20", "s108"), ("k150t20", "s150"),
                 ("k72", "s108"), ("k108", "s150"), ("k72t20", "s108"), ("k108t20", "s150"), ("k108t20", "k108"),
                 ("k108t10", "k108t20"), ("o108", "k108"), ("o108", "k108t20"), ("o108", "all")):
        print(f"| {a} | {b} | {diff(a, b)} |")
