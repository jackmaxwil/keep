"""The second model's numbers (GLM-5.3-Flash, Section "A second model"), from the stored summaries.

Same metric as ``stats.py``: KL to the full model over the sessions outside tool use, one entry per
distinct text (four agentic sessions share theirs), paired differences with a 95% bootstrap interval
over the 17 texts.

Run 3 is the reported pass: the admission threshold 0.3 on the model's own routing weights, which
sum to its routing scale (2.5), as the first model's threshold is applied. Runs 1 and 2 applied the
threshold to weights normalized to one, by mistake; they are kept as other thresholds, named by
the share of a token's gate weight (``k108s30`` is a share of 0.3, which is 0.75 on the routing
weights; the reported 0.3 is a share of 0.12).

    python stats_glm.py > stats-glm.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "glm-5.3-flash" / "results"
SAME = ["eval-00000", "eval-00001", "eval-00004", "eval-00005"]  # one text, in two domains
RUN: dict = {}
for name, rename in (("residency-run1.json", lambda k: k + "s30" if k.startswith("k") and "t" not in k else k.replace("t20", "s20")),
                     ("residency-run2.json", lambda k: k.replace("t20", "s20").replace("t10", "s10")),
                     ("residency-run3.json", lambda k: k)):
    for key, value in json.loads((RESULTS / name).read_text()).items():
        if not (name != "residency-run3.json" and key in ("o108", "all") and False):
            RUN[rename(key) if key[0] == "k" else ("o108share" if key == "o108" and name == "residency-run2.json" else key)] = value
# Residents against bits (glm53_bits_pod.sh, four groups merged): keys get a "b" prefix, so the run's
# own FP8 rows (bk72, bk108, bk150) sit beside run 3's. cK is K residents at 2 bits, hH the H hottest
# of them with a 1-bit residual (half a 2-bit resident each), f the hot set at native FP8.
BITS = RESULTS / "residency-bits.json"
if BITS.exists():
    RUN.update({"b" + k: v for k, v in json.loads(BITS.read_text()).items() if k[0] != "_"})
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
    for name in ("all", "s72", "k72", "s108", "k108", "s150", "k150", "o108", "k72s10", "k108s10", "k150s10", "k72s20", "k108s20", "k150s20",
                 "k72s30", "k108s30", "k150s30"):
        p = RUN[name]["_policy"]
        share = {"s10": 0.1, "s20": 0.2, "s30": 0.3}.get(name[-3:])
        tau = "static" if p["tau"] > 1e6 else ("hindsight" if p["policy"] == "oracle" else (f"{share * 2.5:g} (share {share})" if share else "0.3 (share 0.12)"))
        print(f"| {name} | {p['residents']} | {tau} | {float(kl(name) @ W):.3f} | {float(kl(name, 'top1') @ W):.3f} | "
              f"{p['gate_weight_resident']:.3f} | {p['admissions_per_token_layer']:.3f} |")
    print("\n| A | B | KL(A) - KL(B) | 95% interval | Texts where A is better |\n|---|---|---:|---|---|")
    for a, b in (("k72", "s72"), ("k108", "s108"), ("k150", "s150"), ("k72", "s108"), ("k108", "s150"), ("k72", "s150"),
                 ("o108", "k108"), ("o108", "all"), ("k108", "k108s10"), ("k108", "k108s20"), ("k108", "k108s30"),
                 ("k108s30", "s150"), ("k108s20", "s150")):
        print(f"| {a} | {b} | {diff(a, b)} |")
    if BITS.exists():
        # memory in 2-bit experts per layer: an FP8 expert is four, a 1-bit residual half
        print("\n| Variant | Residents | Hot | Memory (2-bit experts) | KL | Top-1 | Gate weight resident |\n|---|---:|---:|---:|---:|---:|---:|")
        for name in ("ball", "bk72", "bc72", "bc56h32", "bk108", "bc108", "bc92h32", "bc76h64", "bc108f16", "bk150", "bc150",
                     "bc134h32", "bc118h64", "bc188", "bc172h32", "bc156h64", "bc288", "bc288h32"):
            p, hot = RUN[name]["_policy"], 16 if name.endswith("f16") else (int(name.split("h")[1]) if "h" in name else 0)
            k = p["residents"]
            memory = ("-" if name == "ball" else f"{4 * k}" if name[1] == "k" else
                      f"{k + 3 * hot}" if name.endswith("f16") else f"{k + hot / 2:g}")  # FP8 hot: 3 more than its 2-bit copy
            print(f"| {name} | {k} | {hot} | {memory} | {float(kl(name) @ W):.3f} | {float(kl(name, 'top1') @ W):.3f} | "
                  f"{p['gate_weight_resident']:.3f} |")
        print("\n| A | B | KL(A) - KL(B) | 95% interval | Texts where A is better |\n|---|---|---:|---|---|")
        for a, b in (("bc56h32", "bc72"), ("bc92h32", "bc108"), ("bc76h64", "bc108"), ("bc134h32", "bc150"), ("bc118h64", "bc150"),
                     ("bc172h32", "bc188"), ("bc156h64", "bc188"), ("bc288", "bk72"), ("bc72", "bk72"), ("bc108", "bk108"),
                     ("bc150", "bk150"), ("bc108f16", "bc108"), ("bc288h32", "bc288"), ("bc288", "ball"), ("bk108", "k108")):
            print(f"| {a} | {b} | {diff(a, b)} |")
