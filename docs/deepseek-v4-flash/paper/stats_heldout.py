"""The held-out confirmation (Section "Held-out texts"), from the stored evals.

42 evaluation sessions on which no threshold, hot set or budget was chosen, each cut to its first
8,192 tokens, scored in six groups and three passes (``jobs/heldout_mac.sh``). They are not
untouched: they belong to the 54 sessions on which two rules for ranking experts were compared. Same metric as ``stats.py``: KL to the
full model per text, tool use excluded, paired differences with a 95% bootstrap interval over
texts. The 36 texts outside tool use are all distinct, so each has weight one.

The last two tables need files beside the evals: the pack of token ids and the tuning session
list (openings a text shares with calibration or tuning text), and the archived 4-bit students
of the pod's comparison of ranking rules (``x1/pod/eval``).

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


def boot(d: np.ndarray, reps: int = 10000) -> str:
    idx = np.random.default_rng(0).integers(0, d.size, (reps, d.size))
    lo, hi = np.quantile(d[idx].mean(1), [0.025, 0.975])
    return f"{d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {int((d < 0).sum())} of {d.size}"


def diff(a: str, b: str, mask: np.ndarray | None = None) -> str:
    return boot((kl(a) - kl(b))[slice(None) if mask is None else mask])


def per_position(name: str) -> dict:
    return {Path(p).stem: np.load(p) for p in glob.glob(str(EVALS / f"heldout8k-g*-p*/{name}/eval-*.npz"))}


def shared_opening(least: int = 64) -> dict:
    """Per text, the length of its longest opening run of tokens that also opens a calibration or
    tuning text (0 if shorter than ``least``)."""
    rows = [json.loads(line) for line in (EVALS.parent / "pack/v41-pack.jsonl").open()]
    ids = {r["session"]: np.asarray(r["token_ids"]) for r in rows if r["split"] == "eval"}
    tuning = json.loads((EVALS.parent / "mac-eval-sessions.json").read_text())
    ref = [np.asarray(r["token_ids"]) for r in rows if r["split"] == "calibration"] + [ids[s][:1024] for s in tuning]

    def run(a: np.ndarray, b: np.ndarray) -> int:
        n = min(a.size, b.size)
        apart = np.nonzero(a[:n] != b[:n])[0]
        return int(apart[0]) if apart.size else n

    longest = {s: max(run(ids[s][:8192], r) for r in ref) for s in KEEP}
    return {s: n if n >= least else 0 for s, n in longest.items()}


def archived(student: str, at: dict) -> np.ndarray:
    """KL per text of a 4-bit student scored on the pod, on the positions scored here."""
    out = []
    for s in KEEP:
        z = np.load(EVALS.parent / f"x1/pod/eval/{student}/eval/sessions/{s}.npz")
        mine = np.isin(z["positions"], at[s]["positions"])
        assert mine.sum() == at[s]["positions"].size, s
        out.append(float(z["kl_on_teacher_topk"][mine].mean()))
    return np.array(out)


if __name__ == "__main__":
    print(f"{len(ALL)} sessions, {sum(RUN['k144'][s]['positions'] for s in ALL)} scored positions; {len(KEEP)} texts outside "
          f"tool use, median {np.median(LENGTH):g} positions\n")
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

    pos = {name: per_position(name) for name in ("k96", "s96", "k144", "s144")}
    opening = shared_opening()
    # a position repeats when the token it predicts lies inside the shared opening
    fresh = {s: pos["k144"][s]["positions"] + 1 >= opening[s] for s in KEEP}
    left = [s for s in KEEP if fresh[s].any()]
    print(f"\n{sum(n > 0 for n in opening.values())} texts open with 64 or more tokens that also open a calibration or tuning text "
          f"({', '.join(str(n) for n in sorted(opening.values(), reverse=True) if n)} tokens); "
          f"{sum(int((~f).sum()) for f in fresh.values())} of {sum(f.size for f in fresh.values())} scored positions lie inside them\n")
    print("| A | B | KL(A) - KL(B) without those positions | 95% interval | Texts where A is better |\n|---|---|---:|---|---|")
    for a, b in (("k96", "s96"), ("k144", "s144")):
        print(f"| {a} | {b} | {boot(np.array([pos[a][s]['kl'][fresh[s]].mean() - pos[b][s]['kl'][fresh[s]].mean() for s in left]))} |")
    print("\n| Archived 4-bit student, 192 experts | KL | A | KL(A) - KL(student) | 95% interval | Texts where A is better |\n|---|---:|---|---:|---|---|")
    students = {name: archived(name, pos["k144"]) for name in ("student-fm192", "student-reap192")}
    for name, x in students.items():
        for a in ("k96", "k144"):
            print(f"| {name} | {x.mean():.3f} | {a} | {boot(kl(a) - x)} |")
    print(f"| student-reap192 | {students['student-reap192'].mean():.3f} | student-fm192 | {boot(students['student-fm192'] - students['student-reap192'])} |")
