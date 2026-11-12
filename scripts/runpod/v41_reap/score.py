"""Score masked-router student runs against the teacher, per domain.

    python scripts/runpod/v41_reap/score.py --teacher /workspace/out/teacher \\
        --student 256=/workspace/out/student-256 --student 192=/workspace/out/student-192

For each candidate: mean KL(teacher || student) over the teacher's top-K, the
NLL increase on the real next token, and top-1 agreement, per domain and
overall (position-weighted). Writes JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def score(teacher: Path, student: Path) -> dict[str, dict[str, float]]:
    sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4))
    for path in sorted((student / "eval" / "sessions").glob("*.npz")):
        with np.load(path) as s, np.load(teacher / "eval" / "sessions" / path.name) as t:
            if not np.array_equal(s["positions"], t["positions"]):
                raise ValueError(f"{path.name}: positions differ from the teacher's")
            teacher_nll = t["logsumexp"] - t["target_logit"]
            row = np.array([s["kl_on_teacher_topk"].sum(), (s["student_nll"] - teacher_nll).sum(), s["top1_agree"].sum(), s["positions"].size])
            for domain in [*map(str, s["domains"]), "all"]:
                sums[domain] += row
    return {
        domain: {"kl": kl / n, "nll_delta": dn / n, "top1_agree": agree / n, "positions": int(n)}
        for domain, (kl, dn, agree, n) in sorted(sums.items())
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--student", action="append", required=True, help="KEEP=DIR, repeatable")
    args = parser.parse_args(argv)
    report = {}
    for item in args.student:
        keep, _, directory = item.partition("=")
        report[keep] = score(args.teacher, Path(directory))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
