"""Re-rank experts by total routed mass (sum of g * ||f||), optionally inside an earlier survivors set.

"Prune twice": measure routing on the model that is already pruned (``collect.py --sweep
calibration --mask TABLE``), then keep each layer's top K of that table by the new mass. On the
Mac this cut KL by 0.18 going from 192 to 96 experts; neither the mass criterion nor the
pruned-model routing did so alone (recovery log, step 2c).

    python rerank.py --stats OUT/calibration/sessions --within fm-288.json --keep 192 --out pt-192.json

``--stats`` is a directory of ``collect.py`` session files or one npz holding
``reap__gate_norm_sum`` [layers, experts]. Without ``--within`` every expert is a candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def total_mass(stats: Path) -> np.ndarray:
    paths = sorted(stats.glob("*.npz")) if stats.is_dir() else [stats]
    assert paths, f"no npz under {stats}"
    total = 0
    for path in paths:
        with np.load(path) as z:
            total = total + z["reap__gate_norm_sum"].astype(np.float64)
    return total


def rerank(mass: np.ndarray, keep: int, within: dict[str, list[int]] | None) -> dict[str, list[int]]:
    out = {}
    for layer in range(mass.shape[0]):
        ids = np.asarray(sorted(within[str(layer)]) if within else range(mass.shape[1]))
        assert keep <= len(ids), f"layer {layer}: keep {keep} of {len(ids)}"
        order = np.argsort(-mass[layer, ids], kind="stable")[:keep]
        out[str(layer)] = sorted(int(e) for e in ids[order])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats", required=True, type=Path)
    p.add_argument("--within", type=Path, help="survivors JSON: candidates are this table's experts")
    p.add_argument("--keep", required=True, type=int)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    within = json.loads(args.within.read_text())["survivors"] if args.within else None
    mass = total_mass(args.stats)
    table = rerank(mass, args.keep, within)
    kept = np.mean([mass[int(l), ids].sum() / mass[int(l)].sum() for l, ids in table.items()])
    args.out.write_text(json.dumps({"record_type": "keep_reap_survivors_v1", "criterion": "total routed mass",
                                    "stats": str(args.stats), "within": str(args.within) if args.within else None,
                                    "survivors": table}))
    print(json.dumps({"keep": args.keep, "layers": len(table), "mass_kept_mean": round(float(kept), 4)}))
    return 0


def _check() -> None:
    mass = np.array([[5.0, 1.0, 4.0, 3.0], [0.0, 2.0, 9.0, 1.0]])
    assert rerank(mass, 2, None) == {"0": [0, 2], "1": [1, 2]}
    assert rerank(mass, 1, {"0": [1, 3], "1": [0, 3]}) == {"0": [3], "1": [3]}  # only inside the table


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        _check()
        print("ok")
        sys.exit(0)
    sys.exit(main())
