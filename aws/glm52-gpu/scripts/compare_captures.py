"""Compare two teacher-capture directories (npz per session) for parity.

Two gates:
1. Cross-run parity per shared session file: top-K values/logsumexp/tail agree
   within bf16-forward tolerance; top-1 token identical almost everywhere.
2. Dense-vs-DSA internal consistency: at shared probe positions (<2047), the
   DSA full-session captures must match the dense prefix captures (causality).

Exit code 0 = PASS, 1 = FAIL. Writes a JSON report either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# bf16 forward across different GPU architectures: logit-level noise is real
# but small; the distribution-level quantities are what training consumes.
VALUE_ATOL = 0.05      # absolute logit tolerance (bf16 accumulation noise)
TOP1_MIN_AGREE = 0.995  # fraction of positions whose argmax token matches
LSE_ATOL = 0.05
TAIL_ATOL = 5e-3


def _compare(a: dict, b: dict, label: str) -> dict:
    pos_a, pos_b = a["positions"], b["positions"]
    shared, ia, ib = np.intersect1d(pos_a, pos_b, return_indices=True)
    if shared.size == 0:
        return {"label": label, "pass": False, "reason": "no shared positions"}
    top1_a = a["topk_logit_ids"][ia, 0]
    top1_b = b["topk_logit_ids"][ib, 0]
    top1_agree = float((top1_a == top1_b).mean())
    val_diff = float(
        np.max(np.abs(a["topk_logit_values"][ia, 0] - b["topk_logit_values"][ib, 0]))
    )
    lse_diff = float(np.max(np.abs(a["logsumexp"][ia] - b["logsumexp"][ib])))
    tail_diff = float(np.max(np.abs(a["tail_mass"][ia] - b["tail_mass"][ib])))
    ok = (
        top1_agree >= TOP1_MIN_AGREE
        and val_diff <= VALUE_ATOL
        and lse_diff <= LSE_ATOL
        and tail_diff <= TAIL_ATOL
    )
    return {
        "label": label,
        "pass": bool(ok),
        "shared_positions": int(shared.size),
        "top1_agree": top1_agree,
        "max_top1_value_diff": val_diff,
        "max_logsumexp_diff": lse_diff,
        "max_tail_mass_diff": tail_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--new-dir", required=True)
    parser.add_argument(
        "--dsa-dir",
        default=None,
        help="separate dir holding the dense-prefix/dsa-full pair (defaults to new-dir)",
    )
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    ref_dir, new_dir = Path(args.ref_dir), Path(args.new_dir)

    results = []
    # Gate 1: cross-run parity on every session present in both dirs
    for ref_file in sorted(ref_dir.glob("*.npz")):
        new_file = new_dir / ref_file.name
        if not new_file.exists():
            results.append(
                {"label": f"cross:{ref_file.stem}", "pass": False, "reason": "missing"}
            )
            continue
        results.append(
            _compare(
                dict(np.load(ref_file)), dict(np.load(new_file)), f"cross:{ref_file.stem}"
            )
        )

    # Gate 2: dense-vs-DSA internal consistency within the NEW captures
    dsa_dir = Path(args.dsa_dir) if args.dsa_dir else new_dir
    new_files = {f.stem: dict(np.load(f)) for f in dsa_dir.glob("*.npz")}
    dense = [k for k in new_files if k.endswith("__dense_prefix2048")]
    for dense_key in dense:
        dsa_key = dense_key.replace("__dense_prefix2048", "__dsa_full")
        if dsa_key in new_files:
            results.append(
                _compare(new_files[dense_key], new_files[dsa_key], f"dsa-vs-dense:{dsa_key}")
            )

    overall = bool(results) and all(r["pass"] for r in results)
    report = {"overall_pass": overall, "checks": results}
    Path(args.report).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
