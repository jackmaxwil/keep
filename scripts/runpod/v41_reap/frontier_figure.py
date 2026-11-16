"""Item 5 figure: one policy over the pool of 384, KL against resident expert memory and against
bytes read per token. Series (all at 2-bit lattice codes, 22 sessions, 1,024 tokens):

- static pruning: the top-K experts by routed mass, never changed (tau 1e9);
- routed cache: K residents, admission at gate >= 0.3, a miss re-routed among residents;
- routed + hot: the same with the hottest H residents one bit higher (2-bit base + 1-bit residual);
- tier 0: slots with gate below tau_drop skipped, on the routed and routed + hot points.

Writes ``frontier.md`` (table) and ``frontier.svg`` (two panels) into ``--out``. Standard library
plus numpy, through ``routed_report.rows``.

    python frontier_figure.py --evals ~/Models/v41-reap-run/evals --out docs/deepseek-v4-flash/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from routed_report import rows

SERIES = {  # label: [(eval dir, variant)], in plotting order
    "static pruning": [("frontier", "s48"), ("frontier", "s96"), ("tau384", "s144"), ("frontier", "s200"),
                       ("frontier", "s250"), ("tau384", "q384")],
    "routed cache": [("c1-residents", f"k{k}") for k in (32, 48, 64, 96, 144, 200, 250)] + [("tau384", "q384")],
    "routed + 3-bit hot": [("b2-nested", "k234n32"), ("b2-nested", "k218n64"), ("b2-nested", "k202n96")],
    "tier 0, 144 residents": [("c1-residents", "k144"), ("frontier", "k144d05"), ("frontier", "k144d10"),
                              ("frontier", "k144d20")],
    "tier 0, 218 + 64 hot": [("b2-nested", "k218n64"), ("frontier", "k218n64d05"), ("frontier", "k218n64d10")],
}
COLORS = dict(zip(SERIES, ["#6b7280", "#2563eb", "#dc2626", "#16a34a", "#d97706"]))  # one color per series in both panels


def point(evals: Path, d: str, v: str) -> dict:
    row = next(r for r in rows(evals / d, 30.0, 546.0) if r["variant"] == f"{d}:{v}")
    doms = json.loads((evals / d / "summary.json").read_text())["variants"][v]["_domains"]
    row["kl_no_tool"] = round(sum(x["kl"] for k, x in doms.items() if k != "tool-use") / (len(doms) - 1), 4)
    return row


def panel(series: dict, x: str, y: str, box: tuple, title: str, xlabel: str) -> list[str]:
    """One scatter-and-line panel as SVG elements inside ``box`` = (left, top, width, height)."""
    left, top, w, h = box
    pts = [p for s in series.values() for p in s]
    x0, x1 = min(p[x] for p in pts), max(p[x] for p in pts)
    y0, y1 = 0.0, max(p[y] for p in pts) * 1.05
    sx = lambda v: left + (v - x0) / (x1 - x0 or 1) * w
    sy = lambda v: top + h - (v - y0) / (y1 - y0) * h
    ylabel = "KL against the full model, no tool-use"
    out = [f'<rect x="{left}" y="{top}" width="{w}" height="{h}" fill="none" stroke="#999"/>',
           f'<text x="{left + w / 2}" y="{top - 10}" text-anchor="middle" font-weight="bold">{title}</text>',
           f'<text x="{left + w / 2}" y="{top + h + 38}" text-anchor="middle">{xlabel}</text>',
           (f'<text x="{left - 42}" y="{top + h / 2}" text-anchor="middle" '
           f'transform="rotate(-90 {left - 42} {top + h / 2})">{ylabel}</text>')]
    for i in range(6):
        vx, vy = x0 + (x1 - x0) * i / 5, y0 + (y1 - y0) * i / 5
        out += [f'<text x="{sx(vx)}" y="{top + h + 16}" text-anchor="middle" font-size="11">{vx:.1f}</text>',
                f'<text x="{left - 6}" y="{sy(vy) + 4}" text-anchor="end" font-size="11">{vy:.2f}</text>']
    for label, s in series.items():
        color = COLORS[label]
        line = " ".join(f"{sx(p[x]):.1f},{sy(p[y]):.1f}" for p in s)
        out.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2"/>')
        out += [f'<circle cx="{sx(p[x]):.1f}" cy="{sy(p[y]):.1f}" r="3.5" fill="{color}"/>' for p in s]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evals", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = {label: [point(args.evals.expanduser(), d, v) for d, v in pts] for label, pts in SERIES.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["| Series | Variant | Expert memory (GB) | Read per token (GB) | Dropped slots | KL | KL, no tool-use |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for label, s in data.items():
        lines += [f"| {label} | {r['variant']} | {r['resident_gb']} | {r['read_gb_per_tok']} | {r['drop_slots']:.1%} | "
                  f"{r['kl']:.3f} | {r['kl_no_tool']:.3f} |" for r in s]
    (args.out / "frontier.md").write_text("\n".join(lines) + "\n")
    memory = {k: data[k] for k in ("static pruning", "routed cache", "routed + 3-bit hot")}
    tier0 = {k: data[k] for k in ("tier 0, 144 residents", "tier 0, 218 + 64 hot")}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="440" font-family="sans-serif" font-size="13">',
           '<rect width="1000" height="440" fill="white"/>',
           *panel(memory, "resident_gb", "kl_no_tool", (70, 40, 380, 300), "Pruning, offload, precision",
                  "resident expert memory (GB)"),
           *panel(tier0, "read_gb_per_tok", "kl_no_tool", (580, 40, 380, 300), "Tier 0 (per-token skip)",
                  "routed expert bytes read per token (GB)")]
    for i, (label, color) in enumerate(COLORS.items()):
        svg += [f'<rect x="{70 + i * 185}" y="405" width="12" height="12" fill="{color}"/>',
                f'<text x="{88 + i * 185}" y="416" font-size="12">{label}</text>']
    (args.out / "frontier.svg").write_text("\n".join(svg + ["</svg>"]) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
