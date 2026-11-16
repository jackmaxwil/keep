"""Figures of the paper. KL numbers are from stats.md (``stats.py``), generation counts from the build plan.

    uv run --with matplotlib python figures.py [--png DIR]

Writes fig-*.pdf beside this file; ``--png`` also writes PNG copies for a quick look.
"""

from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).parent
PNG = Path(sys.argv[sys.argv.index("--png") + 1]) if "--png" in sys.argv else None
BITS = {"k80n32": (0.014, 0.004, 0.024), "k128n32": (-0.004, -0.014, 0.005), "k168n64": (-0.023, -0.042, -0.002)}  # stats.md
GREY, BLUE, RED, GREEN, ORANGE = "#6b7280", "#0072B2", "#D55E00", "#009E73", "#E69F00"  # Okabe-Ito
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42})


def save(fig, name: str) -> None:
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight")
    if PNG:
        fig.savefig(PNG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def frontier() -> None:
    """stats.md, KL over ten domains: static sets, routed residency, the hot set at 92 GB, the REAP checkpoint."""
    static = [(48, 17.7, 1.640), (96, 35.4, 1.010), (144, 53.1, 0.654), (200, 73.7, 0.427), (250, 92.2, 0.303)]
    routed = [(32, 11.8, 0.732), (48, 17.7, 0.582), (64, 23.6, 0.496), (96, 35.4, 0.390), (144, 53.1, 0.320),
              (200, 73.7, 0.279), (250, 92.2, 0.254)]
    whole = (141.6, 0.235)
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.axvline(92.2, color="black", lw=0.6, ls=(0, (4, 3)))
    ax.text(93.5, 0.78, "128 GB laptop", fontsize=6.5, rotation=90, va="center")
    ax.plot([x for _, x, _ in static] + [whole[0]], [y for *_, y in static] + [whole[1]], "s-", color=GREY, ms=4, lw=1.2,
            label="static set, 2 bits")
    ax.plot([x for _, x, _ in routed] + [whole[0]], [y for *_, y in routed] + [whole[1]], "o-", color=BLUE, ms=4, lw=1.2,
            label="routed residency, 2 bits")
    ax.plot([92.2], [0.215], "*", color=RED, ms=10, label="routed residency, 64 hot at 3 bits")
    ax.plot([144.4], [0.389], "D", color=ORANGE, ms=5, label="pruned to 192 experts, 4 bits")
    ax.plot([70.8], [0.486], "D", color=ORANGE, ms=5, mfc="white", label="the same 192 experts, 2 bits")
    for k, x, y in static:
        ax.annotate(str(k), (x, y), (4, 3), textcoords="offset points", fontsize=6.5, color=GREY)
    for k, x, y in routed:
        ax.annotate(str(k), (x, y), (-14, -8) if k == 250 else (-2, -9), textcoords="offset points", fontsize=6.5,
                    color=BLUE, ha="center")
    ax.annotate("384", whole, (0, -10), textcoords="offset points", fontsize=6.5, ha="center")
    ax.set_xlabel("Memory held by routed experts (GB)")
    ax.set_ylabel("KL to the full model (nats)")
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 1.75)
    ax.legend(loc="upper right", fontsize=7)
    save(fig, "fig-frontier")


def bits() -> None:
    """stats.md: KL(hot set variant) - KL(2 bits only) at equal expert memory, 95% interval over texts."""
    rows = [  # label, difference, low, high
        ("35 GB: 80 residents, 32 hot at 3 bits\nvs. 96 residents", *BITS["k80n32"]),
        ("53 GB: 128 residents, 32 hot at 3 bits\nvs. 144 residents", *BITS["k128n32"]),
        ("53 GB: 127 residents, 16 hot at 4 bits\nvs. 144 residents", -0.010, -0.022, 0.003),
        ("53 GB: 110 residents, 32 hot at 4 bits\nvs. 144 residents", 0.011, -0.003, 0.021),
        ("65 GB: 144 residents, 32 hot at 4 bits\nvs. 177 residents", -0.015, -0.033, 0.005),
        ("74 GB: 168 residents, 64 hot at 3 bits\nvs. 200 residents", *BITS["k168n64"]),
        ("92 GB: 218 residents, 64 hot at 3 bits\nvs. 250 residents", -0.039, -0.058, -0.017),
        ("92 GB: 217 residents, 32 hot at 4 bits\nvs. 250 residents", -0.036, -0.056, -0.011),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for i, (label, d, lo, hi) in enumerate(rows):
        three = "3 bits" in label
        ax.errorbar(d, len(rows) - 1 - i, xerr=[[d - lo], [hi - d]], fmt="o" if three else "s", color=RED if three else BLUE,
                    ms=4, lw=0.9, capsize=2)
    ax.plot([], [], "o", color=RED, ms=4, label="hot set at 3 bits")
    ax.plot([], [], "s", color=BLUE, ms=4, label="hot set at 4 bits")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=6.5)
    ax.set_xlabel("KL(with hot set) $-$ KL(2 bits only), nats")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    save(fig, "fig-bits")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def generation() -> None:
    """Sampling tables (C2, more samples, 200 residents, 180 with the drafter): correct scripts of n."""
    pts = [(48, 0, 8), (96, 2, 8), (120, 4, 8), (144, 17, 24), (200, 19, 24)]
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    lo, hi = wilson(18, 24)
    ax.axhspan(lo, hi, color=ORANGE, alpha=0.15, lw=0)
    ax.axhline(18 / 24, color=ORANGE, lw=1, label="pruned to 192 experts, 4 bits, waiting offload (18/24)")
    for j, (x, k, n) in enumerate(pts):
        lo, hi = wilson(k, n)
        ax.errorbar(x, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="o", color=BLUE, ms=4, lw=0.9, capsize=2,
                    label="routed residency, 2 bits" if j == 0 else None)
        ax.annotate(f"{k}/{n}", (x, k / n), (5, -3), textcoords="offset points", fontsize=7)
    lo, hi = wilson(21, 24)
    ax.errorbar(180, 21 / 24, yerr=[[21 / 24 - lo], [hi - 21 / 24]], fmt="^", color=RED, ms=5, lw=0.9, capsize=2,
                label="routed residency, speculative decoding")
    ax.annotate("21/24", (180, 21 / 24), (-24, 2), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Residents per layer (of 384)")
    ax.set_ylabel("Sampled scripts correct")
    ax.set_xlim(30, 215)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.22)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper left", fontsize=6.5)
    save(fig, "fig-generation")


def system() -> None:
    fig, ax = plt.subplots(figsize=(5.4, 2.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, text, color="#f3f4f6", edge="black"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec=edge, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7)

    def arrow(a, b, color="black", style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=7, lw=0.8, color=color, linestyle=ls,
                                     shrinkA=0, shrinkB=0))

    box(1, 33, 18, 9, "router scores\nall $N$ experts")
    box(28, 33, 23, 9, "top-$k$ among the\n$K$ residents", "#dbeafe")
    box(59, 33, 24, 9, "run the served\nexperts", "#dbeafe")
    box(1, 3, 18, 9, "top-$k$ over the\npool (demand)", "#fef3c7")
    box(28, 3, 23, 9, "admit if absent\nand gate $\\geq \\tau$", "#fef3c7")
    box(59, 3, 24, 9, "background read\ninto a spare row", "#fef3c7")
    box(59, 18, 24, 9, "point the router\nat the new expert", "#fef3c7")
    ax.text(93.5, 37.5, "token\noutput", ha="center", va="center", fontsize=7)
    arrow((20, 37.5), (27.5, 37.5))
    arrow((52, 37.5), (58.5, 37.5), BLUE)
    arrow((84, 37.5), (88, 37.5), BLUE)
    arrow((10, 32.5), (10, 13))
    arrow((20, 7.5), (27.5, 7.5), ORANGE)
    arrow((52, 7.5), (58.5, 7.5), ORANGE)
    arrow((71, 13), (71, 17.5), ORANGE)
    ax.plot([58.5, 39.5], [22.5, 22.5], color=ORANGE, lw=0.8, ls=(0, (3, 2)))
    arrow((39.5, 22.5), (39.5, 32.5), ORANGE, ls=(0, (3, 2)))
    ax.text(38, 26.5, "changes the residents\nfor later tokens", fontsize=6.5, color=ORANGE, va="center", ha="right")
    ax.text(39.5, 43.6, "this token (never waits)", fontsize=6.5, color=BLUE, ha="center")
    save(fig, "fig-system")


def space() -> None:
    """Schematic of the design space: fraction of the pool in memory against precision, lines of equal memory."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    x = np.linspace(0.04, 1.05, 200)
    for share, at in ((0.125, 0.86), (0.25, 0.93), (0.5, 0.97)):
        ax.plot(x, 4 * share / x, color="#d1d5db", lw=0.9, zorder=0)
        ax.text(at, 4 * share / at + 0.08, f"1/{round(1 / share)} of full memory", fontsize=6, color=GREY, ha="right")
    pts = [  # x, y, filled, color, marker, label, text offset
        (1.0, 4.0, True, "black", "o", "full model", (-6, 7)),
        (0.5, 4.0, False, GREY, "s", "static pruning", (0, 8)),
        (0.25, 4.0, True, ORANGE, "o", "exact offloading\n(a miss waits)", (0, 8)),
        (1.0, 2.0, True, GREY, "s", "uniform\nquantization", (-8, -19)),
        (0.65, 2.0, False, GREY, "s", "pruning +\nquantization", (0, -19)),
        (0.375, 2.0, True, BLUE, "o", "routed residency\n(a miss is re-routed)", (-4, -19)),
        (0.33, 2.33, True, RED, "*", "with a hot set", (-34, 6)),
    ]
    for px, py, filled, color, marker, label, off in pts:
        ax.plot([px], [py], marker, ms=9 if marker == "*" else 6.5, color=color, mfc=color if filled else "white", mew=1.2)
        ax.annotate(label, (px, py), off, textcoords="offset points", fontsize=7, ha="center", linespacing=0.95)
    ax.annotate("", (0.4, 2.0), (0.625, 2.0), arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": BLUE, "shrinkA": 4, "shrinkB": 5})
    ax.text(0.51, 2.12, "indistinguishable", fontsize=6.5, color=BLUE, ha="center")
    ax.set_xlabel("Fraction of the expert pool in memory, $K/N$")
    ax.set_ylabel("Precision of residents (bits per weight)")
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 5)
    ax.set_yticks([0, 1, 2, 3, 4])
    save(fig, "fig-space")


if __name__ == "__main__":
    frontier(), generation(), system(), space()
    if BITS:
        bits()
