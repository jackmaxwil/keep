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
    """Sampling tables (C2, more samples, 200 residents, 180 with the drafter; gen2 of 2026-11-21): correct scripts of n."""
    pts = [(48, 0, 8), (96, 4, 24), (120, 14, 24), (144, 17, 24), (200, 19, 24)]
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    lo, hi = wilson(18, 24)
    ax.axhspan(lo, hi, color=ORANGE, alpha=0.15, lw=0)
    ax.axhline(18 / 24, color=ORANGE, lw=1, label="pruned to 192 experts, 4 bits, waiting offload (18/24)")
    for j, (x, k, n) in enumerate(pts):
        lo, hi = wilson(k, n)
        ax.errorbar(x, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="o", color=BLUE, ms=4, lw=0.9, capsize=2,
                    label="routed residency, 2 bits" if j == 0 else None)
        ax.annotate(f"{k}/{n}", (x, k / n), (5, -3), textcoords="offset points", fontsize=7)
    lo, hi = wilson(9, 24)
    ax.errorbar(144, 9 / 24, yerr=[[9 / 24 - lo], [hi - 9 / 24]], fmt="s", color=GREY, ms=4, lw=0.9, capsize=2,
                label="static set, 2 bits")
    ax.annotate("9/24", (144, 9 / 24), (5, -3), textcoords="offset points", fontsize=7)
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


def sparsity() -> None:
    """Experts per layer N and active per token k, from each model's published configuration."""
    rows = [("GLM-5.3", 256, 8), ("GLM-5.3-Flash", 288, 8), ("DeepSeek-V4.1-Flash-1017", 384, 6), ("Kimi K3.1 Pro", 896, 16)]
    fig, ax = plt.subplots(figsize=(5.4, 1.9))
    for i, (name, n, k) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, 100, color="#e5e7eb", height=0.6)
        ax.barh(y, 100 * k / n, color=BLUE, height=0.6)
        ax.text(101.5, y, f"{k} of {n} active ({100 * k / n + 1e-9:.1f}%)".replace(".0%", "%"), va="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=7.5)
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of a layer's expert weights used by one token (%)")
    ax.spines["bottom"].set_bounds(0, 100)
    save(fig, "fig-sparsity")


def missrules() -> None:
    """Schematic: what serves a token when one of its demanded experts is absent, under each miss rule."""
    fig, axes = plt.subplots(1, 4, figsize=(6.4, 1.75))
    rules = [("Wait", "exact; the token blocks\non a storage read"), ("Substitute", "no wait; memory for a\ncoarse copy of every expert"),
             ("Re-route", "no wait, no extra memory;\na different expert runs"), ("Skip", "no wait, no memory;\nthe contribution is lost")]
    for ax, (title, note) in zip(axes, rules):
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        ax.axis("off")
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=2)
        ax.text(0, 26.5, "demand", fontsize=6.5, color=GREY)
        ax.text(0, 15.5, "served", fontsize=6.5, color=GREY)
        for j, name in enumerate(("a", "b", "c")):
            miss = j == 1
            ax.add_patch(FancyBboxPatch((1 + 9.5 * j, 19), 7.5, 5.5, boxstyle="round,pad=0.3", fc="white" if miss else "#dbeafe",
                                        ec=RED if miss else "black", lw=0.8, ls=(0, (2, 1.5)) if miss else "-"))
            ax.text(4.75 + 9.5 * j, 21.75, name, ha="center", va="center", fontsize=7.5, color=RED if miss else "black")
            fc, ec, label, color = "#dbeafe", "black", name, "black"
            if miss:
                fc, ec, label, color = {"Wait": ("#fef3c7", ORANGE, "b", "black"), "Substitute": ("#f3f4f6", GREY, "b̃", GREY),
                                        "Re-route": ("#dbeafe", BLUE, "d", BLUE), "Skip": ("white", "white", "—", GREY)}[title]
            ax.add_patch(FancyBboxPatch((1 + 9.5 * j, 8), 7.5, 5.5, boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=0.8))
            ax.text(4.75 + 9.5 * j, 10.75, label, ha="center", va="center", fontsize=7.5, color=color)
        ax.text(29.5, 26.5, "b is absent", fontsize=6, color=RED, ha="right")
        ax.text(15, 2.5, note, fontsize=6, ha="center", va="center", linespacing=1.0)
    axes[0].text(29.5, 15.6, "after the read", fontsize=6, color=ORANGE, ha="right")
    axes[1].text(29.5, 15.6, "coarse copy of b", fontsize=6, color=GREY, ha="right")
    axes[2].text(29.5, 15.6, "next best resident", fontsize=6, color=BLUE, ha="right")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig-missrules")


def locality() -> None:
    """routing.json: demanded gate weight that K experts per layer would serve, static set against a session's own set."""
    import json
    d = json.loads((HERE / "routing.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    k = d["k"]
    for key, color, label in (("own", BLUE, "the text's own best $K$"), ("static", GREY, "static set (chosen on calibration text)")):
        a.fill_between(k, [100 * v for v in d[key + "_lo"]], [100 * v for v in d[key + "_hi"]], color=color, alpha=0.15, lw=0)
        a.plot(k, [100 * v for v in d[key + "_mean"]], "o-", color=color, ms=3, lw=1.2, label=label)
    a.axvline(144, color="black", lw=0.5, ls=(0, (3, 3)))
    a.set_xlabel("Experts per layer in the set, $K$ (of 384)")
    a.set_ylabel("Demanded gate weight served (%)")
    a.set_xlim(0, 390)
    a.set_ylim(0, 102)
    a.legend(loc="lower right", fontsize=6.5)
    layers = range(len(d["layer_own_144"]))
    b.plot(layers, [100 * v for v in d["layer_own_144"]], "-", color=BLUE, lw=1.2)
    b.plot(layers, [100 * v for v in d["layer_static_144"]], "-", color=GREY, lw=1.2)
    b.set_xlabel("Layer")
    b.set_ylabel("Served at $K=144$ (%)")
    b.set_ylim(0, 102)
    b.set_xlim(0, 39)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig-locality")


def timeline() -> None:
    """Schematic: an expert demanded at token t is read in the background and serves a few tokens later."""
    fig, ax = plt.subplots(figsize=(5.6, 1.9))
    ax.set_xlim(-12, 70)
    ax.set_ylim(0, 30)
    ax.axis("off")
    for i in range(7):
        served = i >= 4
        ax.add_patch(FancyBboxPatch((1 + 9.5 * i, 20.5), 8, 5.5, boxstyle="round,pad=0.25", fc="#dbeafe" if served else "#fef3c7",
                                    ec=BLUE if served else ORANGE, lw=0.8))
        ax.text(5 + 9.5 * i, 23.25, "$t$" if i == 0 else f"$t{{+}}{i}$", ha="center", va="center", fontsize=7.5)
    ax.text(-1.5, 23.25, "tokens", ha="right", va="center", fontsize=7)
    ax.text(-1.5, 13.5, "storage", ha="right", va="center", fontsize=7)
    ax.text(-1.5, 5, "router", ha="right", va="center", fontsize=7)
    ax.add_patch(FancyBboxPatch((20.5, 11.5), 15, 4, boxstyle="round,pad=0.25", fc="#f3f4f6", ec="black", lw=0.7))
    ax.text(28, 13.5, "read $e$ into a row", ha="center", va="center", fontsize=6.5)
    ax.plot([5, 5], [20, 13.5], color="black", lw=0.7)
    ax.add_patch(FancyArrowPatch((5, 13.5), (20, 13.5), arrowstyle="-|>", mutation_scale=6, lw=0.7, color="black"))
    ax.text(6, 18, "$e$ demanded, absent", fontsize=6, va="center")
    ax.text(6, 10.2, "seen two calls later", fontsize=5.5, ha="left", va="center")
    ax.plot([36, 36], [2.5, 16], color="black", lw=0.7, ls=(0, (3, 2)))
    ax.plot([-0.5, 36], [5, 5], color=ORANGE, lw=2.2, solid_capstyle="butt")
    ax.plot([36, 67], [5, 5], color=BLUE, lw=2.2, solid_capstyle="butt")
    ax.text(17.5, 7.2, "bias of $e$ is $-\\infty$: re-routed", fontsize=6.5, color=ORANGE, ha="center")
    ax.text(51.5, 7.2, "bias restored: $e$ serves", fontsize=6.5, color=BLUE, ha="center")
    ax.text(36, 1, "bytes in place", fontsize=6, ha="center")
    save(fig, "fig-timeline")


def order() -> None:
    """Schematic of Section 5: gain per byte from residents and from bits as memory grows, and the order it implies."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    m = np.linspace(0.02, 1, 300)
    knee = 0.2
    residents, bits = 1.05 * np.exp(-2.6 * (m - knee)), np.where(m < knee, 2.2, 0.36)
    cross = m[np.argmin(np.abs(residents - 0.36) + (m < knee) * 9)]
    ax.axvspan(0, knee, color=RED, alpha=0.08, lw=0)
    ax.axvspan(knee, cross, color=BLUE, alpha=0.08, lw=0)
    ax.axvspan(cross, 1, color=RED, alpha=0.08, lw=0)
    ax.plot(m[m >= knee], residents[m >= knee], color=BLUE, lw=1.6, label="next byte as a resident")
    ax.plot(m[m >= knee], bits[m >= knee], color=RED, lw=1.6, label="next byte as precision on the most used experts")
    ax.plot([0.02, knee], [1.22, 1.22], color=RED, lw=1.6)
    ax.annotate("", (knee, 1.22), (knee, 0.36), arrowprops={"arrowstyle": "-", "lw": 0.6, "ls": (0, (2, 2)), "color": RED})
    for x, text in ((knee / 2, "bits, up to\na usable code"), ((knee + cross) / 2, "residents"), ((cross + 1) / 2, "bits for the\nmost used experts")):
        ax.text(x, 1.1, text, ha="center", va="top", fontsize=7, linespacing=1.0)
    ax.annotate("reach: every expert reachable, at no cost in memory", ((knee + 1) / 2, 1.39), ha="center", fontsize=7, color=GREEN)
    ax.plot([knee, 1], [1.36, 1.36], color=GREEN, lw=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Memory for experts")
    ax.set_ylabel("Gain per byte (schematic)")
    ax.legend(loc="center right", fontsize=6.5, bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig-order")


def domains() -> None:
    """domains.json: KL per domain at 144 residents, static set against routed residency, and the whole pool."""
    import json
    d = json.loads((HERE / "domains.json").read_text())
    names = {"agentic-python": "agentic", "bash": "shell", "code-diffs": "code\ndiffs",
             "cyber": "security", "networking": "networks", "business": "business", "business-email": "business\nemail",
             "english-chat": "chat", "english-prose": "prose"}
    keys = sorted(names, key=lambda k: -(d[k]["tau384:s144"] - d[k]["c1-residents:k144"]))
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    for j, (variant, color, label) in enumerate((("tau384:s144", GREY, "static set, 144"), ("c1-residents:k144", BLUE, "routed residency, 144"),
                                                 ("tau384:q384", "#93c5fd", "whole pool, 384"))):
        ax.bar([i + 0.27 * (j - 1) for i in range(len(keys))], [d[k][variant] for k in keys], 0.25, color=color, label=label)
    ax.set_xticks(range(len(keys)), [names[k] for k in keys], fontsize=6.5)
    ax.set_ylabel("KL to the full model (nats)")
    ax.legend(loc="upper right", fontsize=7)
    save(fig, "fig-domains")


def admission() -> None:
    """stats.md and the threshold sweep: KL and admissions per token against the admission threshold, 144 residents."""
    taus, kl, adm = ["0.1", "0.2", "0.3", "0.45"], [0.313, 0.307, 0.320, 0.415], [16.7, 7.5, 2.6, 1.1]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    bx = ax.twinx()
    bx.bar(range(4), adm, 0.45, color="#e5e7eb", zorder=0)
    bx.set_ylabel("Experts read from storage per token", color=GREY)
    bx.set_ylim(0, 40)
    bx.spines["right"].set_visible(True)
    bx.tick_params(axis="y", colors=GREY)
    ax.set_zorder(bx.get_zorder() + 1)
    ax.patch.set_visible(False)
    for y, text, color in ((0.654, "static set (never admits)", GREY), (0.255, "best fixed set per text, in hindsight", GREEN),
                           (0.235, "whole pool in memory", "black")):
        ax.axhline(y, color=color, lw=0.8, ls=(0, (4, 3)))
        ax.text(3.45, y + (0.012 if y > 0.24 else -0.03), text, fontsize=6.5, color=color, ha="right")
    ax.plot(range(4), kl, "o-", color=BLUE, ms=4.5, lw=1.3)
    ax.text(1.0, 0.345, "routed residency", fontsize=7, color=BLUE, ha="center")
    ax.set_xticks(range(4), taus)
    ax.set_xlabel("Admission threshold $\\tau$ on the gate weight")
    ax.set_ylabel("KL to the full model (nats)")
    ax.set_ylim(0.18, 0.72)
    ax.set_xlim(-0.5, 3.5)
    save(fig, "fig-admission")


def speed() -> None:
    """Decode speed ranges of the serving table (build plan)."""
    rows = [("(a) 48 experts at 4 bits, offload that waits", 3.3, 4.5, ORANGE), ("(b) 48 experts at 4 bits, static set", 30, 32, GREY),
            ("(c) pool of 192, static set of 144", 28.6, 30.1, GREY), ("(d) pool of 192, routed residency, 144, $\\tau=0.2$", 26.0, 27.9, BLUE),
            ("(e) pool of 384, routed residency, 144, $\\tau=0.3$", 26.7, 27.2, BLUE),
            ("(f) pool of 384, routed residency, 183,\nspeculative decoding", 30, 40, RED)]
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    for i, (label, lo, hi, color) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, hi - lo, left=lo, height=0.5, color=color)
        ax.plot([lo, hi], [y, y], "|", color=color, ms=9, mew=1.5)
        ax.text(hi + 1, y, f"{lo:.1f}–{hi:.1f}" if lo % 1 or hi % 1 else f"{lo:g}–{hi:g}", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=7)
    ax.set_xlim(0, 48)
    ax.set_xlabel("Decode speed (tokens per second)")
    save(fig, "fig-speed")


if __name__ == "__main__":
    frontier(), generation(), system(), space()
    sparsity(), missrules(), locality(), timeline(), order(), domains(), admission(), speed()
    if BITS:
        bits()
