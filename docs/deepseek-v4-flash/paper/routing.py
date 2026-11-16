"""Routing statistics for the paper's concentration and locality figure, from stored routes of the full model.

For the evaluation sessions of the main metric (first 1,024 tokens, tool use excluded, one copy of the
shared agentic text): the share of demanded gate weight that the K experts of a layer would serve,
for the static set (ranked on calibration text, as the simulator's warm set) and for the best
fixed set of each session (ranked on the session itself).

    python routing.py [--run ~/Models/v41-reap-run] > routing.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RUN = Path(sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else "~/Models/v41-reap-run").expanduser()
SKIP = {"eval-00001", "eval-00004", "eval-00005"}  # copies of eval-00000 at this window
KS = [16, 32, 48, 64, 96, 128, 144, 168, 200, 250, 300, 384]

if __name__ == "__main__":
    order = np.argsort(-np.load(RUN / "stats/calibration/reap-stats.npz")["reap__gate_norm_sum"], axis=1, kind="stable")
    static, own, names = [], [], []
    for name in json.loads((RUN / "mac-eval-sessions.json").read_text()):
        z = np.load(RUN / "stats/eval/sessions" / f"{name}.npz")
        if z["domains"][0] == "tool-use" or name in SKIP:
            continue
        ids, gates = z["routes__expert_ids"][:1024].astype(int), z["routes__gate_weights"][:1024].astype(float)
        mass = np.zeros(order.shape)  # [layers, experts]: gate weight demanded in this session
        for layer in range(order.shape[0]):
            np.add.at(mass[layer], ids[:, layer].ravel(), gates[:, layer].ravel())
        mass /= mass.sum(1, keepdims=True)
        by_static = np.take_along_axis(mass, order, 1).cumsum(1)  # [layers, K]
        by_own = -np.sort(-mass, 1)
        static.append(by_static), own.append(by_own.cumsum(1)), names.append(name)
    static, own = np.array(static), np.array(own)  # [sessions, layers, experts]
    pick = [k - 1 for k in KS]
    print(json.dumps({
        "sessions": names, "k": KS,
        "static_mean": static.mean((0, 1))[pick].round(4).tolist(), "own_mean": own.mean((0, 1))[pick].round(4).tolist(),
        "static_lo": static.mean(1).min(0)[pick].round(4).tolist(), "static_hi": static.mean(1).max(0)[pick].round(4).tolist(),
        "own_lo": own.mean(1).min(0)[pick].round(4).tolist(), "own_hi": own.mean(1).max(0)[pick].round(4).tolist(),
        "layer_static_144": static.mean(0)[:, 143].round(4).tolist(), "layer_own_144": own.mean(0)[:, 143].round(4).tolist(),
        "used_per_layer": float(((own[:, :, -1:] - own < 1e-9).argmax(2) + 1).mean().round(1)),
    }, indent=1))
