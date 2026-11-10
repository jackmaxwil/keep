"""A fixed resident expert set under stock oMLX, with no copy of the checkpoint (K1's speed floor).

``make`` writes a model directory that links every file of a DeepSeek-V4.1 source checkpoint and
replaces two: ``config.json`` (``n_routed_experts`` = C) and the tensor index, which lists only
each layer's C hottest experts under the names 0..C-1. ``compact-map.json`` records which stored
tensor each of those names means and which router rows to keep.

Any other first argument runs the oMLX CLI with three patches, all read-side: shard headers gain
the new names as aliases, and the router weight and biases are cut to the kept rows as they load.
The model then routes each token to its top experts among the C resident ones, every expert is in
memory, and oMLX's fused decode kernels apply (they do not with its own expert offload).

    python omlx_compact.py make SRC DST --experts 48 --mass reap50-pruned.npz
    OMLX_COMPACT=DST/compact-map.json python omlx_compact.py serve --model-dir ...
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

EXPERT = re.compile(r"(layers\.(\d+)\.ffn\.experts)\.(\d+)\.(.+)$")
GATE = re.compile(r"layers\.(\d+)\.ffn\.gate\.(weight|bias|bias_vl)$")


def make(src: Path, dst: Path, experts: int, mass_file: Path) -> None:
    import numpy as np

    mass = np.load(mass_file)["reap__gate_norm_sum"]  # [layer, expert] routed gate mass
    by_mass = [np.argsort(-row, kind="stable")[:experts].tolist() for row in mass]
    hot = [sorted(row) for row in by_mass]
    rank = [[by.index(e) for e in row] for by, row in zip(by_mass, hot)]  # 0 is the hottest
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name not in ("config.json", "model.safetensors.index.json") and not (dst / item.name).exists():
            (dst / item.name).symlink_to(item.resolve())
    config = json.loads((src / "config.json").read_text())
    text = config.get("text_config", config)
    assert mass.shape == (text["num_hidden_layers"], text["n_routed_experts"]), mass.shape
    text["n_routed_experts"] = experts
    (dst / "config.json").write_text(json.dumps(config, indent=1))
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map, alias = {}, {}
    for key, filename in index["weight_map"].items():
        m = EXPERT.match(key)
        if m is None:
            weight_map[key] = filename
        elif int(m[3]) in hot[int(m[2])]:
            new = f"{m[1]}.{hot[int(m[2])].index(int(m[3]))}.{m[4]}"
            weight_map[new], alias[new] = filename, key
    index["weight_map"] = weight_map
    (dst / "model.safetensors.index.json").write_text(json.dumps(index))
    (dst / "compact-map.json").write_text(json.dumps(
        {"experts": experts, "source": str(src.resolve()), "hot": hot, "rank": rank, "alias": alias}))
    print(f"{dst}: {experts} experts per layer, {len(alias)} tensors renamed")


def patch(spec: dict, full: dict | None = None) -> None:
    import mlx.core as mx
    from omlx.patches.deepseek_v41 import loading, residency, storage

    alias, hot = spec["alias"], [mx.array(rows) for rows in spec["hot"]]

    def aliased(header):  # a fresh dict: the stored names are read before any is overwritten
        return {**header, **{new: header[old] for new, old in alias.items() if old in header}}

    opened, read_header, source_weights = storage.TensorFile.__init__, residency.header_with_offset, loading.iter_source_weights
    headers = {}

    def tensor_file(self, path):
        opened(self, path)
        self.header = aliased(self.header)

    def header_with_offset(*args):
        if args not in headers:
            header, offset = read_header(*args)
            headers[args] = aliased(header), offset
        return headers[args]

    def iter_source_weights(*args, **kwargs):
        for tensors, specs in source_weights(*args, **kwargs):
            for key in tensors:
                m = GATE.search(key)
                if m:
                    if full is not None:  # the whole router, for callers that track demand
                        full[int(m[1]), m[2]] = tensors[key]
                    tensors[key] = tensors[key][hot[int(m[1])]]
            yield tensors, specs

    storage.TensorFile.__init__ = tensor_file
    residency.header_with_offset = header_with_offset
    loading.iter_source_weights = iter_source_weights


if __name__ == "__main__":
    if sys.argv[1:2] == ["make"]:
        p = argparse.ArgumentParser()
        p.add_argument("src", type=Path)
        p.add_argument("dst", type=Path)
        p.add_argument("--experts", type=int, required=True)
        p.add_argument("--mass", type=Path, required=True)
        a = p.parse_args(sys.argv[2:])
        make(a.src.expanduser(), a.dst.expanduser(), a.experts, a.mass.expanduser())
    else:
        patch(json.loads(Path(os.environ["OMLX_COMPACT"]).read_text()))
        from omlx.cli import main

        sys.exit(main())
