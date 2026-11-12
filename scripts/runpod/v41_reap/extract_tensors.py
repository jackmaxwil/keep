"""Raw tensors out of an HF safetensors checkpoint, as one safetensors file on stdout.

Either one layer's chosen routed experts, renumbered 0..n-1 in the order given, or every router
tensor of the decoder layers. Names are the reference names (``stream.reference_name``), bytes
are copied untouched. Standard library only, so it runs on any pod:

    ssh pod python3 extract_tensors.py /workspace/v41-hf --layer 3 --experts 1,2,5 > layer-03.safetensors
    ssh pod python3 extract_tensors.py /workspace/v41-hf --gates > router.safetensors
"""

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path


def reference_name(name: str) -> str:  # stream.reference_name
    name = name.removeprefix("model.").replace("self_attn", "attn")
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")
    return name.replace("weight_scale_inv", "scale").replace("e_score_correction_bias", "bias")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path)
    p.add_argument("--layer", type=int)
    p.add_argument("--experts", help="comma-separated expert ids of --layer")
    p.add_argument("--gates", action="store_true")
    args = p.parse_args()
    index = json.loads((args.root / "model.safetensors.index.json").read_text())["weight_map"]
    order = {int(e): i for i, e in enumerate(args.experts.split(","))} if args.experts else {}
    picked: dict[str, tuple[str, str]] = {}  # new name -> (shard, stored name)
    for stored, shard in index.items():
        ref = reference_name(stored)
        if args.gates:
            if re.fullmatch(r"layers\.\d+\.ffn\.gate\..+", ref):
                picked[ref] = (shard, stored)
        else:
            m = re.fullmatch(rf"layers\.{args.layer}\.ffn\.experts\.(\d+)\.(.+)", ref)
            if m and int(m[1]) in order:
                picked[f"layers.{args.layer}.ffn.experts.{order[int(m[1])]}.{m[2]}"] = (shard, stored)
    assert picked and (args.gates or len(picked) == 6 * len(order)), len(picked)

    shards: dict[str, tuple[int, dict, int]] = {}
    header, slabs, offset = {}, [], 0
    for name in sorted(picked):
        shard, stored = picked[name]
        if shard not in shards:
            fd = os.open(args.root / shard, os.O_RDONLY)
            (size,) = struct.unpack("<Q", os.pread(fd, 8, 0))
            shards[shard] = fd, json.loads(os.pread(fd, size, 8)), 8 + size
        fd, entries, start = shards[shard]
        begin, end = entries[stored]["data_offsets"]
        header[name] = {"dtype": entries[stored]["dtype"], "shape": entries[stored]["shape"],
                        "data_offsets": [offset, offset + end - begin]}
        slabs.append((fd, start + begin, end - begin))
        offset += end - begin
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    out = sys.stdout.buffer
    out.write(struct.pack("<Q", len(blob)) + blob)
    for fd, begin, size in slabs:
        done = 0
        while done < size:
            chunk = os.pread(fd, min(size - done, 1 << 24), begin + done)
            assert chunk, "truncated shard"
            out.write(chunk)
            done += len(chunk)


if __name__ == "__main__":
    main()
