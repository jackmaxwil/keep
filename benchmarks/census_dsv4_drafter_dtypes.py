"""Census the pinned DSV4-Flash drafter tensor dtypes straight from shard headers.

Reproduces the Task 7 finding in docs/deepseek-v4-flash/2026-08-21-compounded-speed-verdict.md: the
mtp.{0,1,2} routed experts already ship as native FP4 (I8 packed two-per-byte
with F8_E8M0 group scales), so there are no dense BF16 experts to convert.
"""

from __future__ import annotations

import collections
import json
import re
import struct
import sys
from pathlib import Path

CHECKPOINT = Path("/Users/jack.mazac/models/DeepSeek-V4-Flash-0731")


def shard_header(checkpoint: Path, shard: str, cache: dict) -> dict:
    if shard not in cache:
        with open(checkpoint / shard, "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            cache[shard] = json.loads(handle.read(length))
    return cache[shard]


def main(checkpoint: Path = CHECKPOINT) -> int:
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    cache: dict = {}
    tensors = collections.Counter()
    payload = collections.Counter()
    expert_i8 = 0
    bf16_names = []
    per_block = collections.Counter()

    for name, shard in weight_map.items():
        if not name.startswith("mtp"):
            continue
        header = shard_header(checkpoint, shard, cache)[name]
        dtype = header["dtype"]
        start, end = header["data_offsets"]
        tensors[dtype] += 1
        payload[dtype] += end - start
        if dtype == "I8":
            per_block[re.match(r"mtp\.(\d+)", name).group(1)] += 1
            if ".ffn.experts." in name:
                expert_i8 += 1
        elif dtype == "BF16":
            bf16_names.append(name)

    for dtype, count in tensors.most_common():
        print(f"{dtype:10s} tensors={count:5d} bytes={payload[dtype] / 1e9:8.3f} GB")
    print(
        f"total      tensors={sum(tensors.values()):5d} bytes={sum(payload.values()) / 1e9:8.3f} GB"
    )
    print(f"routed FP4 expert tensors: {expert_i8}  per-block I8: {dict(per_block)}")
    print(f"BF16 expert tensors: {sum('.ffn.experts.' in n for n in bf16_names)}")
    print("BF16 tensors are norms/gates/heads only:")
    for name in bf16_names:
        print("  ", name)

    assert expert_i8 == 2304, expert_i8
    assert sum(".ffn.experts." in n for n in bf16_names) == 0
    print("\nOK: drafter routed experts are already native FP4; nothing to convert.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
