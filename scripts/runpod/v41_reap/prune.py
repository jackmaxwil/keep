"""Write a REAP-pruned copy of a DeepSeek-V4.1 HF checkpoint, byte for byte.

Streams each source shard and writes a shard of the same name holding:

- every untouched tensor, copied unchanged;
- the kept routed experts, renumbered ``0..keep-1`` in ascending source order;
- the router ``weight`` / ``bias`` / ``bias_vl`` rows of the kept experts.

Tensors are copied as raw safetensors bytes, so FP8 weights, ue8m0 scales and
packed FP4 experts are never decoded and nothing needs torch. The plan comes
from ``keep.quality.reap.prune_checkpoint_plan``; the drafter (``mtp.*``) is
left untouched unless ``--mtp-survivors`` is given. ``config.json`` and
``inference/config.json`` get the new ``n_routed_experts``; the index and a
sha256 manifest are rewritten.

Usage::

    python scripts/runpod/v41_reap/prune.py --src /workspace/v41-hf \\
        --survivors /workspace/out/survivors-192.json --out /workspace/v41-reap192
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from keep.quality.reap import prune_checkpoint_plan

INDEX = "model.safetensors.index.json"
COPY_CHUNK = 64 << 20


def read_header(handle: BinaryIO) -> tuple[dict[str, Any], int]:
    (size,) = struct.unpack("<Q", handle.read(8))
    return json.loads(handle.read(size)), 8 + size


def copy_range(src: BinaryIO, dst: BinaryIO, start: int, length: int) -> None:
    src.seek(start)
    while length:
        chunk = src.read(min(COPY_CHUNK, length))
        if not chunk:
            raise OSError("source shard ended early")
        dst.write(chunk)
        length -= len(chunk)


def prune_shard(src_path: Path, dst_path: Path, plan: dict[str, Any]) -> list[str]:
    """Write one pruned shard; return the tensor names it holds."""

    rename, slice_rows, drop = plan["rename"], plan["slice_rows"], set(plan["drop"])
    with src_path.open("rb") as src:
        header, base = read_header(src)
        metadata = header.pop("__metadata__", None)
        jobs: list[tuple[str, dict[str, Any], list[tuple[int, int]]]] = []
        for name in sorted(header, key=lambda n: header[n]["data_offsets"][0]):
            info = header[name]
            start, end = info["data_offsets"]
            if name in drop:
                continue
            if name in slice_rows:
                rows = info["shape"][0]
                row_bytes, rem = divmod(end - start, rows)
                if rem:
                    raise ValueError(f"{name}: {end - start} bytes is not {rows} equal rows")
                kept = slice_rows[name]
                ranges = [(base + start + r * row_bytes, row_bytes) for r in kept]
                shape = [len(kept), *info["shape"][1:]]
                jobs.append((name, {"dtype": info["dtype"], "shape": shape}, ranges))
            else:
                jobs.append((rename.get(name, name), {"dtype": info["dtype"], "shape": info["shape"]}, [(base + start, end - start)]))
        out_header: dict[str, Any] = {} if metadata is None else {"__metadata__": metadata}
        offset = 0
        for name, info, ranges in jobs:
            length = sum(n for _, n in ranges)
            out_header[name] = {**info, "data_offsets": [offset, offset + length]}
            offset += length
        blob = json.dumps(out_header, separators=(",", ":")).encode()
        blob += b" " * (-len(blob) % 8)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst_path.with_name(dst_path.name + ".tmp")
        with tmp.open("wb") as dst:
            dst.write(struct.pack("<Q", len(blob)))
            dst.write(blob)
            for _, _, ranges in jobs:
                for start, length in ranges:
                    copy_range(src, dst, start, length)
        tmp.replace(dst_path)
    return [name for name, _, _ in jobs]


def set_expert_count(node: Any, keep: int) -> int:
    """Set every ``n_routed_experts`` key (not the drafter's ``dspark_*``); return how many."""

    hits = 0
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "n_routed_experts":
                node[key] = keep
                hits += 1
            else:
                hits += set_expert_count(value, keep)
    elif isinstance(node, list):
        hits += sum(set_expert_count(item, keep) for item in node)
    return hits


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_survivors(path: Path) -> dict[int, list[int]]:
    payload = json.loads(path.read_text())
    if payload.get("record_type") != "keep_reap_survivors_v1":
        raise ValueError(f"{path} is not a keep_reap_survivors_v1 table")
    return {int(block): [int(e) for e in kept] for block, kept in payload["survivors"].items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path, help="HF checkpoint directory")
    parser.add_argument("--survivors", required=True, type=Path, help="reap-select output for the backbone layers")
    parser.add_argument("--mtp-survivors", type=Path, help="optional survivor table for the mtp.* drafter")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--skip-shard", action="append", default=[],
                        help="a shard the plan leaves untouched (the Engram tables): indexed but not written; use the source's file")
    args = parser.parse_args(argv)

    index = json.loads((args.src / INDEX).read_text())
    weight_map: dict[str, str] = index["weight_map"]
    survivors = load_survivors(args.survivors)
    keep = len(next(iter(survivors.values())))
    plan = prune_checkpoint_plan(weight_map, survivors, prefix="layers")
    if args.mtp_survivors:
        mtp = prune_checkpoint_plan(plan["untouched"], load_survivors(args.mtp_survivors), prefix="mtp")
        plan = {
            "rename": {**plan["rename"], **mtp["rename"]},
            "slice_rows": {**plan["slice_rows"], **mtp["slice_rows"]},
            "drop": plan["drop"] + mtp["drop"],
        }
        # ponytail: the drafter's expert count lives under dspark_n_routed_experts; set it by hand if you prune it.

    args.out.mkdir(parents=True, exist_ok=True)
    new_map: dict[str, str] = {}
    total = 0
    for shard in sorted(set(weight_map.values())):
        if shard in args.skip_shard:
            with (args.src / shard).open("rb") as handle:
                names = [n for n in read_header(handle)[0] if n != "__metadata__"]
            touched = set(names) & (set(plan["rename"]) | set(plan["slice_rows"]) | set(plan["drop"]))
            if touched:
                raise ValueError(f"{shard} cannot be skipped: the plan changes {len(touched)} of its tensors")
            total += (args.src / shard).stat().st_size
        else:
            names = prune_shard(args.src / shard, args.out / shard, plan)
            total += (args.out / shard).stat().st_size
        new_map.update({name: shard for name in names})
        print(f"{shard}: {len(names)} tensors", flush=True)

    for path in sorted(args.src.rglob("*")):
        rel = path.relative_to(args.src)
        if path.is_dir() or path.suffix == ".safetensors" or rel.name == INDEX or rel.parts[0].startswith("."):
            continue
        target = args.out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel.name == "config.json":
            config = json.loads(path.read_text())
            if set_expert_count(config, keep) == 0:
                raise ValueError(f"{rel}: no n_routed_experts key to update")
            target.write_text(json.dumps(config, indent=4) + "\n")
        else:
            shutil.copy2(path, target)

    expected = len(weight_map) - len(plan["drop"])
    if len(new_map) != expected:
        raise AssertionError(f"wrote {len(new_map)} tensors, expected {expected}")
    out_index = {"metadata": {**index.get("metadata", {}), "total_size": total}, "weight_map": dict(sorted(new_map.items()))}
    (args.out / INDEX).write_text(json.dumps(out_index, indent=2) + "\n")
    (args.out / "keep-prune.json").write_text(
        json.dumps(
            {
                "record_type": "keep_reap_pruned_checkpoint_v1",
                "source_index_sha256": sha256_file(args.src / INDEX),
                "survivors_sha256": sha256_file(args.survivors),
                "keep": keep,
                "tensors_dropped": len(plan["drop"]),
                "router_tensors_sliced": len(plan["slice_rows"]),
                "experts_renamed": len(plan["rename"]),
            },
            indent=2,
        )
        + "\n"
    )
    manifest = {
        str(p.relative_to(args.out)): sha256_file(p)
        for p in sorted(args.out.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS"
    }
    (args.out / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in manifest.items()))
    print(json.dumps({"tensors": len(new_map), "bytes": total, "keep": keep}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
