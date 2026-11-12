"""A compact model directory as real files, without the routed experts its loader never reads.

SRC is an ``omlx_compact.py make`` directory: config, tensor index and ``compact-map.json`` beside
links to a checkpoint's shards. The lattice cache (``omlx_lattice.py``) loads only the few experts
the map names and takes every other routed expert from its codes, so the rest of the FP4 experts
are dead weight in a download. DST gets every file of SRC; a shard that loses nothing is cloned
(``cp -c``, free on APFS) and the others are rewritten with the tensors that stay, names and bytes
untouched. Each rewritten shard is read back and compared with the source by CRC. Standard library
only.

    python slim_checkpoint.py SRC DST
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

EXPERT = re.compile(r"layers\.\d+\.ffn\.experts\.\d+\.")  # a routed expert of a decoder layer (not the drafter's)
CHUNK = 1 << 24


def header(fd: int) -> tuple[dict, int]:
    (size,) = struct.unpack("<Q", os.pread(fd, 8, 0))
    return json.loads(os.pread(fd, size, 8)), 8 + size


def crc(fd: int, begin: int, size: int, out=None) -> int:
    """CRC of ``size`` bytes at ``begin``; with ``out``, the bytes are also written to it."""
    value, done = 0, 0
    while done < size:
        chunk = os.pread(fd, min(size - done, CHUNK), begin + done)
        assert chunk, "truncated shard"
        value, done = zlib.crc32(chunk, value), done + len(chunk)
        if out:
            out.write(chunk)
    return value


def slim(src: Path, dst: Path, keep: set[str]) -> tuple[int, int]:
    """Write shard ``src`` to ``dst`` without its routed experts outside ``keep``. (bytes in, bytes out)."""
    fd = os.open(src, os.O_RDONLY)
    entries, start = header(fd)
    names = [n for n in entries if n != "__metadata__" and (n in keep or not EXPERT.match(n))]
    if len(names) == len(entries) - ("__metadata__" in entries):
        os.close(fd)
        if subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True, check=False).returncode:  # not APFS: a plain copy
            shutil.copyfile(src, dst)
        return src.stat().st_size, src.stat().st_size
    names.sort(key=lambda n: entries[n]["data_offsets"][0])
    out_header, offset = {k: entries[k] for k in ("__metadata__",) if k in entries}, 0
    for n in names:
        begin, end = entries[n]["data_offsets"]
        out_header[n] = {**entries[n], "data_offsets": [offset, offset + end - begin]}
        offset += end - begin
    blob = json.dumps(out_header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(dst, "wb") as out:
        out.write(struct.pack("<Q", len(blob)) + blob)
        sums = [crc(fd, start + begin, end - begin, out) for begin, end in (entries[n]["data_offsets"] for n in names)]
    os.close(fd)
    fd = os.open(dst, os.O_RDONLY)  # read back: the same names, shapes and bytes
    back, start = header(fd)
    assert [n for n in back if n != "__metadata__"] == names
    for n, want in zip(names, sums):
        begin, end = back[n]["data_offsets"]
        assert (back[n]["dtype"], back[n]["shape"]) == (entries[n]["dtype"], entries[n]["shape"]), n
        assert crc(fd, start + begin, end - begin) == want, n
    os.close(fd)
    return src.stat().st_size, dst.stat().st_size


def main() -> None:
    src, dst = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    spec = json.loads((src / "compact-map.json").read_text())
    keep = set(spec["alias"].values())  # stored names of the experts the loader reads
    dst.mkdir(parents=True)
    total_in = total_out = 0
    for item in sorted(src.iterdir()):
        real = item.resolve()
        if item.name in (".cache", "SHA256SUMS"):  # download bookkeeping and sums of the full shards
            continue
        if item.name == "compact-map.json":
            (dst / item.name).write_text(json.dumps(spec | {"source": "."}))
        elif item.name.endswith(".safetensors") and item.name.startswith("model-"):
            a, b = slim(real, dst / item.name, keep)
            total_in, total_out = total_in + a, total_out + b
            print(f"{item.name}: {a / 1e9:.2f} GB -> {b / 1e9:.2f} GB", flush=True)
        elif real.is_dir():
            shutil.copytree(real, dst / item.name)
        else:
            shutil.copyfile(real, dst / item.name)
    print(f"{dst}: {total_in / 1e9:.1f} GB -> {total_out / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
