"""K5: can macOS's page cache hold expert bytes that MLX computes on in place?

The idea under test: map one file holding every expert, alias it as an MLX array, and let the
kernel decide which pages stay in memory. That only works if

1. MLX aliases a file mapping (no copy into its own buffers),
2. using a small part of the aliased buffer on the GPU leaves the rest out of memory.

Result on the M4 Max, mlx 0.32.4 (2026-11-11): 1 holds for every mapping mode (same address, no
page touched by aliasing). 2 fails: a kernel that reads 1 MiB of a 2 GiB aliased buffer brings
every page of it into memory. One buffer over all experts would therefore page in the whole
pool on first use, so the design is dead in this form; the cost of a cold read was not measured
because no read is cold after the first.

Each trial writes its own file with F_NOCACHE, so its pages start out of the page cache
(checked with mincore). Prints one JSON line per measurement.

    ~/Development/_refs/omlx/.venv/bin/python scripts/v41_build/k5_pagecache.py DIR [--gb 2]
"""

import argparse
import ctypes
import fcntl
import json
import mmap
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

PAGE = mmap.PAGESIZE
libc = ctypes.CDLL(None, use_errno=True)


def cold_file(path: Path, nbytes: int) -> None:
    """``nbytes`` of uint16 noise, written past the page cache."""
    chunk = np.random.default_rng(0).integers(0, 1 << 16, 1 << 23, dtype=np.uint16).tobytes()
    assert nbytes % len(chunk) == 0 and len(chunk) % PAGE == 0
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    for _ in range(nbytes // len(chunk)):
        assert os.write(fd, chunk) == len(chunk)
    os.fsync(fd)
    os.close(fd)


def in_memory(m: np.memmap) -> np.ndarray:
    """One flag per page of the mapping: is it in memory (mincore)?"""
    vec = (ctypes.c_char * (m.nbytes // PAGE))()
    if libc.mincore(ctypes.c_void_p(m.ctypes.data), ctypes.c_size_t(m.nbytes), vec):
        raise OSError(ctypes.get_errno(), "mincore")
    return (np.frombuffer(vec, np.uint8) & 1).astype(bool)


def alias(m: np.memmap) -> mx.array:
    a = mx.asarray(m, copy=False)  # raises if MLX would copy
    mx.eval(a)
    return a


def timed_sum(a: mx.array, start: int, count: int) -> float:
    t = time.perf_counter()
    mx.eval(a[start : start + count].astype(mx.float32).sum())
    return (time.perf_counter() - t) * 1e3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", type=Path)
    p.add_argument("--gb", type=int, default=2)
    args = p.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    nbytes, mb = args.gb << 30, (1 << 20) // 2  # mb: uint16 elements per MiB
    say = lambda **kw: print(json.dumps(kw), flush=True)
    say(mlx=mx.__version__, page=PAGE, file_gb=args.gb, wired_limit=mx.set_wired_limit(0))
    mx.eval(mx.zeros(8).sum())  # Metal is up before anything is timed

    # 1. Aliasing, for each way of mapping the file.
    modes = {}
    for mode in ("r", "r+", "c"):
        path = args.dir / f"alias-{mode.replace('+', 'w')}.bin"
        cold_file(path, nbytes)
        m = np.memmap(path, dtype=np.uint16, mode=mode)
        cold = float(in_memory(m).mean())
        active = mx.get_active_memory()
        try:
            a = alias(m)
        except Exception as e:  # noqa: BLE001  (the message is the result)
            say(test="alias", mode=mode, ok=False, error=f"{type(e).__name__}: {e}"[:200])
            continue
        same = np.asarray(memoryview(a)).ctypes.data == m.ctypes.data
        say(test="alias", mode=mode, ok=True, same_address=bool(same), cold_before=cold,
            in_memory_after_alias=float(in_memory(m).mean()), mlx_active_added_mb=(mx.get_active_memory() - active) >> 20)
        if same:
            modes[mode] = path
        del a, m
    if not modes:
        say(verdict="dead: MLX copies every mapping from Python; try allocator::make_buffer natively")
        return
    mode = next(iter(modes))

    # 2. Does touching 1 MiB on the GPU bring the whole buffer into memory?
    for where in (0, nbytes // 4):  # element offsets: the start and the middle of the file
        path = args.dir / f"touch-{where}.bin"  # a new file each time: a rewritten one keeps its cached pages
        cold_file(path, nbytes)
        m = np.memmap(path, dtype=np.uint16, mode=mode)
        a = alias(m)
        before = in_memory(m)
        ms = timed_sum(a, where, mb)
        after = in_memory(m)
        say(test="touch-1MiB", at_gb=where * 2 / (1 << 30), ms=round(ms, 2), in_memory_before=float(before.mean()),
            in_memory_after=float(after.mean()), pages_after=int(after.sum()), pages_needed=(mb * 2) // PAGE)
        del a, m

    for f in args.dir.glob("*.bin"):
        f.unlink()


if __name__ == "__main__":
    main()
