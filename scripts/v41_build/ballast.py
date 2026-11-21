"""Hold wired memory so that a larger Mac behaves like a smaller one.

    python ballast.py GIB [READY_FILE]

Maps GIB gibibytes of anonymous memory and wires them with ``mlock``, so the system can neither
compress nor swap them: what is left for everything else is the machine's memory minus GIB. Writes
READY_FILE once the memory is held, then sleeps until killed. macOS refuses past
``vm.global_user_wire_limit`` (about 85% of memory); the script then exits with status 1.

This emulates the memory of a smaller machine, not the machine: the GPU's recommended working set
and every limit a runtime derives from ``hw.memsize`` still describe the larger one.
"""

import ctypes
import mmap
import signal
import sys
from pathlib import Path


def hold(gib: float) -> mmap.mmap:
    size = int(gib * 2**30)
    m = mmap.mmap(-1, size)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if libc.mlock(ctypes.addressof(ctypes.c_char.from_buffer(m)), size):
        raise OSError(ctypes.get_errno(), f"mlock of {gib} GiB refused")
    return m


if __name__ == "__main__":
    try:
        held = hold(float(sys.argv[1]))
    except OSError as e:
        raise SystemExit(f"ballast: {e}")
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(f"{sys.argv[1]}\n")
    print(f"holding {sys.argv[1]} GiB wired", flush=True)
    signal.pause()
