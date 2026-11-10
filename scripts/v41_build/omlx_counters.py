"""Run the oMLX CLI with the V4.1 expert-slot counters written to a file.

oMLX's server logs expert-cache hits and misses per request only for its generic offload wrapper;
the V4.1 wrapper keeps the same counters but nothing reads them. This starts ``omlx`` unchanged
and appends one JSON line to ``$OMLX_COUNTERS`` whenever the totals move:

    OMLX_COUNTERS=/tmp/counters.jsonl python omlx_counters.py serve --model-dir ...
"""

import json
import os
import sys
import threading
import time

from omlx.cli import main
from omlx.patches.deepseek_v41 import moe_offload

slots, made = [], moe_offload._ExpertSlots.__init__


def init(self, *args, **kwargs):
    made(self, *args, **kwargs)
    slots.append(self)


def watch(path):
    last = None
    while True:
        time.sleep(0.5)
        now = [sum(getattr(s, f) for s in slots) for f in ("hits", "misses", "fetched_bytes")]
        if slots and now != last:
            last = now
            with open(path, "a") as f:
                f.write(json.dumps({"t": round(time.time(), 2), "layers": len(slots), "capacity": slots[0].plan.capacity,
                                    "hits": now[0], "misses": now[1], "fetched_mb": round(now[2] / 1e6, 1)}) + "\n")


moe_offload._ExpertSlots.__init__ = init
threading.Thread(target=watch, args=(os.environ["OMLX_COUNTERS"],), daemon=True).start()
sys.exit(main())
