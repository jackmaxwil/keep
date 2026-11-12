"""D3: decode speed of a running server while another process takes memory.

Sends one request after another to an OpenAI-compatible endpoint. After a baseline it allocates
incompressible memory in steps, holds it, lets it go and keeps measuring. Stepping stops early at
a critical pressure level or at ``--max-swap-gb`` of swap. Writes ``requests.jsonl`` (one line per
request) and ``samples.jsonl`` (every 5 s: memory taken here, swap, pageouts, compressor, free
pages, pressure level, the server's footprint) into ``--out``. Standard library only.

    python d3_pressure.py --pid SERVER_PID --request REQ.json --out DIR
"""

import argparse
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def system(pid: int) -> dict:
    vm = {k.strip('" '): int(v) for k, v in re.findall(r'^(.+?):\s+(\d+)\.$', sh("vm_stat"), re.MULTILINE)}
    page = int(sh("sysctl", "-n", "hw.pagesize"))
    swap = float(re.search(r"used = ([\d.]+)M", sh("sysctl", "-n", "vm.swapusage"))[1]) / 1024
    return {"level": int(sh("sysctl", "-n", "kern.memorystatus_vm_pressure_level")), "swap_gb": round(swap, 2),
            "free_gb": round(vm["Pages free"] * page / 2**30, 1),
            "compressor_gb": round(vm["Pages occupied by compressor"] * page / 2**30, 1),
            "pageouts": vm["Pageouts"], "swapouts": vm["Swapouts"],
            "server": sh("top", "-l", "1", "-pid", str(pid), "-stats", "mem").split()[-1]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:8017/v1/chat/completions")
    p.add_argument("--step-gb", type=int, default=4)
    p.add_argument("--max-gb", type=int, default=40)
    p.add_argument("--max-swap-gb", type=float, default=24)
    p.add_argument("--every", type=int, default=45, help="seconds per step")
    p.add_argument("--rest", type=int, default=120, help="seconds of baseline, of hold and of recovery")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    body, t0, state, held = args.request.read_bytes(), time.time(), {"phase": "baseline", "run": True}, []

    def requests() -> None:
        with (args.out / "requests.jsonl").open("w") as f:
            while state["run"]:
                start, phase, taken = time.time() - t0, state["phase"], len(held)
                req = urllib.request.Request(args.url, body, {"Content-Type": "application/json"})
                try:
                    usage = json.load(urllib.request.urlopen(req, timeout=1800))["usage"]
                except Exception as e:  # noqa: BLE001  (a dead server is a result)
                    usage = {"error": f"{type(e).__name__}: {e}"[:200]}
                    time.sleep(5)
                f.write(json.dumps({"t": round(start, 1), "end": round(time.time() - t0, 1), "phase": phase, "taken_gb": taken,
                                    **{k: usage.get(k) for k in ("completion_tokens", "generation_tokens_per_second",
                                                                 "time_to_first_token", "error") if k in usage}}) + "\n")
                f.flush()

    def samples() -> None:
        with (args.out / "samples.jsonl").open("w") as f:
            while state["run"]:
                state["last"] = system(args.pid)
                f.write(json.dumps({"t": round(time.time() - t0, 1), "phase": state["phase"], "taken_gb": len(held),
                                    **state["last"]}) + "\n")
                f.flush()
                time.sleep(5)

    threads = [threading.Thread(target=fn, daemon=True) for fn in (requests, samples)]
    for t in threads:
        t.start()
    time.sleep(args.rest)
    state["phase"] = "step"
    while len(held) < args.max_gb and state["last"]["level"] < 4 and state["last"]["swap_gb"] < args.max_swap_gb:
        held.extend(os.urandom(1 << 30) for _ in range(args.step_gb))  # random: the compressor gains nothing
        time.sleep(args.every)
    state["phase"] = "hold"
    time.sleep(args.rest)
    held.clear()
    state["phase"] = "recovery"
    time.sleep(args.rest)
    state["run"] = False
    threads[0].join(timeout=300)


if __name__ == "__main__":
    main()
