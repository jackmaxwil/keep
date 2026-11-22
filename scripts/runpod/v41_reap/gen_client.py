"""Send the generation test's requests to a running server and save each response.

    python gen_client.py URL DIR OUT TASK=N [TASK=N ...]

Reads DIR/req-TASK.json, sends N copies of each with up to 24 in flight, and writes
OUT/gen-full-TASK-I.json (I from 1) in the shape score_dupe_script.py reads. Standard library only.
"""

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def send(url: str, body: bytes, dest: Path) -> str:
    if dest.exists():
        return f"{dest.name} kept"
    req = urllib.request.Request(f"{url}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        data = r.read()
    dest.write_bytes(data)
    usage = json.loads(data).get("usage", {})
    return f"{dest.name} {usage.get('completion_tokens')} tokens"


def main(url: str, src: Path, out: Path, jobs: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    work = []
    for job in jobs:
        task, n = job.split("=")
        body = (src / f"req-{task}.json").read_bytes()
        work += [(body, out / f"gen-full-{task}-{i}.json") for i in range(1, int(n) + 1)]
    with ThreadPoolExecutor(24) as pool:
        for line in pool.map(lambda w: send(url, *w), work):
            print(line, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
