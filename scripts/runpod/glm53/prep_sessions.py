"""Session text for ``residency.py``, from the DeepSeek-tokenized pack of the V4.1 case study.

The same texts the first model was scored on: the Mac's 22 evaluation sessions (the first 1,024
DeepSeek tokens of each) and the 52 calibration records (512 tokens), decoded to plain text with
DeepSeek's chat markers removed. ``residency.py`` tokenizes them with the second model's tokenizer.

    python prep_sessions.py --run ~/Models/v41-reap-run --out ~/Models/v41-reap-run/glm53
"""

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

MARKER = re.compile(r"<｜[^｜]*｜>")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run = args.run.expanduser()
    tok = AutoTokenizer.from_pretrained(run / "bundle/models/v41-lattice")
    records: dict[str, list] = {}
    for line in (run / "pack/v41-pack.jsonl").open():
        if line.strip():
            r = json.loads(line)
            records.setdefault(r["session"], []).append(r)

    def text(name: str, tokens: int) -> dict:
        session, _, index = name.partition("#")
        r = records[session][int(index or 0)]
        return {"domain": r["domain"], "text": MARKER.sub("", tok.decode(r["token_ids"][:tokens])).strip()}

    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for file, names, tokens in (("eval.json", "mac-eval-sessions.json", 1024), ("cal.json", "cal-cost-records.json", 512)):
        sessions = {name: text(name, tokens) for name in json.loads((run / names).read_text())}
        (out / file).write_text(json.dumps(sessions))
        print(file, len(sessions), "sessions,", len({s["text"] for s in sessions.values()}), "distinct texts")


if __name__ == "__main__":
    main()
