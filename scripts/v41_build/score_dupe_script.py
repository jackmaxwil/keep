"""Score sampled answers to the duplicate-finder request (k1/creq-nothink.json).

An answer is correct when its first Python code block runs on a fixed test tree and prints the
three duplicate files of that tree, and none of the unique files.

    python score_dupe_script.py RESPONSE.json [...]   # one line per file, then the count
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DUPES = ["alpha.txt", "copy_of_alpha.txt", "deep_alpha.dat"]
UNIQUE = ["beta.txt", "gamma.log", "empty_dir_marker.txt"]


def tree(root: Path) -> None:
    (root / "sub" / "deeper").mkdir(parents=True)
    (root / "alpha.txt").write_text("same bytes\n")
    (root / "sub" / "copy_of_alpha.txt").write_text("same bytes\n")
    (root / "sub" / "deeper" / "deep_alpha.dat").write_text("same bytes\n")
    (root / "beta.txt").write_text("different\n")
    (root / "sub" / "gamma.log").write_text("also different\n")
    (root / "sub" / "empty_dir_marker.txt").write_text("third unique\n")


def score(answer: str, root: Path) -> tuple[bool, str]:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", answer, re.S)
    if not m:
        return False, "no code block"
    script = root.parent / "answer.py"
    script.write_text(m.group(1))
    for extra in ([], ["--directory"], ["--dir"], ["--path"], ["--root"]):
        try:
            r = subprocess.run([sys.executable, "-I", str(script), *extra, str(root)], capture_output=True, text=True,
                               timeout=30, cwd=root.parent)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if r.returncode != 2:  # 2: argparse rejected the arguments; try the next spelling
            break
    out = r.stdout
    if r.returncode != 0:
        return False, f"exit {r.returncode}: {(r.stderr.strip().splitlines() or [''])[-1][:80]}"
    missing = [d for d in DUPES if d not in out]
    extra = [u for u in UNIQUE if u in out]
    if missing or extra:
        return False, f"wrong output (missing {missing}, extra {extra})"
    return True, "ok"


def main(paths: list[str]) -> int:
    good = 0
    for p in paths:
        r = json.loads(Path(p).read_text())
        try:
            answer = r["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            answer = ""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "tree"
            tree(root)
            ok, why = score(answer, root)
        good += ok
        print(f"{Path(p).name}\t{'PASS' if ok else 'FAIL'}\t{why}")
    print(f"correct {good} of {len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
