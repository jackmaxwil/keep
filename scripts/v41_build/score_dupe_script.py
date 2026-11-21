"""Score sampled answers to the script requests of the generation tests.

An answer is correct when its first Python code block runs on a fixed fixture and prints what the
task asks for. Tasks (``--task``, default ``dupes``; ``--prompt TASK`` prints a task's request):

- ``dupes``    the duplicate-finder request (k1/creq-nothink.json): the three duplicate files of a tree
- ``csvsum``   sums of the numeric columns of a CSV file, as one line of JSON
- ``largest``  the three largest files under a directory, largest first

    python score_dupe_script.py [--task TASK] RESPONSE.json [...]   # one line per file, then the count
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


PROMPTS = {
    "csvsum": "Write a Python script that takes a CSV file path as its only argument and prints one line of JSON mapping each "
              "numeric column name to the sum of that column. Ignore columns that are not numeric. Include a main guard and argparse.",
    "largest": "Write a Python script that takes a directory path and prints the three largest files under it, largest first, "
               "one per line: the size in bytes, a tab, then the path. Walk subdirectories. Include a main guard and argparse.",
}
FLAGS = {"dupes": ["--directory", "--dir", "--path", "--root"], "csvsum": ["--file", "--csv", "--input", "--path"],
         "largest": ["--directory", "--dir", "--path", "--root"]}


def fixture(task: str, root: Path) -> Path:
    """Build the task's fixture under ``root`` and return the path the script is given."""
    if task == "dupes":
        tree(root)
        return root
    root.mkdir(parents=True)
    if task == "csvsum":
        (root / "data.csv").write_text("name,qty,price\na,2,1.5\nb,3,2.5\nc,5,4.0\n")
        return root / "data.csv"
    (root / "sub").mkdir()
    for name, size in (("a.bin", 5000), ("sub/b.bin", 3000), ("sub/c.txt", 1000), ("d.txt", 10)):
        (root / name).write_bytes(b"x" * size)
    return root


def judge(task: str, out: str) -> str:
    """Empty when ``out`` is what the task asks for, else what is wrong."""
    if task == "dupes":
        missing, extra = [d for d in DUPES if d not in out], [u for u in UNIQUE if u in out]
        return f"wrong output (missing {missing}, extra {extra})" if missing or extra else ""
    if task == "csvsum":
        found = re.findall(r"\{[^{}]*\}", out)
        try:
            got = json.loads(found[-1]) if found else None
        except json.JSONDecodeError:
            got = None
        ok = isinstance(got, dict) and set(got) == {"qty", "price"} and abs(float(got["qty"]) - 10) < 1e-6 and abs(float(got["price"]) - 8) < 1e-6
        return "" if ok else f"wrong output: {out.strip()[:60]!r}"
    lines = [line for line in out.splitlines() if line.strip()]
    want = [("5000", "a.bin"), ("3000", "b.bin"), ("1000", "c.txt")]
    ok = len(lines) >= 3 and all(size in line and name in line for (size, name), line in zip(want, lines[-3:] if len(lines) > 3 else lines))
    return "" if ok and "d.txt" not in out else f"wrong output: {out.strip()[:60]!r}"


def score(answer: str, root: Path, task: str = "dupes") -> tuple[bool, str]:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", answer, re.DOTALL)
    if not m:
        return False, "no code block"
    script = root.parent / "answer.py"
    script.write_text(m.group(1))
    target = fixture(task, root)
    for extra in ([], *([f] for f in FLAGS[task])):
        try:
            r = subprocess.run([sys.executable, "-I", str(script), *extra, str(target)], capture_output=True, text=True,
                               timeout=30, cwd=root.parent)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if r.returncode != 2:  # 2: argparse rejected the arguments; try the next spelling
            break
    out = r.stdout
    if r.returncode != 0:
        return False, f"exit {r.returncode}: {(r.stderr.strip().splitlines() or [''])[-1][:80]}"
    wrong = judge(task, out)
    return (False, wrong) if wrong else (True, "ok")


def main(paths: list[str]) -> int:
    task = "dupes"
    if paths[:1] == ["--prompt"]:
        print(PROMPTS[paths[1]])
        return 0
    if paths[:1] == ["--task"]:
        task, paths = paths[1], paths[2:]
    good = 0
    for p in paths:
        r = json.loads(Path(p).read_text())
        try:
            answer = r["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            answer = ""
        with tempfile.TemporaryDirectory() as d:
            ok, why = score(answer, Path(d) / "tree", task)
        good += ok
        print(f"{Path(p).name}\t{'PASS' if ok else 'FAIL'}\t{why}")
    print(f"correct {good} of {len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
