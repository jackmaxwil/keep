"""A LiveCodeBench code-generation subset against any OpenAI-compatible server.

The problems are the most recent ``--n`` (default 50) of a ``code_generation_lite`` release file, by contest date.
Prompts, answer extraction (the last fenced block) and grading (every public and private test, 6 s a test,
stripped-line or decimal equality for stdin problems, ``==`` on the JSON value for call-based ones) follow the
benchmark's own runner. Each answer runs in a subprocess under ``sandbox-exec`` with no network and writes only
in a temporary directory, so scoring is done on macOS, also for answers generated elsewhere.

    python lcb.py gen --data test6.jsonl --url http://127.0.0.1:8017/v1 --model NAME --out DIR [--extra JSON]
    python lcb.py score --data test6.jsonl DIR          # writes DIR/scores.json
    python lcb.py compare A/scores.json B/scores.json   # paired by problem
    python lcb.py demo                                  # self-check of the scorer
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will generate a "
          "correct Python program that matches the specification and passes all tests.")
WITH_STARTER = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
WITHOUT_STARTER = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the "
                   "sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, "
                   "it reads the inputs, runs the algorithm and writes output to STDOUT.")
MODULES = "string re datetime collections heapq bisect copy math random statistics itertools functools operator io sys json"
PRELUDE = "".join(f"from {m} import *\n" for m in MODULES.split() + ["builtins", "typing"]) + \
    "".join(f"import {m}\n" for m in MODULES.split()) + "sys.setrecursionlimit(50000)\n"
TIMEOUT = 6
HARNESS = '''
import json as _j, signal as _s, sys as _y
def _run():
    tests, fn, mark = _j.load(open(_y.argv[1])), _y.argv[2], _y.argv[3]
    def _alarm(*_): raise TimeoutError
    _s.signal(_s.SIGALRM, _alarm)
    for t in tests:
        _s.alarm(TIMEOUT)
        got = getattr(Solution(), fn)(*[_j.loads(line) for line in t["input"].split("\\n")])
        _s.alarm(0)
        if isinstance(got, tuple): got = list(got)
        if got != _j.loads(t["output"]): _y.exit(1)
    print(mark)
_run()
'''.replace("TIMEOUT", str(TIMEOUT))


class _NoGlobals(pickle.Unpickler):  # the private tests are a pickled str; nothing else may load
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"{module}.{name}")


def problems(path: str, n: int) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    rows.sort(key=lambda r: (r["contest_date"], r["question_id"]))
    return rows[-n:]


def tests_of(row: dict) -> list[dict]:
    private = row["private_test_cases"]
    try:
        private = json.loads(private)
    except json.JSONDecodeError:
        private = json.loads(_NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(private)))).load())
    return json.loads(row["public_test_cases"]) + private


def prompt(row: dict) -> str:
    text = f"### Question:\n{row['question_content']}\n\n"
    if row["starter_code"]:
        text += f"### Format: {WITH_STARTER}\n```python\n{row['starter_code']}\n```\n\n"
    else:
        text += f"### Format: {WITHOUT_STARTER}\n```python\n# YOUR CODE HERE\n```\n\n"
    return text + "### Answer: (use the provided format with backticks)\n\n"


def extract(answer: str) -> str:
    lines = answer.rsplit("</think>", 1)[-1].split("\n")
    marks = [i for i, line in enumerate(lines) if "```" in line]
    return "\n".join(lines[marks[-2] + 1:marks[-1]]) if len(marks) >= 2 else ""


def same_output(got: str, want: str) -> bool:
    a, b = ([line.strip() for line in v.strip().split("\n")] for v in (got, want))
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != y:
            try:
                if [Decimal(e) for e in x.split()] != [Decimal(e) for e in y.split()]:
                    return False
            except ArithmeticError:
                return False
    return True


def grade(code: str, tests: list[dict], fn_name: str | None, sandbox: bool = True) -> bool:
    """True when ``code`` passes every test. Runs it in a subprocess, sandboxed, in a fresh directory."""
    if not code.strip():
        return False
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        cmd = [sys.executable, "-I", "sol.py"]
        if sandbox:
            profile = f'(version 1)(allow default)(deny network*)(deny file-write*)(allow file-write* (subpath "{tmp}") (literal "/dev/null"))'
            cmd = ["sandbox-exec", "-p", profile] + cmd

        def run(args: list[str], stdin: str, seconds: float) -> str | None:
            try:  # ponytail: no memory limit (RLIMIT_AS does not hold on macOS); a runaway answer is cut by the timeout
                r = subprocess.run(cmd + args, input=stdin, capture_output=True, text=True, timeout=seconds, cwd=tmp, check=False)
            except subprocess.TimeoutExpired:
                return None
            return r.stdout if r.returncode == 0 else None

        if fn_name:
            (tmp / "sol.py").write_text(PRELUDE + "\n" + code + "\n" + HARNESS)
            (tmp / "tests.json").write_text(json.dumps(tests))
            mark = f"lcb-pass-{random.getrandbits(64):x}"
            out = run(["tests.json", fn_name, mark], "", min(TIMEOUT * len(tests), 600) + 5)
            return out is not None and out.strip().endswith(mark)
        (tmp / "sol.py").write_text(PRELUDE + "\n" + code + "\n")
        for t in tests:
            out = run([], t["input"], TIMEOUT + 2)
            if out is None or not same_output(out, t["output"]):
                return False
        return True


def gen(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extra = json.loads(args.extra)
    jobs = [(row, s) for row in problems(args.data, args.n) for s in range(args.samples)]

    def one(job: tuple[dict, int]) -> None:
        row, s = job
        path = out / f"{row['question_id']}-{s}.json"
        if path.exists():
            return
        body = {"model": args.model, "temperature": 1.0, "top_p": 1.0, "max_tokens": args.max_tokens,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(row)}], **extra}
        t = time.time()
        req = urllib.request.Request(args.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=args.timeout))
        r["_seconds"] = time.time() - t
        path.write_text(json.dumps(r))
        u = r.get("usage", {})
        print(path.name, u.get("completion_tokens"), "tokens", round(r["_seconds"]), "s", r["choices"][0].get("finish_reason"), flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, jobs))


def score(args: argparse.Namespace) -> None:
    if not args.no_sandbox and not shutil.which("sandbox-exec"):
        sys.exit("sandbox-exec not found: score on macOS (or pass --no-sandbox inside a throwaway container)")
    out, rows = Path(args.dir), problems(args.data, args.n)

    def one(row: dict) -> tuple[str, list]:
        fn = json.loads(row["metadata"] or "{}").get("func_name")
        got = []
        for path in sorted(out.glob(f"{row['question_id']}-*.json")):
            message = json.loads(path.read_text())["choices"][0]["message"]
            got.append(grade(extract(message.get("content") or ""), tests_of(row), fn, not args.no_sandbox))
        return row["question_id"], got

    with ThreadPoolExecutor(args.workers) as pool:
        solved = dict(pool.map(one, rows))
    files = [json.loads(p.read_text()) for p in out.glob("*-*.json")]
    tokens, seconds = (sum(v) for v in zip(*[(f.get("usage", {}).get("completion_tokens", 0), f["_seconds"]) for f in files])) if files else (0, 0)
    trials = sum(len(v) for v in solved.values())
    result = {"data": Path(args.data).name, "problems": len(rows), "dates": [rows[0]["contest_date"][:10], rows[-1]["contest_date"][:10]],
              "trials": trials, "passed": sum(map(sum, solved.values())), "completion_tokens": tokens, "seconds": round(seconds),
              "truncated": sum(f["choices"][0].get("finish_reason") == "length" for f in files),
              "python": sys.version.split()[0], "solved": solved,
              "difficulty": {r["question_id"]: r["difficulty"] for r in rows}}
    (out / "scores.json").write_text(json.dumps(result, indent=1))
    print(f"{result['passed']}/{trials} trials pass; {result['truncated']} truncated; problems dated {result['dates'][0]} to {result['dates'][1]}")
    for level in ("easy", "medium", "hard"):
        v = [x for q, xs in solved.items() if result["difficulty"][q] == level for x in xs]
        print(f"  {level}: {sum(v)}/{len(v)}")


def compare(args: argparse.Namespace) -> None:
    a, b = (json.loads(Path(p).read_text())["solved"] for p in (args.a, args.b))
    ids = sorted(q for q in a if a[q] and b.get(q))
    d = [sum(a[q]) / len(a[q]) - sum(b[q]) / len(b[q]) for q in ids]
    rng = random.Random(0)
    boot = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(10000))
    rate = lambda s: sum(sum(s[q]) / len(s[q]) for q in ids) / len(ids)
    print(f"{len(ids)} problems: A {rate(a):.3f}  B {rate(b):.3f}  A-B {sum(d) / len(d):+.3f} [{boot[249]:+.3f}, {boot[9749]:+.3f}]"
          f"  A better on {sum(x > 0 for x in d)}, B better on {sum(x < 0 for x in d)}")


def demo() -> None:
    stdin = [{"input": "1 2\n", "output": "3\n"}, {"input": "5 5", "output": "10.0"}]
    call = [{"input": "[1, 2, 3]\n2", "output": "[3, 4, 5]"}]
    add = "a, b = map(int, input().split())\nprint(a + b)"
    assert grade(add, stdin, None)
    assert not grade(add.replace("a + b", "a - b"), stdin, None)
    assert not grade("while True: pass", stdin[:1], None)
    assert not grade("import socket\nsocket.create_connection(('1.1.1.1', 53), 2)\n" + add, stdin, None), "network must be denied"
    assert not grade(f"open({str(Path.home() / 'lcb-sandbox-probe')!r}, 'w')\n" + add, stdin, None), "writes outside tmp must be denied"
    assert not (Path.home() / "lcb-sandbox-probe").exists()
    sol = "class Solution:\n    def shift(self, xs: List[int], k: int) -> List[int]:\n        return %s"
    assert grade(sol % "[x + k for x in xs]", call, "shift") and grade(sol % "tuple(x + k for x in xs)", call, "shift")
    assert not grade(sol % "xs", call, "shift") and not grade(sol % "print('lcb-pass') or exit(0)", call, "shift")
    assert extract("first\n```python\nold\n```\ntext\n```python\nnew\n```\n") == "new" and extract("no code") == ""
    packed = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(stdin)))).decode()
    row = {"public_test_cases": json.dumps(call), "private_test_cases": packed}
    assert tests_of(row) == call + stdin
    try:
        _NoGlobals(io.BytesIO(pickle.dumps(Path("x")))).load()
        raise AssertionError("a pickle with globals must not load")
    except pickle.UnpicklingError:
        pass
    assert "# YOUR CODE HERE" in prompt({"question_content": "q", "starter_code": ""})
    print("lcb demo ok")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--data", required=True)
    g.add_argument("--url", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=50)
    g.add_argument("--samples", type=int, default=2)
    g.add_argument("--max-tokens", type=int, default=16384)
    g.add_argument("--extra", default="{}", help="JSON merged into each request body, e.g. the server's thinking switch")
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--timeout", type=float, default=7200)
    s = sub.add_parser("score")
    s.add_argument("dir")
    s.add_argument("--data", required=True)
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--no-sandbox", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    sub.add_parser("demo")
    args = p.parse_args()
    {"gen": gen, "score": score, "compare": compare, "demo": lambda _: demo()}[args.cmd](args)


if __name__ == "__main__":
    main()
