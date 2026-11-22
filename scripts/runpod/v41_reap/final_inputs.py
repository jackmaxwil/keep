"""Inputs for final_pod.sh, built on the Mac from files it already has.

    python final_inputs.py OUT

OUT/sessions-22.jsonl  the 22 tuning sessions of the paper, cut to their first 1,024 tokens, in the
                       pack format collect.py reads (the Mac's local_eval.py --max-tokens 1024 scores
                       the same 1,023 positions per session)
OUT/fm-192.json        the 192 experts per layer of largest routed mass (x1/fm-192.json)
OUT/reap-192.json      the pruned checkpoint's 192 (x1/survivors-192.json)
OUT/req-TASK.json      the three script requests of the generation test, at the Mac rows' settings:
                       temperature 1.0, top-p 1.0, 4,096 tokens, thinking off
OUT/expected-ids.json  each request's prompt rendered by the checkpoint's own encoding/encoding.py
                       and tokenizer, for the pod to compare with what vLLM renders
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

RUN = Path("~/Models/v41-reap-run").expanduser()
MODEL = RUN / "bundle/models/v41-lattice"  # tokenizer and encoding/ are the checkpoint's own
SCORER = Path(__file__).resolve().parents[2] / "v41_build/score_dupe_script.py"


def prompt(task: str) -> str:
    if task == "dupes":
        return json.loads((RUN / "k1/creq-nothink.json").read_text())["messages"][0]["content"]
    return subprocess.run([sys.executable, str(SCORER), "--prompt", task], capture_output=True, text=True, check=True).stdout.strip()


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    keep = json.loads((RUN / "mac-eval-sessions.json").read_text())
    with (RUN / "pack/v41-pack.jsonl").open() as src, (out / "sessions-22.jsonl").open("w") as dst:
        for line in src:
            r = json.loads(line)
            if r["split"] == "eval" and r["session"] in keep:
                r["token_ids"] = r["token_ids"][:1024]
                r["target_spans"] = [[1, len(r["token_ids"])]]
                dst.write(json.dumps(r) + "\n")
    assert sum(1 for _ in (out / "sessions-22.jsonl").open()) == len(keep) == 22
    shutil.copy(RUN / "x1/fm-192.json", out / "fm-192.json")
    shutil.copy(RUN / "x1/survivors-192.json", out / "reap-192.json")

    sys.path.insert(0, str(MODEL / "encoding"))
    import encoding
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    expected = {}
    for task in ("dupes", "csvsum", "largest"):
        messages = [{"role": "user", "content": prompt(task)}]
        body = {"model": "full", "messages": messages, "temperature": 1.0, "top_p": 1.0, "max_tokens": 4096,
                "chat_template_kwargs": {"thinking": False}}
        (out / f"req-{task}.json").write_text(json.dumps(body))
        text = encoding.encode_messages(messages, thinking_mode="chat")
        expected[task] = tok.encode(text, add_special_tokens=False)
    (out / "expected-ids.json").write_text(json.dumps(expected))
    print(f"wrote {out}: 22 sessions, 3 requests, prompts of {[len(v) for v in expected.values()]} tokens")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
