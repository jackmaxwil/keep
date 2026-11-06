"""Build the V4.1 REAP calibration + held-out eval pack from public datasets.

Mix and rationale: docs/research/2026-11-05-v41-reap-calibration-data.md
(section 3). English only, by policy: any record containing CJK, Hangul or
Kana characters, or more than 2% non-Latin letters, is dropped.

Every chat source is rendered with V4.1's own ``encoding/encoding.py`` (tools
on the system message, DSML tool calls) and tokenized with the V4.1
tokenizer. Raw documents go in as BOS + text. Records are capped at
``--max-tokens`` (hard truncation). Short documents are packed to about
``--pack-tokens`` with EOS separators.

Split: each record's group key (repo, task, RFC number, URL host, CVE id,
...) is hashed, and one bucket in ten feeds the eval split until that
domain's eval budget is met, so calibration and eval never share a group.
Calibration records are grouped into sessions of up to ``--session-tokens``;
each eval record is its own session (one logits stream each).

Run on the pod (a CPU pod is enough), not the Mac::

    python scripts/runpod/v41_reap/build_pack.py --model-dir /workspace/v41-hf \\
        --out /workspace/pack

Output: ``v41-pack.jsonl`` (collect.py's input) and ``manifest.json``
(resolved dataset revisions, rows kept/dropped, tokens per domain and split).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import os
import sys
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

K = 1000

# fmt: off
#: name, dataset, load kwargs, adapter, domain (str, or a callable on the row), group-key field, {domain: (calib, eval) tokens}
SOURCES: list[dict[str, Any]] = [
    dict(name="open_swe_traces_dsv4", dataset="nvidia/Open-SWE-Traces",
         load=dict(data_files="data/openhands/deepseek_v4_flash_0731/*/*.parquet", split="train"),
         adapter="openai", domain=lambda r: f"agentic-{r['language']}", group="repo",
         budget={"agentic-python": (400 * K, 50 * K), "agentic-typescript": (300 * K, 40 * K), "agentic-rust": (300 * K, 40 * K)}),
    dict(name="swe_rebench_v2_prs", dataset="nebius/SWE-rebench-V2-PRs", load=dict(split="train"),
         adapter="issue_patch", domain="code-diffs", group="repo", budget={"code-diffs": (400 * K, 50 * K)}),
    # ponytail: no plain-file source. stackv2_edu_filtered's shards are grouped by language with no index,
    # so finding .py/.rs/.ts means scanning most of it; the agent traces already show files through tool output.
    dict(name="nemotron_terminal", dataset="nvidia/Nemotron-Terminal-Corpus", load=dict(name="skill_based_mixed", split="train"),
         adapter="terminus", domain="bash", group="task", budget={"bash": (400 * K, 50 * K)}),
    dict(name="toucan", dataset="Agent-Ark/Toucan-1.5M", load=dict(name="SFT", split="train"),
         adapter="toucan", domain="tool-use", group="tools", budget={"tool-use": (450 * K, 60 * K)}),
    dict(name="circl_vulns", dataset="CIRCL/vulnerability-scores", load=dict(split="train"),
         adapter="advisory", domain="cyber", group="id", pack=True, budget={"cyber": (150 * K, 25 * K)}),
    dict(name="stackexchange", dataset="common-pile/stackexchange_filtered", load=dict(split="train"),
         adapter="stackexchange", domain=lambda r: _se_domain(r), group="url", pack=True,
         budget={"cyber-qa": (150 * K, 25 * K), "networking-qa": (100 * K, 20 * K)}),
    dict(name="rfcs", dataset="rfc-editor.org", load=dict(first=9000, last=9800),
         adapter="rfc", domain="networking", group="rfc", budget={"networking": (150 * K, 25 * K)}),
    dict(name="finepdfs_business", dataset="HuggingFaceFW/finepdfs", load=dict(name="eng_Latn", split="train"),
         adapter="business_pdf", domain="business", group="url", group_host=True, budget={"business": (210 * K, 30 * K)}),
    dict(name="enron", dataset="corbt/enron-emails", load=dict(split="train"),
         adapter="email", domain="business-email", group="from", pack=True, budget={"business-email": (40 * K, 10 * K)}),
    dict(name="wildchat_prompts", dataset="allenai/WildChat-4.8M", load=dict(split="train"),
         adapter="wildchat", domain="english-chat", group="conversation_hash", budget={"english-chat": (50 * K, 15 * K)}),
    dict(name="finewiki", dataset="HuggingFaceFW/finewiki", load=dict(name="en", split="train"),
         adapter="text", domain="english-prose", group="id", budget={"english-prose": (100 * K, 15 * K)}),
]
# fmt: on

SE_SITES = {
    "cyber-qa": ("security.stackexchange.com", "reverseengineering.stackexchange.com"),
    "networking-qa": ("networkengineering.stackexchange.com", "serverfault.com", "unix.stackexchange.com"),
}
BUSINESS_WORDS = re.compile(
    r"\b(invoice|purchase order|statement of work|quarterly report|annual report|board minutes|meeting minutes|"
    r"business plan|proposal|policy|procedure|memorandum|agreement|contract|budget|financial statements?)\b",
    re.IGNORECASE,
)
NON_ENGLISH_SCRIPT = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿ｦ-ﾟ]")


def is_english(text: str) -> bool:
    """No CJK / Hangul / Kana at all, and at most 2% of letters outside Latin."""

    if NON_ENGLISH_SCRIPT.search(text):
        return False
    letters = [c for c in text if c.isalpha()]
    foreign = sum(1 for c in letters if ord(c) > 0x024F)
    return not letters or foreign <= 0.02 * len(letters)


def _se_domain(row: dict[str, Any]) -> str | None:
    site = (row.get("metadata") or {}).get("site")
    return next((domain for domain, sites in SE_SITES.items() if site in sites), None)


# ---------------------------------------------------------------------------
# Adapters: row -> ("chat", messages) | ("doc", text) | None
# ---------------------------------------------------------------------------


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _split_think(content: str) -> tuple[str, str]:
    match = re.match(r"\s*<think>(.*?)</think>\s*(.*)", content or "", re.DOTALL)
    return (match.group(1).strip(), match.group(2)) if match else ("", content or "")


def adapt_openai(row: dict[str, Any]) -> tuple[str, Any] | None:
    messages = []
    for m in _json(row["messages"]):
        msg = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("reasoning_content"):
            msg["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            msg["tool_calls"] = _json(m["tool_calls"])
        if m["role"] == "tool" and m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        messages.append(msg)
    tools = [_json(t) for t in _json(row.get("tools") or [])]
    if tools and messages and messages[0]["role"] == "system":
        messages[0]["tools"] = tools
    elif tools:
        messages.insert(0, {"role": "system", "content": "", "tools": tools})
    return "chat", messages


def adapt_terminus(row: dict[str, Any]) -> tuple[str, Any] | None:
    messages = []
    for m in row["conversations"]:
        if m["role"] == "assistant":
            reasoning, content = _split_think(m["content"])
            messages.append({"role": "assistant", "content": content, **({"reasoning_content": reasoning} if reasoning else {})})
        else:
            messages.append({"role": m["role"], "content": m["content"]})
    return "chat", messages


def adapt_toucan(row: dict[str, Any]) -> tuple[str, Any] | None:
    if "multi-turn" not in (row.get("subset_name") or ""):
        return None
    messages: list[dict[str, Any]] = [{"role": "system", "content": "", "tools": _json(row["tools"])}]
    calls = 0
    for m in _json(row["messages"]):
        role, content = m["role"], m.get("content") or ""
        if role == "tool_call":
            call = ast.literal_eval(content) if isinstance(content, str) else content
            calls += 1
            if messages[-1]["role"] != "assistant":
                messages.append({"role": "assistant", "content": ""})
            messages[-1].setdefault("tool_calls", []).append(
                {"id": f"call_{calls}", "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
            )
        elif role == "tool_response":
            messages.append({"role": "tool", "content": content, "tool_call_id": f"call_{calls}"})
        else:
            messages.append({"role": role, "content": content})
    return "chat", messages


def adapt_issue_patch(row: dict[str, Any]) -> tuple[str, Any] | None:
    issue = row.get("problem_statement") or row.get("pr_description") or ""
    if not issue or not row.get("patch"):
        return None
    return "chat", [
        {"role": "user", "content": f"Repository: {row['repo']}\n\n{issue}\n\nWrite the patch."},
        {"role": "assistant", "content": row["patch"]},
    ]


def adapt_advisory(row: dict[str, Any]) -> tuple[str, Any] | None:
    if not row.get("description"):
        return None
    head = row["id"] + (f": {row['title']}" if row.get("title") else "")
    return "doc", f"{head}\n{row['description']}"


def adapt_stackexchange(row: dict[str, Any]) -> tuple[str, Any] | None:
    return "doc", row["text"]


def adapt_business_pdf(row: dict[str, Any]) -> tuple[str, Any] | None:
    ocr = row.get("ocr_quality_scores") or [0]
    if min(ocr) < 2.0 or not BUSINESS_WORDS.search(row["text"][:4000]):
        return None
    return "doc", row["text"]


def adapt_email(row: dict[str, Any]) -> tuple[str, Any] | None:
    if len(row.get("body") or "") < 200:
        return None
    return "doc", f"From: {row['from']}\nTo: {', '.join(row.get('to') or [])}\nSubject: {row.get('subject', '')}\n\n{row['body']}"


def adapt_wildchat(row: dict[str, Any]) -> tuple[str, Any] | None:
    # Human prompts only: the responses are OpenAI-model output, so they stay out.
    if row.get("language") != "English" or row.get("toxic") or row.get("redacted"):
        return None
    first = next((m["content"] for m in row["conversation"] if m["role"] == "user"), "")
    return ("chat", [{"role": "user", "content": first}]) if len(first) > 40 else None


def adapt_text(row: dict[str, Any]) -> tuple[str, Any] | None:
    return "doc", row["text"]


ADAPTERS: dict[str, Callable[[dict[str, Any]], tuple[str, Any] | None]] = {
    "openai": adapt_openai, "terminus": adapt_terminus, "toucan": adapt_toucan, "issue_patch": adapt_issue_patch,
    "advisory": adapt_advisory, "stackexchange": adapt_stackexchange,
    "business_pdf": adapt_business_pdf, "email": adapt_email, "wildchat": adapt_wildchat, "text": adapt_text,
}  # fmt: skip


def rfc_rows(first: int, last: int) -> Iterator[dict[str, Any]]:
    for number in range(first, last + 1):
        try:
            with urllib.request.urlopen(f"https://www.rfc-editor.org/rfc/rfc{number}.txt", timeout=30) as response:
                yield {"rfc": number, "text": response.read().decode("utf-8", "replace")}
        except OSError:
            continue  # unassigned or not-yet-published numbers


# ---------------------------------------------------------------------------
# Rendering, splitting, packing
# ---------------------------------------------------------------------------


class Renderer:
    def __init__(self, model_dir: Path):
        from tokenizers import Tokenizer

        sys.path.insert(0, str(model_dir / "encoding"))
        import encoding  # V4.1's own chat format (the pinned revision's encoding/encoding.py)

        self.encoding = encoding
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.bos = self.tokenizer.token_to_id(encoding.bos_token)
        self.eos = self.tokenizer.token_to_id(encoding.eos_token)

    def __call__(self, kind: str, payload: Any) -> list[int]:
        if kind == "chat":
            thinking = any(m.get("reasoning_content") for m in payload)
            text = self.encoding.encode_messages(payload, thinking_mode="thinking" if thinking else "chat")
            return self.tokenizer.encode(text, add_special_tokens=False).ids
        return [self.bos, *self.tokenizer.encode(payload, add_special_tokens=False).ids]


def is_eval_group(key: str) -> bool:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10 == 0


def iter_rows(source: dict[str, Any], revision: str | None) -> Iterator[dict[str, Any]]:
    if source["adapter"] == "rfc":
        yield from rfc_rows(**source["load"])
        return
    from datasets import load_dataset

    load = dict(source["load"])
    yield from load_dataset(source["dataset"], streaming=True, revision=revision, **load)


def build(sources: list[dict[str, Any]], render: Callable[[str, Any], list[int]], *, eos: int, max_tokens: int,
          pack_tokens: int, session_tokens: int, revisions: dict[str, str | None], rows: Callable = iter_rows) -> tuple[list[dict], dict]:  # fmt: skip
    records: list[dict[str, Any]] = []
    report: dict[str, Any] = {"sources": {}, "tokens": defaultdict(lambda: Counter())}
    for source in sources:
        adapter = ADAPTERS.get(source["adapter"])
        need = {d: {"calibration": c, "eval": e} for d, (c, e) in source["budget"].items()}
        got = report["tokens"]
        stats = Counter()
        buffers: dict[tuple[str, str], list[int]] = {}
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)

        def emit(domain: str, split: str, ids: list[int], keys: list[str]) -> None:
            records.append({"id": f"{source['name']}-{len(records)}", "split": split, "domain": domain,
                            "source": source["name"], "groups": keys, "token_ids": ids[:max_tokens]})  # fmt: skip
            got[domain][split] += min(len(ids), max_tokens)

        for row in rows(source, revisions.get(source["name"])):
            if all(got[d][s] >= n for d, v in need.items() for s, n in v.items()):
                break
            stats["rows"] += 1
            domain = source["domain"](row) if callable(source["domain"]) else source["domain"]
            if domain not in need:
                stats["off_domain"] += 1
                continue
            converted = adapt_text({"text": row["text"]}) if source["adapter"] == "rfc" else adapter(row)
            if converted is None:
                stats["filtered"] += 1
                continue
            if not is_english(json.dumps(converted[1], ensure_ascii=False)):
                stats["non_english"] += 1
                continue
            key = str(row.get(source["group"]))
            if source.get("group_host"):  # hold out whole sites, not single URLs
                key = urlparse(key).netloc or key
            split = "eval" if is_eval_group(key) else "calibration"
            if got[domain][split] >= need[domain][split]:
                stats["over_budget"] += 1
                continue
            ids = render(*converted)
            stats["kept"] += 1
            if not source.get("pack"):
                emit(domain, split, ids, [key])
                continue
            buf = buffers.setdefault((domain, split), [])
            buf.extend(ids + [eos])
            groups[(domain, split)].append(key)
            if len(buf) >= pack_tokens:
                emit(domain, split, buf, groups.pop((domain, split)))
                buffers[(domain, split)] = []
        for (domain, split), buf in buffers.items():
            if buf:
                emit(domain, split, buf, groups[(domain, split)])
        report["sources"][source["name"]] = {"dataset": source["dataset"], "revision": revisions.get(source["name"]), **stats}

    # Sessions: each eval record alone; calibration interleaved across domains, cut at session_tokens.
    calibration = sorted((r for r in records if r["split"] == "calibration"), key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    session, size = 0, 0
    for record in calibration:
        if size and size + len(record["token_ids"]) > session_tokens:
            session, size = session + 1, 0
        record["session"] = f"cal-{session:04d}"
        record["target_spans"] = []
        size += len(record["token_ids"])
    for i, record in enumerate(r for r in records if r["split"] == "eval"):
        record["session"] = f"eval-{i:05d}"
        record["target_spans"] = [[1, len(record["token_ids"])]]
    report["tokens"] = {d: dict(c) for d, c in sorted(report["tokens"].items())}
    return calibration + [r for r in records if r["split"] == "eval"], report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, type=Path, help="V4.1 HF snapshot (tokenizer.json, encoding/)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--pack-tokens", type=int, default=6144)
    parser.add_argument("--session-tokens", type=int, default=131072)
    parser.add_argument("--only", help="comma-separated source names (debugging)")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi

    sources = [s for s in SOURCES if not args.only or s["name"] in args.only.split(",")]
    api = HfApi()
    revisions = {s["name"]: (api.dataset_info(s["dataset"]).sha if "/" in s["dataset"] else None) for s in sources}
    render = Renderer(args.model_dir)
    records, report = build(sources, render, eos=render.eos, max_tokens=args.max_tokens, pack_tokens=args.pack_tokens,
                            session_tokens=args.session_tokens, revisions=revisions)  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    pack = args.out / "v41-pack.jsonl"
    with pack.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    report.update(
        record_type="keep_v41_reap_pack_v1",
        pack_sha256=hashlib.sha256(pack.read_bytes()).hexdigest(),
        tokenizer_sha256=hashlib.sha256((args.model_dir / "tokenizer.json").read_bytes()).hexdigest(),
        encoding_sha256=hashlib.sha256((args.model_dir / "encoding" / "encoding.py").read_bytes()).hexdigest(),
        records=len(records),
        sessions=len({r["session"] for r in records}),
        settings={"max_tokens": args.max_tokens, "pack_tokens": args.pack_tokens, "session_tokens": args.session_tokens},
    )
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("records", "sessions", "tokens")}, indent=2))
    return 0


if __name__ == "__main__":
    status = main()
    sys.stdout.flush()
    # Streaming `datasets` leaves native worker threads behind, and interpreter
    # finalization then aborts with PyGILState_Release (exit 134). That happened
    # on 2026-11-06 after the pack was fully written, and it stopped the pod.
    # Skip finalization: every output is already closed.
    os._exit(status)
