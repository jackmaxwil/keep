"""Deterministic terminal and self-contained HTML reports for ``keep build``.

The report is intentionally a read-only view of an append-only build ledger
and its evidence files.  It does not reinterpret a recorded gate rejection as
an execution error: ``gate_failed`` is rendered as ``VALID BLOCKED``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


_TERMINAL_EVENTS = {"completed", "gate_failed", "failed"}


@dataclass
class _Step:
    step_id: str
    step_key: str
    op: str = "-"
    started_at: str | None = None
    terminal: dict[str, Any] | None = None
    evidence_path: Path | None = None
    evidence: dict[str, Any] | None = None
    evidence_error: str | None = None

    @property
    def event(self) -> str | None:
        return self.terminal.get("event") if self.terminal else None

    @property
    def status(self) -> str:
        if self.event == "gate_failed":
            return "VALID BLOCKED"
        if self.event == "completed":
            return "COMPLETED"
        if self.event == "failed":
            return "FAILED"
        if self.started_at is not None:
            return "RUNNING"
        return "UNKNOWN"

    @property
    def elapsed(self) -> str:
        if self.started_at is None or self.terminal is None:
            return "-"
        finished_at = _string_or_none(self.terminal.get("ts"))
        if finished_at is None:
            return "-"
        try:
            start = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            finish = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        except ValueError:
            return "-"
        return f"{(finish - start).total_seconds():.3f}s"

    @property
    def evidence_state(self) -> str:
        if self.evidence is not None:
            return "present"
        if self.evidence_error is not None:
            return "unreadable"
        return "missing"


@dataclass
class _Report:
    source: Path
    is_standalone: bool
    steps: list[_Step] = field(default_factory=list)

    @property
    def identity(self) -> str:
        return self.source.name

    @property
    def source_label(self) -> str:
        return "standalone evidence" if self.is_standalone else "run directory"


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _read_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, str(error)
    if not isinstance(value, dict):
        return None, "expected JSON object"
    return value, None


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _evidence_candidates(run: Path, step: _Step, records: Iterable[dict[str, Any]]) -> list[Path]:
    candidates: list[Path] = []
    for record in records:
        paths = record.get("evidence_paths")
        if not isinstance(paths, list):
            continue
        for raw_path in paths:
            if isinstance(raw_path, str):
                candidate = Path(raw_path)
                if candidate.exists():
                    candidates.append(candidate)
    for record in records:
        output_dir = record.get("output_dir")
        if isinstance(output_dir, str):
            candidate = Path(output_dir).parent / "evidence" / "evidence_json.json"
            if candidate.exists():
                candidates.append(candidate)
    prefix = f"{step.step_id}-{step.step_key[:8]}"
    steps_dir = run / "steps"
    if steps_dir.exists():
        candidates.extend(sorted(steps_dir.glob(f"{prefix}*/evidence/evidence_json.json")))
    return sorted({candidate.resolve() for candidate in candidates}, key=str)


def _build_run_report(run: Path) -> _Report:
    rows = _records(run / "ledger.jsonl")
    steps_by_key: dict[str, _Step] = {}
    records_by_key: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        step_key = _string_or_none(row.get("step_key"))
        step_id = _string_or_none(row.get("step_id"))
        if step_key is None or step_id is None:
            continue
        step = steps_by_key.setdefault(step_key, _Step(step_id=step_id, step_key=step_key))
        records_by_key.setdefault(step_key, []).append(row)
        if isinstance(row.get("op"), str):
            step.op = row["op"]
        if row.get("event") == "started" and step.started_at is None:
            step.started_at = _string_or_none(row.get("ts"))
        if row.get("event") in _TERMINAL_EVENTS:
            step.terminal = row

    for step_key, step in steps_by_key.items():
        candidates = _evidence_candidates(run, step, records_by_key[step_key])
        if candidates:
            step.evidence_path = candidates[0]
            step.evidence, step.evidence_error = _read_object(candidates[0])
    return _Report(source=run, is_standalone=False, steps=list(steps_by_key.values()))


def _build_evidence_report(path: Path) -> _Report:
    evidence, error = _read_object(path)
    status = "completed"
    if evidence is not None and _evidence_is_blocked(evidence):
        status = "gate_failed"
    step = _Step(
        step_id="evidence",
        step_key=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "-",
        op=(evidence or {}).get("record_type", "evidence") if evidence else "evidence",
        terminal={"event": status},
        evidence_path=path,
        evidence=evidence,
        evidence_error=error,
    )
    return _Report(source=path, is_standalone=True, steps=[step])


def _load_report(source: str | Path) -> _Report:
    path = Path(source)
    if path.is_dir():
        return _build_run_report(path)
    return _build_evidence_report(path)


def _leaf_items(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _leaf_items(value[key], prefix + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaf_items(item, prefix + (str(index),))
    else:
        yield prefix, value


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    if value is None:
        return "null"
    return str(value)


def _evidence_is_blocked(evidence: Mapping[str, Any]) -> bool:
    gate_status = evidence.get("gate_status")
    if isinstance(gate_status, str) and "blocked" in gate_status.lower():
        return True
    for key, value in evidence.items():
        if key.endswith("_gate_pass") and value is False:
            return True
    return False


def _gate_rows(report: _Report) -> list[tuple[str, str, str, list[str]]]:
    rows: list[tuple[str, str, str, list[str]]] = []
    for step in report.steps:
        terminal_gate = (step.terminal or {}).get("gate_result")
        if isinstance(terminal_gate, Mapping):
            profile = _format_value(terminal_gate.get("profile", "-"))
            blockers = terminal_gate.get("reasons")
            blocker_list = [str(item) for item in blockers] if isinstance(blockers, list) else []
            status = "PASSED" if terminal_gate.get("passed") is True else "VALID BLOCKED"
            rows.append((step.step_id, profile, status, blocker_list))
            continue
        if step.evidence is not None and _evidence_is_blocked(step.evidence):
            profile = _format_value(step.evidence.get("profile", step.evidence.get("record_type", "-")))
            missing = step.evidence.get("missing_requirements")
            blockers = [str(item) for item in missing] if isinstance(missing, list) else []
            rows.append((step.step_id, profile, "VALID BLOCKED", blockers))
    return rows


def _evidence_values(report: _Report) -> Iterable[tuple[tuple[str, ...], Any]]:
    for step in report.steps:
        if step.evidence is not None:
            yield from _leaf_items(step.evidence)


def _artifact_rows(report: _Report) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path, value in _evidence_values(report):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not path:
            continue
        name = ".".join(path)
        lowered = name.lower()
        if "bytes" in lowered:
            rows.append((name, f"{_format_value(value)} bytes"))
        elif "bpw" in lowered:
            rows.append((name, f"{_format_value(value)} bpw"))
    return sorted(set(rows))


def _memory_rows(report: _Report) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    markers = ("memory", "clean", "pageout", "swapout", "wired")
    for path, value in _evidence_values(report):
        name = ".".join(path)
        if any(marker in name.lower() for marker in markers):
            rows.append((name, _format_value(value)))
    return sorted(set(rows))


def _provenance_rows(report: _Report) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path, value in _evidence_values(report):
        name = ".".join(path)
        if isinstance(value, str) and ("sha256" in name.lower() or name.lower().endswith("hash")):
            rows.append((name, value))
    return sorted(set(rows))


def _terminal_table(rows: list[tuple[str, ...]]) -> list[str]:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return [" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip() for row in rows]


def render_terminal(source: str | Path) -> str:
    """Render a deterministic terminal report for a run directory or evidence JSON."""

    report = _load_report(source)
    lines = ["KEEP build report", f"run: {report.identity}", f"source: {report.source_label}", "", "steps:"]
    table = [("step", "op", "key", "status", "elapsed", "evidence")]
    table.extend(
        (step.step_id, step.op, step.step_key, step.status, step.elapsed, step.evidence_state)
        for step in report.steps
    )
    lines.extend(_terminal_table(table))

    lines.extend(["", "gates:"])
    gates = _gate_rows(report)
    if not gates:
        lines.append("none recorded")
    for step_id, profile, status, blockers in gates:
        lines.append(f"{step_id} [{profile}]: {status}")
        if blockers:
            lines.append(f"  blockers: {', '.join(blockers)}")

    for title, rows in (
        ("artifact accounting", _artifact_rows(report)),
        ("memory cleanliness", _memory_rows(report)),
        ("provenance hashes", _provenance_rows(report)),
    ):
        lines.extend(["", f"{title}:"])
        if rows:
            lines.extend(f"{name}: {value}" for name, value in rows)
        else:
            lines.append("not recorded")
    return "\n".join(lines) + "\n"


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _html_key_values(rows: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p>not recorded</p>"
    return _html_table(["field", "value"], [[name, value] for name, value in rows])


def _evidence_path_label(report: _Report, path: Path) -> str:
    try:
        return str(path.relative_to(report.source))
    except ValueError:
        return str(path)


def render_html(source: str | Path) -> str:
    """Render a deterministic, self-contained HTML report for a build input."""

    report = _load_report(source)
    steps = [
        [step.step_id, step.op, step.step_key, step.status, step.elapsed, step.evidence_state]
        for step in report.steps
    ]
    timeline = [
        [
            step.step_id,
            step.started_at or "-",
            _string_or_none((step.terminal or {}).get("ts")) or "-",
            step.elapsed,
            step.status,
        ]
        for step in report.steps
    ]
    gates = _gate_rows(report)
    gate_rows = [[step, profile, status, ", ".join(blockers) or "-"] for step, profile, status, blockers in gates]
    evidence_hashes = []
    raw_sections = []
    for step in report.steps:
        if step.evidence_path is None:
            continue
        path_label = _evidence_path_label(report, step.evidence_path)
        digest = hashlib.sha256(step.evidence_path.read_bytes()).hexdigest()
        evidence_hashes.append([step.step_id, path_label, digest])
        raw = json.dumps(step.evidence, indent=2, sort_keys=True) if step.evidence is not None else step.evidence_error or "missing"
        raw_sections.append(
            "<details><summary>raw JSON: "
            + html.escape(step.step_id)
            + "</summary><pre>"
            + html.escape(raw)
            + "</pre></details>"
        )
    style = "body{font-family:ui-monospace,monospace;margin:2rem;color:#17202a;background:#fff}h1{margin-bottom:.2rem}h2{margin-top:2rem}table{border-collapse:collapse;width:100%;margin:.6rem 0}th,td{border:1px solid #ccd1d1;padding:.45rem;text-align:left;vertical-align:top}th{background:#ebedef}details{margin:.6rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8f9f9;padding:1rem}.blocked{color:#922b21;font-weight:700}"
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>KEEP build report"
        + "</title><style>"
        + style
        + "</style></head><body><h1>KEEP build report</h1><p>run: "
        + html.escape(report.identity)
        + "<br>source: "
        + html.escape(report.source_label)
        + "</p><h2>steps</h2>"
        + _html_table(["step", "op", "key", "status", "elapsed", "evidence"], steps)
        + "<h2>step timeline</h2>"
        + _html_table(["step", "started", "finished", "elapsed", "status"], timeline)
        + "<h2>gates</h2>"
        + (_html_table(["step", "profile", "outcome", "blockers"], gate_rows) if gate_rows else "<p>none recorded</p>")
        + "<h2>artifact accounting</h2>"
        + _html_key_values(_artifact_rows(report))
        + "<h2>memory cleanliness</h2>"
        + _html_key_values(_memory_rows(report))
        + "<h2>provenance hashes</h2>"
        + _html_key_values(_provenance_rows(report))
        + "<h2>evidence hashes</h2>"
        + (_html_table(["step", "path", "sha256"], evidence_hashes) if evidence_hashes else "<p>missing</p>")
        + "<h2>raw JSON</h2>"
        + ("".join(raw_sections) if raw_sections else "<p>missing</p>")
        + "</body></html>"
    )


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Configure a ``keep report`` parser without registering the command."""

    parser.add_argument("source", type=Path, help="build run directory or evidence JSON file")
    parser.add_argument("--format", choices=("terminal", "html"), default="terminal")
    parser.add_argument("-o", "--output", type=Path, help="write report to this path")
    return parser


def run(args: argparse.Namespace) -> int:
    """Run the CLI-ready renderer configured by :func:`configure_parser`."""

    rendered = render_html(args.source) if args.format == "html" else render_terminal(args.source)
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0
