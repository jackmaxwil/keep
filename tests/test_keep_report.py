from __future__ import annotations

import json
from pathlib import Path

from mlx_vq.build.report import render_html, render_terminal
from mlx_vq.build.cli import main as keep_main


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tiny_run(tmp_path: Path) -> Path:
    run = tmp_path / "tiny-run"
    prepare_dir = run / "steps" / "prepare-abcd1234"
    gate_dir = run / "steps" / "quality-efgh5678"
    _write_json(
        prepare_dir / "evidence" / "evidence_json.json",
        {
            "artifact_bytes": 128,
            "artifact_bpw": 1.25,
            "memory_clean": True,
            "recipe_sha256": "a" * 64,
        },
    )
    _write_json(
        gate_dir / "evidence" / "evidence_json.json",
        {
            "gate_status": "quality_gate_blocked",
            "missing_requirements": ["holdout_eval"],
            "source_sha256": "b" * 64,
        },
    )
    records = [
        {
            "event": "started",
            "op": "make-artifact",
            "output_dir": str(prepare_dir / "out"),
            "step_id": "prepare",
            "step_key": "abcd1234full",
            "ts": "2026-07-10T12:00:00+00:00",
        },
        {
            "event": "completed",
            "step_id": "prepare",
            "step_key": "abcd1234full",
            "ts": "2026-07-10T12:00:02+00:00",
        },
        {
            "event": "started",
            "op": "quality-gate",
            "output_dir": str(gate_dir / "out"),
            "step_id": "quality",
            "step_key": "efgh5678full",
            "ts": "2026-07-10T12:01:00+00:00",
        },
        {
            "event": "gate_failed",
            "gate_result": {
                "passed": False,
                "profile": "quality",
                "reasons": ["quality_floor", "holdout_eval"],
            },
            "step_id": "quality",
            "step_key": "efgh5678full",
            "ts": "2026-07-10T12:01:03+00:00",
        },
        {
            "event": "started",
            "op": "unknown-op",
            "output_dir": str(run / "steps" / "missing-ijkl9012" / "out"),
            "step_id": "missing",
            "step_key": "ijkl9012full",
            "ts": "2026-07-10T12:02:00+00:00",
        },
    ]
    run.mkdir(parents=True, exist_ok=True)
    (run / "ledger.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return run


def test_terminal_render_matches_golden_blocked_run(tmp_path: Path) -> None:
    run = _tiny_run(tmp_path)

    assert render_terminal(run) == """KEEP build report
run: tiny-run
source: run directory

steps:
step    | op            | key          | status        | elapsed | evidence
prepare | make-artifact | abcd1234full | COMPLETED     | 2.000s  | present
quality | quality-gate  | efgh5678full | VALID BLOCKED | 3.000s  | present
missing | unknown-op    | ijkl9012full | RUNNING       | -       | missing

gates:
quality [quality]: VALID BLOCKED
  blockers: quality_floor, holdout_eval

artifact accounting:
artifact_bpw: 1.25 bpw
artifact_bytes: 128 bytes

memory cleanliness:
memory_clean: true

provenance hashes:
recipe_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
source_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
"""


def test_missing_evidence_is_explicit_and_unknown_ops_are_preserved(tmp_path: Path) -> None:
    terminal = render_terminal(_tiny_run(tmp_path))

    assert "missing | unknown-op    | ijkl9012full | RUNNING       | -       | missing" in terminal
    assert "unknown-op" in terminal


def test_html_is_self_contained_and_exposes_timeline_hashes_and_raw_json(tmp_path: Path) -> None:
    html = render_html(_tiny_run(tmp_path))

    assert "<style>" in html
    assert "step timeline" in html
    assert "evidence hashes" in html
    assert "raw JSON" in html
    assert "<details>" in html
    assert "VALID BLOCKED" in html
    assert "http://" not in html
    assert "https://" not in html


def test_rendering_is_deterministic_for_run_and_standalone_evidence(tmp_path: Path) -> None:
    run = _tiny_run(tmp_path)
    evidence = run / "steps" / "quality-efgh5678" / "evidence" / "evidence_json.json"

    assert render_terminal(run) == render_terminal(run)
    assert render_html(run) == render_html(run)
    standalone = render_terminal(evidence)
    standalone_html = render_html(evidence)
    assert standalone == render_terminal(evidence)
    assert standalone_html == render_html(evidence)
    assert "source: standalone evidence" in standalone
    assert "VALID BLOCKED" in standalone
    assert "holdout_eval" in standalone
    assert "evidence hashes" in standalone_html


def test_cli_report_writes_self_contained_html(tmp_path: Path) -> None:
    output = tmp_path / "report.html"

    assert keep_main(["report", str(_tiny_run(tmp_path)), "--html", str(output)]) == 0

    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")
