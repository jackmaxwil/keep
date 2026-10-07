"""Evidence-bound Markdown model-card rendering for KEEP artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_profile(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "not supplied"
    return str(value)


def _evidence_rows(evidence_paths: Sequence[Path], keys: tuple[str, ...]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path in evidence_paths:
        evidence = _load_object(path)
        for key in keys:
            if key in evidence:
                rows.append((key, _value(evidence[key])))
    return rows


def _table(rows: Sequence[tuple[str, str]], *, empty_label: str) -> str:
    if not rows:
        rows = [(empty_label, "not yet measured")]
    return "| Measure | Evidence |\n| --- | --- |\n" + "\n".join(
        f"| {name} | {value} |" for name, value in rows
    )


def render(
    profile_path: str | Path,
    audit_path: str | Path,
    family_gate_path: str | Path,
    *,
    evaluation_paths: Sequence[str | Path] = (),
    benchmark_paths: Sequence[str | Path] = (),
    release: bool = False,
) -> str:
    """Render a model card using only the supplied profile and evidence files."""

    profile_file = Path(profile_path)
    audit_file = Path(audit_path)
    gate_file = Path(family_gate_path)
    profile = _load_profile(profile_file)
    audit = _load_object(audit_file)
    gate = _load_object(gate_file)
    evaluations = [Path(path) for path in evaluation_paths]
    benchmarks = [Path(path) for path in benchmark_paths]
    family_gate_pass = gate.get("family_gate_pass") is True

    title = "GLM-5.2-REAP-KEEP-504B"
    lines: list[str] = []
    if release and not family_gate_pass:
        lines.extend(["# PREVIEW — NOT A RELEASE", "", "Requested release rendering is blocked: `family_gate_pass=false`.", ""])
    lines.extend([f"# {title}", "", "## Identity", ""])
    lines.extend(
        [
            f"- Source: `{_value(profile.get('hf_model_id'))} @ {_value(profile.get('revision'))}`",
            f"- Profile: `{_value(profile.get('name'))}`",
            f"- Family gate: `family_gate_pass={_value(gate.get('family_gate_pass'))}`",
            "",
            "## Artifact accounting",
            "",
        ]
    )
    accounting_keys = (
        "actual_whole_model_tensor_payload_bytes",
        "actual_whole_model_tensor_payload_bpw",
        "actual_routed_payload_bytes",
        "actual_routed_bpw",
        "whole_model_parameter_count",
    )
    accounting = [(key, _value(audit[key])) for key in accounting_keys if key in audit]
    lines.append(_table(accounting, empty_label="Audited artifact accounting"))
    lines.extend(
        [
            "",
            "## Method summary",
            "",
            "KEEP/RAMP artifact workflow.",
            "",
            "## Quality",
            "",
            _table(_evidence_rows(evaluations, ("quality_score", "perplexity", "score", "pass")), empty_label="Quality"),
            "",
            "## Speed and memory",
            "",
            _table(
                _evidence_rows(benchmarks, ("warm_tokens_per_second", "tokens_per_second", "peak_memory_bytes", "cold_residency_memory_clean")),
                empty_label="Speed",
            ),
        ]
    )
    cold_clean = next((value for name, value in _evidence_rows(benchmarks, ("cold_residency_memory_clean",)) if name == "cold_residency_memory_clean"), None)
    lines.extend([f"\nCold residency: {'clean' if cold_clean == 'true' else 'not clean' if cold_clean == 'false' else 'not yet measured'}.", "", "## Claim boundaries", ""])
    lines.extend(
        [
            "- Reference is the dequantized source artifact; this is not a BF16-teacher claim.",
            "- `activation_quantization_emulated=false`.",
            "- No BF16-teacher claim.",
            "- No W4A4 parity claim.",
            "",
            "## Reproduction",
            "",
            f"```bash\nkeep get {_value(profile.get('name'))}\nkeep build <recipe.yaml>\n```",
            "",
            "## Source license and attribution",
            "",
            f"- Attribution: `{_value(profile.get('hf_model_id'))}`.",
            f"- Source license: {_value(profile.get('license'))} (only profile-supplied metadata is reported).",
            "",
            "## Limitations",
            "",
            "- Quality, speed, and memory claims are limited to the supplied evidence files.",
            "- Cold and warm residency are reported separately; missing evidence is not inferred.",
        ]
    )
    return "\n".join(lines) + "\n"


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("profile", type=Path, help="model profile YAML")
    parser.add_argument("artifact_audit", type=Path, help="composite artifact audit JSON")
    parser.add_argument("family_gate", type=Path, help="family-gate evidence JSON")
    parser.add_argument("--eval", dest="evaluations", type=Path, action="append", default=[])
    parser.add_argument("--benchmark", dest="benchmarks", type=Path, action="append", default=[])
    parser.add_argument("-o", "--out", type=Path, help="write Markdown to this path")
    parser.add_argument("--release", action="store_true", help="request release rendering")
    return parser


def run(args: argparse.Namespace) -> int:
    rendered = render(args.profile, args.artifact_audit, args.family_gate, evaluation_paths=args.evaluations, benchmark_paths=args.benchmarks, release=args.release)
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.write_text(rendered, encoding="utf-8")
    return 0
