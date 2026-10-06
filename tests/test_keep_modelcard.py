from __future__ import annotations

import json
from pathlib import Path

from mlx_vq.build import modelcard
from mlx_vq.build.cli import main as keep_main


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def _inputs(tmp_path: Path, *, gate_pass: bool = True) -> tuple[Path, Path, Path]:
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "name: glm52-reap-504b-v2\n"
        "hf_model_id: 0xSero/glm-5.2-reap-504B-v2\n"
        "revision: pinned-revision\n",
        encoding="utf-8",
    )
    audit = _write_json(
        tmp_path / "audit.json",
        {
            "actual_whole_model_tensor_payload_bytes": 123456,
            "actual_whole_model_tensor_payload_bpw": 1.25,
            "whole_model_values_actual": True,
            "actual_routed_payload_bytes": 100000,
        },
    )
    gate = _write_json(
        tmp_path / "gate.json",
        {"family_gate_pass": gate_pass, "activation_quantization_emulated": False},
    )
    return profile, audit, gate


def test_render_model_card_uses_audited_evidence_only(tmp_path: Path) -> None:
    profile, audit, gate = _inputs(tmp_path)
    evaluation = _write_json(tmp_path / "eval.json", {"quality_score": 0.91})
    benchmark = _write_json(
        tmp_path / "benchmark.json",
        {"warm_tokens_per_second": 42.0, "cold_residency_memory_clean": False},
    )

    rendered = modelcard.render(
        profile, audit, gate, evaluation_paths=[evaluation], benchmark_paths=[benchmark], release=True
    )

    assert "# GLM-5.2-REAP-KEEP-504B" in rendered
    assert "0xSero/glm-5.2-reap-504B-v2 @ pinned-revision" in rendered
    assert "123456" in rendered
    assert "1.25" in rendered
    assert "quality_score" in rendered
    assert "warm_tokens_per_second" in rendered
    assert "Cold residency: not clean" in rendered
    assert "activation_quantization_emulated=false" in rendered
    assert "No BF16-teacher claim" in rendered
    assert "No W4A4 parity claim" in rendered


def test_blocked_release_renders_watermarked_preview(tmp_path: Path) -> None:
    profile, audit, gate = _inputs(tmp_path, gate_pass=False)

    rendered = modelcard.render(profile, audit, gate, release=True)

    assert "# PREVIEW — NOT A RELEASE" in rendered
    assert "family_gate_pass=false" in rendered


def test_missing_eval_evidence_is_explicit(tmp_path: Path) -> None:
    profile, audit, gate = _inputs(tmp_path)

    rendered = modelcard.render(profile, audit, gate)

    assert "| Quality | not yet measured |" in rendered
    assert "| Speed | not yet measured |" in rendered


def test_cli_writes_model_card(tmp_path: Path) -> None:
    profile, audit, gate = _inputs(tmp_path)
    output = tmp_path / "README.md"

    assert (
        keep_main(
            [
                "model-card",
                str(profile),
                str(audit),
                str(gate),
                "--out",
                str(output),
                "--release",
            ]
        )
        == 0
    )

    assert output.read_text(encoding="utf-8").startswith("# GLM-5.2-REAP-KEEP-504B")
