from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mlx_vq.quality.selective_precision import (
    build_custom_selective_precision_candidate,
    build_selective_precision_candidates,
    candidate_to_json,
    discover_artifact_group_files,
    materialize_linked_candidate,
    parse_high_bit_policy,
)


def _report(layers: list[int], *, top_metric: str = "source_routed_glu") -> dict:
    return {
        "summary": {
            "layer_rankings": [
                {
                    "layer": layer,
                    "rank": rank,
                    "mean_source_weighted_rel_l2": 1.0 / rank,
                    "dominant_projection_counts": {top_metric: 1},
                }
                for rank, layer in enumerate(layers, start=1)
            ],
            "layer_projection_rankings": [
                {
                    "layer": layers[0],
                    "metric": "source_up_proj",
                    "rank": 1,
                    "mean_rel_l2": 0.9,
                    "max_rel_l2": 0.91,
                },
                {
                    "layer": layers[0],
                    "metric": "source_routed_glu",
                    "rank": 2,
                    "mean_rel_l2": 0.8,
                    "max_rel_l2": 0.81,
                },
            ],
        }
    }


def _write_group_files(root: Path, *, byte: bytes, layers: range) -> None:
    root.mkdir()
    for layer in layers:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            (root / f"layer-{layer:05d}-{projection}.safetensors").write_bytes(byte)


def test_build_selective_precision_candidates_maps_glu_to_gate_up() -> None:
    candidates = build_selective_precision_candidates(
        hard_report=_report([16, 36, 41]),
        tail_report=_report([14, 17, 15, 16]),
    )
    by_name = {candidate.name: candidate for candidate in candidates}

    assert by_name["lane3-hard16-up"].policy_dict == {
        "16:up_proj": 16,
    }
    assert by_name["lane3-hard16-glu"].policy_dict == {
        "16:gate_proj": 16,
        "16:up_proj": 16,
    }
    assert by_name["lane3-hard-top3-glu"].high_bit_keys == (
        "16:gate_proj",
        "16:up_proj",
        "36:gate_proj",
        "36:up_proj",
        "41:gate_proj",
        "41:up_proj",
    )
    assert "14:gate_proj" in by_name["lane3-hard16-longtail-glu"].high_bit_keys
    assert "17:up_proj" in by_name["lane3-combined-top-glu"].high_bit_keys


def test_candidate_storage_and_materialization_use_high_bit_for_policy_groups(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    high_bit = tmp_path / "high-bit"
    _write_group_files(baseline, byte=b"b", layers=range(1, 3))
    _write_group_files(high_bit, byte=b"hh", layers=range(1, 3))
    groups = discover_artifact_group_files(
        baseline_dir=baseline,
        high_bit_dir=high_bit,
    )
    candidate = build_selective_precision_candidates(
        hard_report=_report([1, 2]),
    )[0]

    payload = candidate_to_json(candidate, groups)
    materialized = materialize_linked_candidate(
        candidate=candidate,
        baseline_dir=baseline,
        high_bit_dir=high_bit,
        output_dir=tmp_path / "candidate",
        groups=groups,
    )
    reused = materialize_linked_candidate(
        candidate=candidate,
        baseline_dir=baseline,
        high_bit_dir=high_bit,
        output_dir=tmp_path / "candidate",
        groups=groups,
        allow_existing=True,
    )

    assert payload["storage"]["baseline_logical_bytes"] == 6
    assert payload["storage"]["candidate_logical_bytes"] == 7
    assert payload["storage"]["high_bit_group_count"] == 1
    assert materialized["high_bit_link_count"] == 1
    assert materialized["baseline_link_count"] == 5
    assert materialized["preexisting"] is False
    assert reused["preexisting"] is True
    assert reused["high_bit_link_count"] == 1
    assert os.readlink(tmp_path / "candidate" / "layer-00001-up_proj.safetensors").endswith(
        "high-bit/layer-00001-up_proj.safetensors"
    )
    assert os.readlink(tmp_path / "candidate" / "layer-00001-gate_proj.safetensors").endswith(
        "baseline/layer-00001-gate_proj.safetensors"
    )
    assert os.readlink(tmp_path / "candidate" / "layer-00001-down_proj.safetensors").endswith(
        "baseline/layer-00001-down_proj.safetensors"
    )


def test_materialization_preserves_high_bit_continuous_sidecars(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    high_bit = tmp_path / "high-bit"
    _write_group_files(baseline, byte=b"b", layers=range(1, 2))
    _write_group_files(high_bit, byte=b"hh", layers=range(1, 2))
    sidecar_path = high_bit / "continuous_params" / "layer-00001-up_proj.safetensors"
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_bytes(b"sidecar")
    (high_bit / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "continuous_parameters": {
                    "schema_version": 1,
                    "enabled": True,
                    "sidecars": [
                        {
                            "layer": 1,
                            "projection": "up_proj",
                            "path": "continuous_params/layer-00001-up_proj.safetensors",
                            "tensors": ["output_bias"],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    groups = discover_artifact_group_files(
        baseline_dir=baseline,
        high_bit_dir=high_bit,
    )
    candidate = build_custom_selective_precision_candidate(
        name="l1-up",
        policy="1:up_proj",
    )

    materialized = materialize_linked_candidate(
        candidate=candidate,
        baseline_dir=baseline,
        high_bit_dir=high_bit,
        output_dir=tmp_path / "candidate",
        groups=groups,
    )
    manifest = json.loads((tmp_path / "candidate" / "conversion-manifest.json").read_text())

    assert materialized["continuous_sidecar_count"] == 1
    assert (tmp_path / "candidate" / "continuous_params" / "layer-00001-up_proj.safetensors").read_bytes() == b"sidecar"
    assert manifest["continuous_parameters"]["enabled"] is True
    assert manifest["continuous_parameters"]["sidecars"][0]["projection"] == "up_proj"


def test_materialization_drops_continuous_sidecars_for_baseline_groups(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    high_bit = tmp_path / "high-bit"
    _write_group_files(baseline, byte=b"b", layers=range(1, 2))
    _write_group_files(high_bit, byte=b"hh", layers=range(1, 2))
    sidecar_path = high_bit / "continuous_params" / "layer-00001-up_proj.safetensors"
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_bytes(b"sidecar")
    (high_bit / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "continuous_parameters": {
                    "schema_version": 1,
                    "enabled": True,
                    "sidecars": [
                        {
                            "layer": 1,
                            "projection": "up_proj",
                            "path": "continuous_params/layer-00001-up_proj.safetensors",
                            "tensors": ["output_bias"],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    groups = discover_artifact_group_files(
        baseline_dir=baseline,
        high_bit_dir=high_bit,
    )
    candidate = build_custom_selective_precision_candidate(
        name="l1-gate",
        policy="1:gate_proj",
    )

    materialized = materialize_linked_candidate(
        candidate=candidate,
        baseline_dir=baseline,
        high_bit_dir=high_bit,
        output_dir=tmp_path / "candidate",
        groups=groups,
    )

    assert materialized["continuous_sidecar_count"] == 0
    assert not (tmp_path / "candidate" / "conversion-manifest.json").exists()
    assert not (tmp_path / "candidate" / "continuous_params").exists()


def test_custom_selective_precision_candidate_parses_policy_and_rejects_bad_groups() -> None:
    candidate = build_custom_selective_precision_candidate(
        name="lane3-custom",
        policy="17:up_proj,14:gate_proj,14:up_proj,14:gate_proj",
        description="Probe a hand-ranked policy.",
        rationale=("seeded by lane3-hard16-longtail-glu",),
    )

    assert candidate.name == "lane3-custom"
    assert candidate.description == "Probe a hand-ranked policy."
    assert candidate.source == "custom_selective_precision_policy"
    assert candidate.rationale == ("seeded by lane3-hard16-longtail-glu",)
    assert candidate.high_bit_policy == (
        ("14:gate_proj", 16),
        ("14:up_proj", 16),
        ("17:up_proj", 16),
    )
    assert parse_high_bit_policy("21:gate_proj") == (("21:gate_proj", 16),)
    with pytest.raises(ValueError, match="unsupported projection"):
        parse_high_bit_policy("21:router_proj")
    with pytest.raises(ValueError, match="layer:projection"):
        parse_high_bit_policy("21")


def test_plan_cli_accepts_filtered_custom_candidate(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    high_bit = tmp_path / "high-bit"
    _write_group_files(baseline, byte=b"b", layers=range(1, 4))
    _write_group_files(high_bit, byte=b"hh", layers=range(1, 4))
    hard_report = tmp_path / "hard.json"
    hard_report.write_text(json.dumps(_report([1, 2, 3])), encoding="utf-8")
    output_json = tmp_path / "plan.json"
    materialize_root = tmp_path / "materialized"

    subprocess.run(
        [
            sys.executable,
            "benchmarks/plan_glm45_air_selective_precision.py",
            "--hard-report",
            str(hard_report),
            "--baseline-artifact-dir",
            str(baseline),
            "--high-bit-artifact-dir",
            str(high_bit),
            "--output-json",
            str(output_json),
            "--materialize-root",
            str(materialize_root),
            "--custom-candidate",
            "lane3-custom=1:gate_proj,2:up_proj",
            "--candidate",
            "lane3-custom",
        ],
        check=True,
        cwd=Path.cwd(),
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["name"] == "lane3-custom"
    assert payload["candidates"][0]["code_bits_policy"] == {
        "1:gate_proj": 16,
        "2:up_proj": 16,
    }
    assert payload["materialized"][0]["high_bit_link_count"] == 2
    assert os.readlink(
        materialize_root / "lane3-custom" / "layer-00001-gate_proj.safetensors"
    ).endswith("high-bit/layer-00001-gate_proj.safetensors")
