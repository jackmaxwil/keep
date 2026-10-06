from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlx_vq.quality.glm52_sky_training_config import (
    build_sky_training_config,
    validate_sky_training_config,
)

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "aws/glm52-gpu/scripts/build_sky_training_config.py"


def test_training_config_locks_canonical_candidate_and_resume_schedule() -> None:
    config = validate_sky_training_config(build_sky_training_config())
    assert config["seed"] == 20260712
    assert config["window_size"] == 64
    assert config["checkpoint_every_steps"] == 50
    assert config["s3_sync_every_steps"] == 250
    assert config["s3_sync_every_seconds"] == 300
    assert config["layer"] == 77
    assert config["projections"] == ["gate_proj", "up_proj", "down_proj"]
    assert config["rank"] == 4
    assert config["learning_rate"] == 0.2
    assert config["topk"] == 2048
    assert config["expansion_layers"] == [75, 74, 76, 72, 73, 71, 70]
    assert config["automatic_promotion"] is False


def test_training_config_rejects_unknown_or_changed_fields() -> None:
    config = build_sky_training_config()
    with pytest.raises(ValueError, match="schema"):
        validate_sky_training_config({**config, "surprise": True})
    with pytest.raises(ValueError, match="body SHA-256"):
        validate_sky_training_config({**config, "rank": 8})


def test_training_config_builder_does_not_import_mlx_package(tmp_path) -> None:
    output = tmp_path / "training-config.json"
    wrapper = """
import importlib.abc
import runpy
import sys

class RejectMlxPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx_vq" or fullname.startswith("mlx_vq."):
            raise ImportError(f"control-plane builder imported {fullname}")
        return None

sys.meta_path.insert(0, RejectMlxPackage())
sys.argv = [sys.argv[1], "--output", sys.argv[2]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", wrapper, str(BUILDER), str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    validate_sky_training_config(json.loads(output.read_text()))
