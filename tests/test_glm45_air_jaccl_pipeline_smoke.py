from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_jaccl_pipeline_smoke_singleton_mode_reports_payload_shape() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/smoke_glm45_air_jaccl_pipeline.py",
            "--backend",
            "ring",
            "--tokens",
            "2",
            "--hidden-size",
            "4",
            "--iterations",
            "1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    record = json.loads(completed.stdout)
    assert record["status"] == "singleton"
    assert record["rank"] == 0
    assert record["size"] == 1
    assert record["tokens"] == 2
    assert record["hidden_size"] == 4
    assert record["payload_bytes"] == 16
    assert "point-to-point send/recv was skipped" in record["note"]


def test_jaccl_pipeline_smoke_rejects_invalid_shape_before_init() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/smoke_glm45_air_jaccl_pipeline.py",
            "--backend",
            "ring",
            "--tokens",
            "0",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--tokens must be positive" in completed.stderr
