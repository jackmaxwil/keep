from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_views_cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "materialize_glm45_air_pipeline_views.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_views_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pipeline_view_test_helpers():
    path = Path(__file__).with_name("test_glm45_air_pipeline_views.py")
    spec = importlib.util.spec_from_file_location("pipeline_view_helpers", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_materialized_synthetic_views(tmp_path: Path) -> Path:
    helpers = _load_pipeline_view_test_helpers()
    views_cli = _load_views_cli()
    index_path = helpers._write_synthetic_source(tmp_path)
    plan = views_cli.build_pipeline_view_plan(
        source_dir=tmp_path,
        index_path=index_path,
        rank_budget_bytes=1_000_000,
    )
    out_dir = tmp_path / "views"
    views_cli.materialize_pipeline_views(plan, output_dir=out_dir)
    return out_dir


def test_distributed_pipeline_forward_probe_singleton_checks_rank_view(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    views_dir = _write_materialized_synthetic_views(tmp_path)
    roots = {"0": str(views_dir / "rank-0")}

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/probe_glm45_air_distributed_pipeline_forward.py",
            "--backend",
            "ring",
            "--rank-view-roots-json",
            json.dumps(roots),
            "--pipeline-size",
            "2",
            "--sentinel-max-elements",
            "1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    record = json.loads(completed.stdout)
    assert record["rank"] == 0
    assert record["distributed_size"] == 1
    assert record["pipeline_size"] == 2
    assert record["first_layer"] == 2
    assert record["last_layer"] == 3
    assert record["parameter_keys_match_plan"] is True
    assert record["did_eval_sentinels"] is True
    assert record["did_eval_full_weights"] is False
    assert record["forward_attempted"] is False
    assert record["sentinel_results"]
    assert record["pre_forward_all_sum"] == 1.0


def test_distributed_pipeline_forward_probe_rejects_bad_roots_json_before_init(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/probe_glm45_air_distributed_pipeline_forward.py",
            "--backend",
            "ring",
            "--rank-view-roots-json",
            "[]",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--rank-view-roots-json must be a JSON object" in completed.stderr


def test_distributed_pipeline_forward_probe_rejects_empty_forward_tokens(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/probe_glm45_air_distributed_pipeline_forward.py",
            "--backend",
            "ring",
            "--rank-view-roots-json",
            json.dumps({"0": str(tmp_path)}),
            "--attempt-forward",
            "--input-token-ids",
            "",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--input-token-ids must not be empty" in completed.stderr
