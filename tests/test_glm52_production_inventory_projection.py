from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mlx_vq.quality.glm52_s3_artifact_audit import (
    build_s3_artifact_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "aws/glm52-gpu/scripts/project_glm52_production_inventory_objects.py"
)


def _inventory(*, version_id: str | None = "source-v1") -> dict[str, object]:
    item: dict[str, object] = {
        "key": "source-snapshot/config.json",
        "size": 12,
        "sha256": "a" * 64,
        "kind": "source_model",
        "safetensors": False,
        "run_scope": "shared",
    }
    if version_id is not None:
        item["version_id"] = version_id
    repository = {
        "key": (
            "campaigns/glm52-sky-20260729/repository/"
            f"keep-{'b' * 64}.tar.gz"
        ),
        "size": 20,
        "sha256": "b" * 64,
        "kind": "repository_tar",
        "safetensors": False,
        "run_scope": "shared",
        "version_id": "repository-v1",
    }
    return build_s3_artifact_inventory(
        run_id="glm52-sky-20260729",
        bucket="keep-glm52-models-246813579024-us-west-2",
        objects=[item, repository],
    )


def test_projection_emits_exact_versioned_object_list(tmp_path: Path) -> None:
    source = tmp_path / "inventory.json"
    output = tmp_path / "objects.json"
    source.write_text(json.dumps(_inventory(), sort_keys=True) + "\n")
    result = subprocess.run(
        [
            str(SCRIPT),
            "--inventory",
            str(source),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    objects = json.loads(output.read_bytes())
    assert objects == _inventory()["objects"]


def test_projection_rejects_unversioned_inventory(tmp_path: Path) -> None:
    source = tmp_path / "inventory.json"
    source.write_text(
        json.dumps(_inventory(version_id=None), sort_keys=True) + "\n"
    )
    result = subprocess.run(
        [
            str(SCRIPT),
            "--inventory",
            str(source),
            "--output",
            str(tmp_path / "objects.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires exact VersionIds" in result.stderr
