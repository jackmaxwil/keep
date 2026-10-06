from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "aws/glm52-gpu/scripts/build_production_s3_inventory.py"
try:
    import boto3 as _boto3  # noqa: F401
except ModuleNotFoundError:
    stubbed_boto3 = True
    sys.modules["boto3"] = types.ModuleType("boto3")
else:
    stubbed_boto3 = False
SPEC = importlib.util.spec_from_file_location("_glm52_production_inventory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
INVENTORY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INVENTORY
SPEC.loader.exec_module(INVENTORY)
if stubbed_boto3:
    sys.modules.pop("boto3", None)


def test_non_vq_index_record_uses_package_index_not_source_index() -> None:
    record = INVENTORY._non_vq_index_record(  # type: ignore[attr-defined]
        {
            "index_sha256": "a" * 64,
            "package_index_sha256": "b" * 64,
        },
        size=105_551,
        version_id="index-version-1",
    )

    assert record == {
        "key": "non-vq-package/model.safetensors.index.json",
        "kind": "non_vq_package",
        "run_scope": "shared",
        "safetensors": False,
        "sha256": "b" * 64,
        "size": 105_551,
        "version_id": "index-version-1",
    }


def test_record_requires_and_preserves_exact_s3_version_id() -> None:
    record = INVENTORY._record(  # type: ignore[attr-defined]
        key="source-snapshot/config.json",
        size=42,
        sha256="c" * 64,
        kind="source_model",
        safetensors=False,
        version_id="source-version-7",
    )
    assert record["version_id"] == "source-version-7"
