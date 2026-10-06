"""Runtime descriptor dispatch tests for legacy Capacity Block and SkyPilot v2."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mlx_vq.quality.glm52_campaign_descriptor import (
    RuntimeCampaignKind,
    resolve_runtime_authority,
    validate_runtime_descriptor,
)
from mlx_vq.quality.glm52_sky_campaign import (
    build_gpu_spend_approval,
    build_sky_campaign_descriptor,
)


def _sha(character: str) -> str:
    return character * 64


def _sky_descriptor() -> dict[str, object]:
    approval = build_gpu_spend_approval(
        ingested_at=datetime(2026, 7, 23, 18, 5, tzinfo=timezone.utc),
        slack_permalink=None,
    )
    return build_sky_campaign_descriptor(
        run_id="glm52-sky-20260723",
        must_start_by=datetime(2026, 7, 24, 6, 5, tzinfo=timezone.utc),
        controller_identity=(
            "arn:aws:iam::246813579024:role/keep-glm52-skypilot-controller"
        ),
        worker_identity="arn:aws:iam::246813579024:role/keep-glm52-gpu-worker",
        vpc_name="keep-glm52-vpc",
        image_id="ami-0123456789abcdef0",
        bucket="keep-glm52-us-west-2-246813579024",
        jobs_bucket="keep-glm52-us-west-2-246813579024",
        repo_tar_key="campaigns/glm52-sky-20260723/repository/repo.tar.gz",
        repo_tar_sha256=_sha("1"),
        campaign_descriptor_key=(
            "campaigns/glm52-sky-20260723/submissions/first/campaign-descriptor-v2.json"
        ),
        approval_key=(
            "campaigns/glm52-sky-20260723/authorities/GPU_SPEND_APPROVAL.json"
        ),
        approval_sha256=str(approval["approval_body_sha256"]),
        artifacts={
            "source_snapshot_prefix": "source-snapshot/",
            "source_snapshot_sha256": _sha("2"),
            "non_vq_prefix": "non-vq-package/",
            "non_vq_package_sha256": _sha("3"),
            "teich_pack_key": "teich-pack/pack.json",
            "teich_pack_sha256": _sha("4"),
            "frozen_prompt_pack_key": "quality/frozen.json",
            "frozen_prompt_pack_sha256": _sha("5"),
            "training_baseline_prefix": "training-baseline/",
            "training_baseline_sha256": _sha("6"),
            "training_config_key": (
                "campaigns/glm52-sky-20260723/authorities/training.json"
            ),
            "training_config_sha256": _sha("7"),
            "artifact_inventory_key": (
                "campaigns/glm52-sky-20260723/inventories/"
                f"artifact-inventory-{_sha('8')}.json"
            ),
            "artifact_inventory_sha256": _sha("8"),
            "qualification_cache_prefix": (
                "qualification-cache/seeds/glm52-sky-20260723/"
                f"{_sha('9')}/"
            ),
            "qualification_cache_manifest_sha256": _sha("9"),
        },
    )


def test_sky_runtime_dispatch_does_not_need_capacity_block_fields() -> None:
    descriptor, kind = validate_runtime_descriptor(_sky_descriptor())
    assert kind is RuntimeCampaignKind.SKYPILOT
    assert descriptor["record_type"] == "glm52_sky_campaign_descriptor_v2"
    assert "capacity_block_end" not in descriptor


def test_sky_runtime_authority_requires_allocation_and_spend_hashes() -> None:
    descriptor = _sky_descriptor()
    authority = resolve_runtime_authority(
        descriptor,
        environ={
            "GLM52_EXECUTION_DEADLINE": "2026-07-25T08:00:00Z",
            "GLM52_GPU_ALLOCATION_SHA256": _sha("a"),
            "GLM52_GPU_SPEND_AUTHORITY_SHA256": _sha("b"),
        },
    )
    assert authority.kind is RuntimeCampaignKind.SKYPILOT
    assert authority.execution_deadline.isoformat() == "2026-07-25T08:00:00+00:00"
    assert authority.gpu_allocation_sha256 == _sha("a")
    assert authority.gpu_spend_authority_sha256 == _sha("b")

    with pytest.raises(ValueError, match="GLM52_EXECUTION_DEADLINE"):
        resolve_runtime_authority(descriptor, environ={})


def _load_campaign_runner():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/run_glm52_campaign.py"
    spec = importlib.util.spec_from_file_location("_glm52_campaign_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_campaign_runner_uses_sky_ledger_and_generic_deadline(
    tmp_path, monkeypatch
) -> None:
    descriptor_path = tmp_path / "campaign.json"
    descriptor_path.write_text(json.dumps(_sky_descriptor()))
    monkeypatch.setenv("GLM52_EXECUTION_DEADLINE", "2026-07-25T08:00:00Z")
    monkeypatch.setenv("GLM52_GPU_ALLOCATION_SHA256", _sha("a"))
    monkeypatch.setenv("GLM52_GPU_SPEND_AUTHORITY_SHA256", _sha("b"))
    monkeypatch.setenv(
        "GLM52_CAMPAIGN_STOP_FILE", str(tmp_path / "run/keep-glm52/STOP")
    )
    runner = _load_campaign_runner()

    controller = runner.CampaignController(
        descriptor_path=descriptor_path,
        root=tmp_path / "campaign-root",
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert controller.runtime.kind is RuntimeCampaignKind.SKYPILOT
    assert controller.execution_deadline.isoformat() == "2026-07-25T08:00:00+00:00"
    assert controller.ledger().__class__.__name__ == "SkyCampaignLedger"
    assert controller.teacher_deadline_arguments() == [
        "--execution-deadline",
        "2026-07-25T08:00:00Z",
    ]
