from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_vq.build.promote import PromotionError, promote_step

from tests.test_build_runner_resume import _stubbed_plan
from mlx_vq.build.executor import execute_plan


def test_promote_completed_step_creates_symlink_and_stamps_status(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path)
    assert execute_plan(plan) == 0

    artifacts_root = tmp_path / "artifacts"
    target = promote_step(
        plan,
        step_id="train",
        published_name="glm45air__dmx2p0__sc-l45__r01__20260701",
        artifacts_root=artifacts_root,
    )
    assert target.is_symlink()
    manifest = json.loads((target / "conversion-manifest.json").read_text())
    assert manifest["status"] == "candidate"
    assert manifest["published_name"] == "glm45air__dmx2p0__sc-l45__r01__20260701"
    # Lineage recorded by the build survives promotion.
    assert manifest["build_steps"][-1]["step_id"] == "train"

    with pytest.raises(PromotionError, match="already exists"):
        promote_step(
            plan,
            step_id="train",
            published_name="glm45air__dmx2p0__sc-l45__r01__20260701",
            artifacts_root=artifacts_root,
        )


def test_promote_refuses_incomplete_or_nonpromotable_steps(tmp_path: Path) -> None:
    plan = _stubbed_plan(tmp_path)
    with pytest.raises(PromotionError, match="not completed"):
        promote_step(
            plan,
            step_id="train",
            published_name="glm45air__dmx2p0__sc-l45__r02__20260701",
            artifacts_root=tmp_path / "artifacts",
        )
    assert execute_plan(plan) == 0
    with pytest.raises(PromotionError, match="only promotable"):
        promote_step(
            plan,
            step_id="eval_report",
            published_name="glm45air__dmx2p0__sc-l45__r03__20260701",
            artifacts_root=tmp_path / "artifacts",
        )
    with pytest.raises(PromotionError, match="lowercase"):
        promote_step(
            plan,
            step_id="train",
            published_name="Bad Name",
            artifacts_root=tmp_path / "artifacts",
        )
