"""``keep promote``: assign a docs/naming.md immutable name to a built artifact.

Build outputs are content-key-named for freshness and resume; promotion is
the human decision that gives a completed, gate-passing step a structured
immutable name (``<base>__<repr><bpw>__<recovery>__<rev>__<date>``) via a
symlink under ``artifacts/`` and stamps the mutable manifest ``status`` tag.
Rev allocation stays a human choice — the runner never invents names.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from mlx_vq.build.executor import _atomic_write_json
from mlx_vq.build.ledger import Ledger
from mlx_vq.build.runner import BuildPlan

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_+.-]*$")


class PromotionError(ValueError):
    pass


def promote_step(
    plan: BuildPlan,
    *,
    step_id: str,
    published_name: str,
    artifacts_root: Path = Path("artifacts"),
) -> Path:
    if not _NAME_RE.match(published_name):
        raise PromotionError(
            f"published name must be lowercase with no spaces (docs/naming.md): {published_name!r}"
        )
    planned = plan.step(step_id)
    if planned.spec.step_class != "promotable":
        raise PromotionError(
            f"step {step_id!r} is class {planned.spec.step_class!r}; only promotable steps publish"
        )
    ledger = Ledger(plan.build_root / "ledger.jsonl")
    terminal = ledger.latest_terminal(planned.step_key)
    if terminal is None or terminal["event"] != "completed":
        state = terminal["event"] if terminal else "never run"
        raise PromotionError(
            f"step {step_id!r} (key {planned.step_key}) is not completed: {state}"
        )
    artifact_dir = planned.out_dir
    manifest_path = artifact_dir / "conversion-manifest.json"
    if not manifest_path.exists():
        raise PromotionError(f"no conversion-manifest.json under {artifact_dir}")

    target = artifacts_root / published_name
    if target.exists() or target.is_symlink():
        raise PromotionError(f"published name already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(os.path.relpath(artifact_dir.resolve(), target.parent.resolve()))

    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "candidate"
    manifest["published_name"] = published_name
    _atomic_write_json(manifest_path, manifest)

    ledger.append(
        "promoted",
        step_id=step_id,
        step_key=planned.step_key,
        published_name=published_name,
        published_path=str(target),
    )
    return target
