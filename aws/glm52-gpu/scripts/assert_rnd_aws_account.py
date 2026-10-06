#!/usr/bin/env python3
"""Read STS identity JSON on stdin and refuse any non-R&D AWS account."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "src/mlx_vq/quality/glm52_sky_campaign.py"
SPEC = importlib.util.spec_from_file_location("_glm52_sky_campaign", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
SKY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SKY
SPEC.loader.exec_module(SKY)
SkyCampaignValidationError = SKY.SkyCampaignValidationError
require_approved_aws_identity = SKY.require_approved_aws_identity


def main() -> int:
    try:
        value = json.load(sys.stdin)
        if not isinstance(value, dict):
            raise SkyCampaignValidationError("STS identity must be an object")
        identity = require_approved_aws_identity(value)
    except (json.JSONDecodeError, SkyCampaignValidationError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(identity["Account"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
