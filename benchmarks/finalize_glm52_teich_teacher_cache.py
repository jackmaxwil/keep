"""Audit raw full-v2 Teich captures and publish schema-v3 cache readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--prompt-pack", type=Path, required=True)
    parser.add_argument("--frozen-prompt-pack", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-sessions", type=int, default=257)
    parser.add_argument("--expected-supervised-positions", type=int, default=2_500_735)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=6144)
    args = parser.parse_args()

    from mlx_vq.quality.glm52_teich_training_cache import (
        finalize_glm52_teich_teacher_cache,
    )

    ready = finalize_glm52_teich_teacher_cache(
        capture_dir=args.capture_dir,
        prompt_pack_path=args.prompt_pack,
        frozen_prompt_pack_path=args.frozen_prompt_pack,
        cache_dir=args.cache_dir,
        expected_session_count=args.expected_sessions,
        expected_supervised_positions=args.expected_supervised_positions,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
    )
    print(json.dumps(ready, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
