"""Standalone artifact-audit entry point for the recipe runner.

Wraps ``mlx_vq.benchmark.quant_compare.audit_vq_artifact_prefill_compatibility``
(the same audit the RC pipeline runs in-process) as a subprocess with a JSON
evidence output, so the ``audit`` op follows the same argv-recorded contract
as every other op.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    from mlx_vq.benchmark.quant_compare import audit_vq_artifact_prefill_compatibility

    audit = audit_vq_artifact_prefill_compatibility(args.artifact_dir)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, sort_keys=True) + "\n")
    print(json.dumps({"artifact_dir": args.artifact_dir, "output_json": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
