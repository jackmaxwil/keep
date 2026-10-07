from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mlx_vq.quality.ebss import build_ebss_prompt_selection


def load_route_records_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def write_ebss_selection_manifest(
    *,
    records_path: Path,
    out_path: Path,
    max_prompts: int,
) -> dict[str, object]:
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists")
    records = load_route_records_jsonl(records_path)
    manifest = build_ebss_prompt_selection(records, max_prompts=max_prompts)
    manifest["records_jsonl"] = str(records_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a deterministic EBSS-style GLM-4.5-Air calibration prompt subset from route records."
    )
    parser.add_argument("--records-jsonl", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--max-prompts", required=True, type=int)
    args = parser.parse_args()

    manifest = write_ebss_selection_manifest(
        records_path=args.records_jsonl,
        out_path=args.out_json,
        max_prompts=args.max_prompts,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
