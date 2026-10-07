from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_memory_clean(row: dict[str, Any]) -> bool:
    return int(row.get("pageouts_delta", 0)) == 0 and int(row.get("swapouts_delta", 0)) == 0


def _row_shard(row: dict[str, Any]) -> str:
    shard = row.get("logit_shard") or row.get("shard")
    if not isinstance(shard, str) or not shard:
        raise ValueError(f"row for {row.get('prompt_id')} does not contain a shard path")
    return shard


def _index_rows(cache_dir: Path) -> dict[str, dict[str, Any]]:
    rows = _read_rows(cache_dir / "metadata.jsonl")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"row in {cache_dir} has invalid prompt_id: {prompt_id!r}")
        if prompt_id in indexed:
            raise ValueError(f"duplicate prompt_id {prompt_id!r} in {cache_dir}")
        indexed[prompt_id] = row
    return indexed


def merge_clean_cache(*, base_dir: Path, repair_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    base_rows = _read_rows(base_dir / "metadata.jsonl")
    repair_rows: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for repair_dir in repair_dirs:
        for prompt_id, row in _index_rows(repair_dir).items():
            repair_rows.setdefault(prompt_id, []).append((repair_dir, row))

    selected: list[tuple[Path, dict[str, Any], str]] = []
    replacements: list[str] = []
    unresolved_dirty: list[str] = []
    for row in base_rows:
        prompt_id = row["prompt_id"]
        if _is_memory_clean(row):
            selected.append((base_dir, row, "base"))
            continue

        clean_repair = next(
            ((repair_dir, repair_row) for repair_dir, repair_row in repair_rows.get(prompt_id, []) if _is_memory_clean(repair_row)),
            None,
        )
        if clean_repair is None:
            unresolved_dirty.append(prompt_id)
            selected.append((base_dir, row, "dirty-base"))
            continue
        repair_dir, repair_row = clean_repair
        selected.append((repair_dir, repair_row, "repair"))
        replacements.append(prompt_id)

    if unresolved_dirty:
        raise SystemExit(
            "missing clean replacements for dirty prompt ids: " + ", ".join(sorted(unresolved_dirty))
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "teacher_logits").mkdir(parents=True, exist_ok=True)

    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for source_dir, row, _source_kind in selected:
            shard = _row_shard(row)
            source = source_dir / shard
            destination = output_dir / shard
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "repair_dirs": [str(path) for path in repair_dirs],
        "row_count": len(selected),
        "replacement_count": len(replacements),
        "replacement_prompt_ids": replacements,
        "all_memory_clean": all(_is_memory_clean(row) for _source_dir, row, _source_kind in selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge clean repair rows into a GLM-4.5-Air teacher cache.")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--repair-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary = merge_clean_cache(
        base_dir=Path(args.base_dir),
        repair_dirs=[Path(path) for path in args.repair_dir],
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_memory_clean"]:
        raise SystemExit("merged cache is not memory clean")


if __name__ == "__main__":
    main()
