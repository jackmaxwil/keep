from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from keep.io.continuous_sidecar import (
    copy_declared_continuous_sidecars,
    link_seed_artifact_groups,
    load_conversion_manifest,
)


SUPPORTED_SURFACES = {
    "embed_tokens",
    "lm_head",
    "router_gates",
}
SUPPORTED_DTYPES = {"bf16", "fp16", "fp32", "source_bfloat16"}
NON_EXPERT_PRECISION_MANIFEST_KEY = "non_expert_precision"


def _copy_optional_tree(seed_root: Path, output_root: Path, name: str) -> bool:
    source = seed_root / name
    if not source.exists():
        return False
    target = output_root / name
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    shutil.copytree(source, target, symlinks=True)
    return True


def _normalize_surfaces(surfaces: list[str]) -> list[str]:
    normalized: list[str] = []
    for surface in surfaces:
        for part in str(surface).split(","):
            cleaned = part.strip()
            if not cleaned:
                continue
            if cleaned not in SUPPORTED_SURFACES:
                raise ValueError(
                    f"unsupported non-expert surface {cleaned!r}; "
                    f"expected one of {sorted(SUPPORTED_SURFACES)}"
                )
            if cleaned not in normalized:
                normalized.append(cleaned)
    if not normalized:
        raise ValueError("at least one non-expert surface is required")
    return normalized


def _normalize_dtype(dtype: str) -> str:
    cleaned = str(dtype).strip().lower()
    aliases = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
    cleaned = aliases.get(cleaned, cleaned)
    if cleaned not in SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported non-expert dtype {dtype!r}; expected one of {sorted(SUPPORTED_DTYPES)}"
        )
    return cleaned


def materialize_non_expert_precision_candidate(
    *,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
    surfaces: list[str],
    dtype: str,
    reason: str | None = None,
) -> dict[str, Any]:
    seed_root = Path(seed_artifact_dir)
    output_root = Path(output_dir)
    if output_root.exists():
        raise FileExistsError(f"{output_root} already exists")

    selected_surfaces = _normalize_surfaces(surfaces)
    selected_dtype = _normalize_dtype(dtype)
    linked_group_count = link_seed_artifact_groups(
        seed_artifact_dir=seed_root,
        output_dir=output_root,
    )
    copied_sidecars = copy_declared_continuous_sidecars(
        seed_artifact_dir=seed_root,
        output_dir=output_root,
    )
    router_corrections_copied = _copy_optional_tree(seed_root, output_root, "router_corrections")

    manifest = dict(load_conversion_manifest(seed_root))
    manifest[NON_EXPERT_PRECISION_MANIFEST_KEY] = {
        "schema_version": 1,
        "enabled": True,
        "kind": "source_precision_policy",
        "surfaces": selected_surfaces,
        "dtype": selected_dtype,
        "reason": reason or "",
        "seed_artifact_dir": str(seed_root),
        "linked_group_count": int(linked_group_count),
        "preserved_sidecar_count": len(copied_sidecars),
        "router_corrections_copied": bool(router_corrections_copied),
        "lane_s_cost": "off_routed_nax_critical_path",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "conversion-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a linked GLM-4.5-Air candidate with an explicit "
            "high-precision non-expert surface policy."
        )
    )
    parser.add_argument("--seed-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--surfaces", required=True)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--reason", default="")
    parser.add_argument("--append-jsonl")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        manifest = materialize_non_expert_precision_candidate(
            seed_artifact_dir=args.seed_artifact_dir,
            output_dir=args.output_dir,
            surfaces=[args.surfaces],
            dtype=args.dtype,
            reason=args.reason,
        )
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    summary = {
        "record_type": "air_non_expert_precision_materialization",
        "schema_version": 1,
        "seed_artifact_dir": args.seed_artifact_dir,
        "output_dir": args.output_dir,
        **manifest[NON_EXPERT_PRECISION_MANIFEST_KEY],
    }
    if args.append_jsonl:
        with Path(args.append_jsonl).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
