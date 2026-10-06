from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mlx_vq.io.continuous_sidecar import (
    copy_declared_continuous_sidecars,
    link_seed_artifact_groups,
)
from mlx_vq.io.logit_bias import (
    load_conversion_manifest,
    write_logit_bias_artifact_manifest,
    write_logit_bias_sidecar,
)


def _parse_token_bias(value: str) -> tuple[int, float]:
    token, sep, bias = value.partition(":")
    if sep != ":":
        raise argparse.ArgumentTypeError("token bias must use TOKEN:BIAS")
    try:
        token_id = int(token)
        bias_value = float(bias)
    except ValueError as error:
        raise argparse.ArgumentTypeError("token bias must use integer TOKEN and numeric BIAS") from error
    if token_id < 0:
        raise argparse.ArgumentTypeError("TOKEN must be non-negative")
    return token_id, bias_value


def _token_bias_map(entries: list[tuple[int, float]]) -> dict[int, float]:
    if not entries:
        raise ValueError("at least one token bias is required")
    token_biases: dict[int, float] = {}
    for token_id, bias in entries:
        if token_id in token_biases:
            raise ValueError(f"duplicate token bias for token {token_id}")
        token_biases[token_id] = bias
    return token_biases


def _parse_position_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    try:
        indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("--position-indices must be comma-separated integers") from error
    if not indices:
        raise ValueError("--position-indices must include at least one integer")
    if len(set(indices)) != len(indices):
        raise ValueError("--position-indices must be unique")
    if any(index < 0 for index in indices):
        raise ValueError("--position-indices must be non-negative")
    return indices


def _parse_token_position(value: str) -> tuple[int, tuple[int, ...]]:
    token, sep, positions = value.partition(":")
    if sep != ":":
        raise argparse.ArgumentTypeError("token position must use TOKEN:POSITION[,POSITION...]")
    try:
        token_id = int(token)
    except ValueError as error:
        raise argparse.ArgumentTypeError("token position must use integer TOKEN") from error
    if token_id < 0:
        raise argparse.ArgumentTypeError("TOKEN must be non-negative")
    try:
        indices = tuple(int(part.strip()) for part in positions.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("POSITION values must be integers") from error
    if not indices:
        raise argparse.ArgumentTypeError("token position must include at least one POSITION")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("token positions must be unique")
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("token positions must be non-negative")
    return token_id, indices


def _token_position_map(entries: list[tuple[int, tuple[int, ...]]] | None) -> dict[int, tuple[int, ...]] | None:
    if not entries:
        return None
    token_positions: dict[int, tuple[int, ...]] = {}
    for token_id, positions in entries:
        if token_id in token_positions:
            raise ValueError(f"duplicate token position scope for token {token_id}")
        token_positions[token_id] = positions
    return token_positions


def _token_position_scope_for_summary(
    *,
    token_biases: dict[int, float],
    token_position_indices: dict[int, tuple[int, ...]] | None,
) -> list[dict[str, Any]] | None:
    if token_position_indices is None:
        return None
    return [
        {
            "token_id": int(token_id),
            "position_indices": (
                None
                if token_id not in token_position_indices
                else [int(index) for index in token_position_indices[token_id]]
            ),
        }
        for token_id in token_biases
    ]


def materialize_logit_bias(
    *,
    seed_artifact_dir: str | Path,
    output_dir: str | Path,
    token_biases: dict[int, float],
    position_indices: tuple[int, ...] | None = None,
    token_position_indices: dict[int, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    seed_root = Path(seed_artifact_dir)
    output_root = Path(output_dir)
    if not token_biases:
        raise ValueError("token_biases must be non-empty")
    if position_indices is not None and token_position_indices is not None:
        raise ValueError("position_indices and token_position_indices cannot both be set")

    linked_group_count = link_seed_artifact_groups(seed_artifact_dir=seed_root, output_dir=output_root)
    preserved_sidecars = copy_declared_continuous_sidecars(seed_artifact_dir=seed_root, output_dir=output_root)
    sidecar = write_logit_bias_sidecar(
        output_dir=output_root,
        token_biases=token_biases,
        position_indices=position_indices,
        token_position_indices=token_position_indices,
    )
    seed_manifest = load_conversion_manifest(seed_root)
    token_position_scope = _token_position_scope_for_summary(
        token_biases=token_biases,
        token_position_indices=token_position_indices,
    )
    scope = {}
    if position_indices is not None:
        scope = {"position_indices": [int(index) for index in position_indices]}
    elif token_position_scope is not None:
        scope = {
            "token_position_indices": [
                entry["position_indices"] for entry in token_position_scope
            ]
        }
    run_manifest = {
        "kind": "sparse_token_logit_bias",
        "seed_artifact_dir": str(seed_root),
        "linked_group_count": int(linked_group_count),
        "preserved_sidecar_count": len(preserved_sidecars),
        "token_biases": [
            {"token_id": int(token_id), "bias": float(bias)}
            for token_id, bias in token_biases.items()
        ],
    }
    if scope:
        run_manifest["scope"] = scope
    write_logit_bias_artifact_manifest(
        seed_artifact_dir=seed_root,
        output_dir=output_root,
        sidecar=sidecar,
        run_manifest=run_manifest,
    )

    return {
        "record_type": "air_logit_bias_materialization",
        "schema_version": 1,
        "seed_artifact_dir": str(seed_root),
        "output_dir": str(output_root),
        "linked_group_count": int(linked_group_count),
        "preserved_sidecar_count": len(preserved_sidecars),
        "token_count": len(token_biases),
        "token_biases": run_manifest["token_biases"],
        "position_indices": None if position_indices is None else [int(index) for index in position_indices],
        "token_position_indices": token_position_scope,
        "seed_manifest_keys": sorted(seed_manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a GLM-4.5-Air sparse final-logit token-bias sidecar artifact."
    )
    parser.add_argument("--seed-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--token-bias",
        action="append",
        type=_parse_token_bias,
        required=True,
        help="Sparse final-logit bias as TOKEN:BIAS. Repeat for multiple tokens.",
    )
    parser.add_argument(
        "--position-indices",
        help="Optional comma-separated sequence position indices where the token bias applies.",
    )
    parser.add_argument(
        "--token-position",
        action="append",
        type=_parse_token_position,
        help=(
            "Optional token-specific scope as TOKEN:POSITION[,POSITION...]. "
            "Repeat for multiple tokens; tokens without an entry apply at all positions."
        ),
    )
    parser.add_argument("--append-jsonl", help="Optional JSONL ledger to append the materialization summary to.")
    args = parser.parse_args()

    summary = materialize_logit_bias(
        seed_artifact_dir=args.seed_artifact_dir,
        output_dir=args.output_dir,
        token_biases=_token_bias_map(args.token_bias),
        position_indices=_parse_position_indices(args.position_indices),
        token_position_indices=_token_position_map(args.token_position),
    )
    if args.append_jsonl:
        with Path(args.append_jsonl).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True) + "\n", end="")


if __name__ == "__main__":
    main()
