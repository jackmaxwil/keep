from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from keep.io.continuous_sidecar import (
    copy_declared_continuous_sidecars,
    write_continuous_artifact_manifest,
)


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
GLU_PROJECTIONS = ("gate_proj", "up_proj")


@dataclass(frozen=True)
class ArtifactGroupFile:
    layer: int
    projection: str
    relative_path: str
    baseline_bytes: int
    high_bit_bytes: int

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.projection}"


@dataclass(frozen=True)
class SelectivePrecisionCandidate:
    name: str
    description: str
    high_bit_policy: tuple[tuple[str, int], ...]
    source: str
    rationale: tuple[str, ...]

    @property
    def policy_dict(self) -> dict[str, int]:
        return dict(self.high_bit_policy)

    @property
    def high_bit_keys(self) -> tuple[str, ...]:
        return tuple(key for key, bits in self.high_bit_policy if bits == 16)


def _layer_rankings(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(report.get("summary", {}).get("layer_rankings", []))


def _projection_rankings(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(report.get("summary", {}).get("layer_projection_rankings", []))


def _source_path(report: Mapping[str, Any]) -> str:
    return str(report.get("output_json") or report.get("path") or "")


def _policy_entries(groups: Iterable[tuple[int, str]]) -> tuple[tuple[str, int], ...]:
    seen: set[str] = set()
    entries: list[tuple[str, int]] = []
    for layer, projection in sorted(groups):
        key = f"{int(layer)}:{projection}"
        if projection not in PROJECTIONS:
            raise ValueError(f"unsupported projection {projection!r}")
        if key not in seen:
            entries.append((key, 16))
            seen.add(key)
    if not entries:
        raise ValueError("candidate policy must contain at least one high-bit group")
    return tuple(entries)


def parse_high_bit_policy(policy: str) -> tuple[tuple[str, int], ...]:
    """Parse a comma-separated layer/projection high-bit policy.

    Policy items use ``layer:projection`` keys such as ``14:gate_proj``.
    Duplicate items are ignored after validation.
    """

    groups: list[tuple[int, str]] = []
    for raw_item in policy.split(","):
        item = raw_item.strip()
        if not item:
            continue
        layer_text, separator, projection = item.partition(":")
        if separator != ":" or not layer_text or not projection:
            raise ValueError(
                f"custom policy item {item!r} must use layer:projection format"
            )
        try:
            layer = int(layer_text)
        except ValueError as exc:
            raise ValueError(f"custom policy layer must be an integer: {item!r}") from exc
        groups.append((layer, projection.strip()))
    return _policy_entries(groups)


def build_custom_selective_precision_candidate(
    *,
    name: str,
    policy: str,
    description: str | None = None,
    source: str = "custom_selective_precision_policy",
    rationale: Iterable[str] = (),
) -> SelectivePrecisionCandidate:
    """Build a reproducible hand-ranked selective precision candidate."""

    candidate_name = name.strip()
    if not candidate_name:
        raise ValueError("custom candidate name must not be empty")
    high_bit_policy = parse_high_bit_policy(policy)
    high_bit_keys = ", ".join(key for key, _bits in high_bit_policy)
    return SelectivePrecisionCandidate(
        name=candidate_name,
        description=description or f"Custom selective precision policy: {high_bit_keys}.",
        high_bit_policy=high_bit_policy,
        source=source,
        rationale=tuple(rationale) or (f"Explicit policy supplied: {high_bit_keys}.",),
    )


def _layers_from_rankings(rankings: list[dict[str, Any]], *, limit: int) -> tuple[int, ...]:
    layers = []
    for row in rankings[:limit]:
        layer = int(row["layer"])
        if layer not in layers:
            layers.append(layer)
    return tuple(layers)


def _layer_glu_groups(layers: Iterable[int]) -> tuple[tuple[int, str], ...]:
    return tuple((int(layer), projection) for layer in layers for projection in GLU_PROJECTIONS)


def _top_layer_glu_policy(report: Mapping[str, Any]) -> tuple[tuple[int, str], ...]:
    rankings = _layer_rankings(report)
    if not rankings:
        raise ValueError("hard report has no layer rankings")
    top_layer = int(rankings[0]["layer"])
    projection_rows = [
        row for row in _projection_rankings(report)
        if int(row.get("layer", -1)) == top_layer
    ]
    groups: set[tuple[int, str]] = set()
    for row in projection_rows[:4]:
        metric = str(row.get("metric", ""))
        if metric == "source_gate_proj":
            groups.add((top_layer, "gate_proj"))
        elif metric == "source_up_proj":
            groups.add((top_layer, "up_proj"))
        elif metric == "source_routed_glu":
            groups.update(_layer_glu_groups((top_layer,)))
    if not groups:
        groups.update(_layer_glu_groups((top_layer,)))
    if any(
        str(row.get("metric")) == "source_routed_glu"
        for row in projection_rows[:4]
    ):
        groups.update(_layer_glu_groups((top_layer,)))
    return tuple(sorted(groups))


def build_selective_precision_candidates(
    *,
    hard_report: Mapping[str, Any],
    tail_report: Mapping[str, Any] | None = None,
    secondary_hard_layers: int = 3,
    tail_layers: int = 4,
) -> tuple[SelectivePrecisionCandidate, ...]:
    """Build Lane 3 candidate policies from Lane 2 source-oracle reports.

    The current artifact format can raise precision only for whole
    layer/projection groups. Expert ids from Lane 2 remain rationale, but are not
    expressible in the resident artifact yet.
    """

    if secondary_hard_layers <= 0:
        raise ValueError("secondary_hard_layers must be positive")
    if tail_layers <= 0:
        raise ValueError("tail_layers must be positive")

    hard_rankings = _layer_rankings(hard_report)
    if not hard_rankings:
        raise ValueError("hard report has no layer rankings")

    hard_top_layer = int(hard_rankings[0]["layer"])
    hard_top_up_groups = ((hard_top_layer, "up_proj"),)
    hard_top_groups = _top_layer_glu_policy(hard_report)
    hard_layers = _layers_from_rankings(hard_rankings, limit=secondary_hard_layers)

    candidates = [
        SelectivePrecisionCandidate(
            name="lane3-hard16-up",
            description=(
                "Minimal hard-token up-projection candidate: raise only the "
                "highest-ranked hard-token layer's up projection to E8P."
            ),
            high_bit_policy=_policy_entries(hard_top_up_groups),
            source="hard_first_token_source_oracle",
            rationale=(
                f"Hard-token source-oracle report ranks layer {hard_top_layer} first.",
                "The top layer/projection row is source_up_proj in the current hard-token report.",
                "This isolates whether up precision helps before protecting the full GLU pair.",
            ),
        ),
        SelectivePrecisionCandidate(
            name="lane3-hard16-glu",
            description=(
                "Minimal hard-token GLU candidate: raise the highest-ranked hard-token "
                "layer's gate/up projections to E8P."
            ),
            high_bit_policy=_policy_entries(hard_top_groups),
            source="hard_first_token_source_oracle",
            rationale=(
                f"Hard-token source-oracle report ranks layer {hard_top_layer} first.",
                "Layer/projection ranking puts the top-layer up projection and routed GLU ahead of other rows.",
                "Resident artifact format is layer/projection-granular, so selected experts are rationale only.",
            ),
        ),
        SelectivePrecisionCandidate(
            name="lane3-hard-top3-glu",
            description=(
                "Hard-token top-3 GLU candidate: raise gate/up projections on the top "
                "three hard-token layers."
            ),
            high_bit_policy=_policy_entries(_layer_glu_groups(hard_layers)),
            source="hard_first_token_source_oracle",
            rationale=(
                "Hard first-token coarse ranking drives the layer set.",
                f"Selected layers: {','.join(str(layer) for layer in hard_layers)}.",
                "Routed GLU dominance maps to gate/up projection protection.",
            ),
        ),
    ]

    if tail_report is not None:
        tail_rankings = _layer_rankings(tail_report)
        if tail_rankings:
            selected_tail_layers = _layers_from_rankings(tail_rankings, limit=tail_layers)
            combined_layers = tuple(sorted(set(selected_tail_layers) | {hard_top_layer}))
            candidates.append(
                SelectivePrecisionCandidate(
                    name="lane3-hard16-longtail-glu",
                    description=(
                        "Hard-token plus long-tail GLU candidate: protect the hard-token "
                        "top layer and the strongest long-recall tail layers."
                    ),
                    high_bit_policy=_policy_entries(_layer_glu_groups(combined_layers)),
                    source="hard_first_token_and_long_recall_tail_source_oracles",
                    rationale=(
                        f"Hard-token source-oracle report ranks layer {hard_top_layer} first.",
                        (
                            "Long-recall tail ranking contributes layers "
                            f"{','.join(str(layer) for layer in selected_tail_layers)}."
                        ),
                        "Tail probe dominance is routed GLU, so protection maps to gate/up projections.",
                    ),
                )
            )

            combined_hard_tail = tuple(sorted(set(hard_layers) | set(selected_tail_layers)))
            candidates.append(
                SelectivePrecisionCandidate(
                    name="lane3-combined-top-glu",
                    description=(
                        "Combined top GLU candidate: protect top hard-token layers plus "
                        "the strongest long-recall tail layers."
                    ),
                    high_bit_policy=_policy_entries(_layer_glu_groups(combined_hard_tail)),
                    source="hard_first_token_and_long_recall_tail_source_oracles",
                    rationale=(
                        f"Hard layers: {','.join(str(layer) for layer in hard_layers)}.",
                        (
                            "Long-recall tail layers: "
                            f"{','.join(str(layer) for layer in selected_tail_layers)}."
                        ),
                        "This is the broadest current n=5 allocation seed and remains directional.",
                    ),
                )
            )

    return tuple(candidates)


def discover_artifact_group_files(
    *,
    baseline_dir: str | Path,
    high_bit_dir: str | Path,
) -> tuple[ArtifactGroupFile, ...]:
    baseline_root = Path(baseline_dir)
    high_bit_root = Path(high_bit_dir)
    groups: list[ArtifactGroupFile] = []
    for path in sorted(baseline_root.glob("layer-*-*.safetensors")):
        parsed = parse_group_filename(path.name)
        if parsed is None:
            continue
        layer, projection = parsed
        high_bit_path = high_bit_root / path.name
        if not high_bit_path.exists():
            raise FileNotFoundError(f"high-bit artifact is missing {path.name}")
        groups.append(
            ArtifactGroupFile(
                layer=layer,
                projection=projection,
                relative_path=path.name,
                baseline_bytes=path.stat().st_size,
                high_bit_bytes=high_bit_path.stat().st_size,
            )
        )
    if not groups:
        raise ValueError(f"no layer group files found under {baseline_root}")
    return tuple(groups)


def parse_group_filename(filename: str) -> tuple[int, str] | None:
    if not filename.startswith("layer-") or not filename.endswith(".safetensors"):
        return None
    stem = filename.removesuffix(".safetensors")
    try:
        _, layer_text, projection = stem.split("-", 2)
    except ValueError:
        return None
    if projection not in PROJECTIONS:
        return None
    return int(layer_text), projection


def estimate_candidate_storage(
    candidate: SelectivePrecisionCandidate,
    groups: tuple[ArtifactGroupFile, ...],
) -> dict[str, Any]:
    high_bit_keys = set(candidate.high_bit_keys)
    baseline_bytes = sum(group.baseline_bytes for group in groups)
    high_bit_bytes = 0
    selected_groups = []
    for group in groups:
        if group.key in high_bit_keys:
            high_bit_bytes += group.high_bit_bytes
            selected_groups.append(group)
        else:
            high_bit_bytes += group.baseline_bytes
    missing = sorted(high_bit_keys - {group.key for group in groups})
    if missing:
        raise ValueError(f"candidate policy references missing artifact groups: {missing}")
    added_bytes = high_bit_bytes - baseline_bytes
    return {
        "baseline_logical_bytes": baseline_bytes,
        "candidate_logical_bytes": high_bit_bytes,
        "added_logical_bytes": added_bytes,
        "size_ratio_vs_baseline": high_bit_bytes / baseline_bytes if baseline_bytes else 0.0,
        "high_bit_group_count": len(selected_groups),
        "total_group_count": len(groups),
        "high_bit_groups": [group.key for group in selected_groups],
    }


def candidate_to_json(
    candidate: SelectivePrecisionCandidate,
    groups: tuple[ArtifactGroupFile, ...],
) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "description": candidate.description,
        "source": candidate.source,
        "rationale": list(candidate.rationale),
        "code_bits_policy": dict(candidate.high_bit_policy),
        "storage": estimate_candidate_storage(candidate, groups),
        "runtime_note": (
            "Mixed 16-bit groups are resident-safe but disable the NAX E8 fast path "
            "for any layer containing a 16-bit projection; those layers fall back to "
            "the Metal VQ path."
        ),
    }


def materialize_linked_candidate(
    *,
    candidate: SelectivePrecisionCandidate,
    baseline_dir: str | Path,
    high_bit_dir: str | Path,
    output_dir: str | Path,
    groups: tuple[ArtifactGroupFile, ...] | None = None,
    allow_existing: bool = False,
) -> dict[str, Any]:
    baseline_root = Path(baseline_dir)
    high_bit_root = Path(high_bit_dir)
    output_root = Path(output_dir)
    preexisting = output_root.exists()
    if preexisting and not allow_existing:
        raise FileExistsError(f"{output_root} already exists")
    if not preexisting:
        output_root.mkdir(parents=True)
    groups = groups or discover_artifact_group_files(
        baseline_dir=baseline_root,
        high_bit_dir=high_bit_root,
    )
    high_bit_keys = set(candidate.high_bit_keys)
    linked = []
    high_bit_count = 0
    baseline_count = 0
    for group in groups:
        source_root = high_bit_root if group.key in high_bit_keys else baseline_root
        if source_root == high_bit_root:
            high_bit_count += 1
        else:
            baseline_count += 1
        source_path = source_root / group.relative_path
        target_path = output_root / group.relative_path
        if preexisting:
            if not target_path.exists():
                raise FileNotFoundError(f"existing candidate is missing {target_path}")
        else:
            os.symlink(_relative_symlink_target(source_path, target_path.parent), target_path)
        linked.append({
            "group": group.key,
            "source": "high_bit" if group.key in high_bit_keys else "baseline",
            "path": group.relative_path,
        })
    sidecars = copy_declared_continuous_sidecars(
        seed_artifact_dir=high_bit_root,
        output_dir=output_root,
        exclude={(group.layer, group.projection) for group in groups if group.key not in high_bit_keys},
    )
    if sidecars:
        write_continuous_artifact_manifest(
            seed_artifact_dir=high_bit_root,
            output_dir=output_root,
            sidecars=sidecars,
            run_manifest={
                "source": "selective_precision_linked_candidate",
                "candidate": candidate.name,
                "baseline_artifact_dir": str(baseline_root),
                "high_bit_artifact_dir": str(high_bit_root),
            },
        )
    return {
        "output_dir": str(output_root),
        "preexisting": preexisting,
        "linked_group_count": len(linked),
        "high_bit_link_count": high_bit_count,
        "baseline_link_count": baseline_count,
        "continuous_sidecar_count": len(sidecars),
        "links": linked,
    }


def _relative_symlink_target(source_path: Path, target_parent: Path) -> str:
    return os.path.relpath(source_path.resolve(), start=target_parent.resolve())
