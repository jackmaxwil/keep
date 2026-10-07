from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from keep.quality.imatrix import ROUTED_PROJECTIONS, ProjectionImatrixEntry
from keep.quality.prompts import QualityPrompt, get_quality_prompts


@dataclass
class ImatrixAccumulator:
    layer: int
    projection: str
    expert: int
    importance_sum: np.ndarray
    affinity_weighted_importance: np.ndarray | None = None
    affinity_score_sum: float = 0.0
    route_count: int = 0
    total_route_count: int = 0
    prompt_ids: list[str] = field(default_factory=list)
    prompt_id_set: set[str] = field(default_factory=set)

    def add(self, entry: ProjectionImatrixEntry) -> None:
        if self.importance_sum.shape != entry.importance_sum.shape:
            raise ValueError(
                "imatrix input dim changed for "
                f"layer {self.layer} {self.projection} expert {self.expert}: "
                f"{self.importance_sum.shape} vs {entry.importance_sum.shape}"
            )
        self.importance_sum += entry.importance_sum.astype(np.float64, copy=False)
        if entry.affinity_weighted_importance is not None:
            if self.affinity_weighted_importance is None:
                self.affinity_weighted_importance = np.zeros(
                    entry.affinity_weighted_importance.shape,
                    dtype=np.float64,
                )
            if self.affinity_weighted_importance.shape != entry.affinity_weighted_importance.shape:
                raise ValueError(
                    "affinity imatrix input dim changed for "
                    f"layer {self.layer} {self.projection} expert {self.expert}: "
                    f"{self.affinity_weighted_importance.shape} vs {entry.affinity_weighted_importance.shape}"
                )
            self.affinity_weighted_importance += entry.affinity_weighted_importance.astype(
                np.float64,
                copy=False,
            )
            self.affinity_score_sum += float(entry.affinity_score_sum or 0.0)
        self.route_count += int(entry.route_count)
        self.total_route_count += int(entry.total_route_count)
        for prompt_id in entry.prompt_ids:
            prompt_text = str(prompt_id)
            if prompt_text not in self.prompt_id_set:
                self.prompt_id_set.add(prompt_text)
                self.prompt_ids.append(prompt_text)


def parse_layer_selection(value: str) -> tuple[int, ...]:
    layers: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid descending layer range {item!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    if not layers:
        raise argparse.ArgumentTypeError("expected at least one layer")
    if min(layers) < 0:
        raise argparse.ArgumentTypeError("layers must be non-negative")
    return tuple(sorted(layers))


def parse_projection_selection(value: str) -> tuple[str, ...]:
    projections = tuple(part.strip() for part in value.split(",") if part.strip())
    if not projections:
        raise argparse.ArgumentTypeError("expected at least one projection")
    unsupported = sorted(set(projections) - set(ROUTED_PROJECTIONS))
    if unsupported:
        raise argparse.ArgumentTypeError(f"unsupported projection(s): {unsupported}")
    if len(set(projections)) != len(projections):
        raise argparse.ArgumentTypeError("duplicate projections are not supported")
    return projections


def selected_prompts(
    *,
    prompt_set: str,
    prompt_ids: list[str] | None,
    max_prompts: int | None,
) -> list[QualityPrompt]:
    prompts = list(get_quality_prompts(prompt_set))
    if prompt_ids:
        wanted = set(prompt_ids)
        prompts = [prompt for prompt in prompts if prompt.prompt_id in wanted]
        missing = sorted(wanted - {prompt.prompt_id for prompt in prompts})
        if missing:
            raise ValueError(f"unknown prompt ids for {prompt_set}: {missing}")
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError("no prompts selected")
    return prompts


def merge_entries(
    accumulators: dict[tuple[int, str, int], ImatrixAccumulator],
    entries: Iterable[ProjectionImatrixEntry],
) -> None:
    for entry in entries:
        key = (entry.layer, entry.projection, entry.expert)
        accumulator = accumulators.get(key)
        if accumulator is None:
            accumulator = ImatrixAccumulator(
                layer=entry.layer,
                projection=entry.projection,
                expert=entry.expert,
                importance_sum=np.zeros(entry.importance_sum.shape, dtype=np.float64),
            )
            accumulators[key] = accumulator
        accumulator.add(entry)


def finalize_entries(
    accumulators: dict[tuple[int, str, int], ImatrixAccumulator],
) -> tuple[ProjectionImatrixEntry, ...]:
    projection_order = {projection: index for index, projection in enumerate(ROUTED_PROJECTIONS)}
    entries: list[ProjectionImatrixEntry] = []
    for key in sorted(accumulators, key=lambda item: (item[0], projection_order[item[1]], item[2])):
        accumulator = accumulators[key]
        importance_sum = accumulator.importance_sum
        route_count = int(accumulator.route_count)
        total_route_count = int(accumulator.total_route_count)
        input_dim = int(importance_sum.shape[0])
        mean_importance = (
            importance_sum / route_count
            if route_count
            else np.zeros(input_dim, dtype=np.float64)
        )
        routing_weighted_importance = (
            importance_sum / total_route_count
            if total_route_count
            else np.zeros(input_dim, dtype=np.float64)
        )
        entries.append(
            ProjectionImatrixEntry(
                layer=accumulator.layer,
                projection=accumulator.projection,
                expert=accumulator.expert,
                importance_sum=importance_sum.astype(np.float32),
                mean_importance=mean_importance.astype(np.float32),
                routing_weighted_importance=routing_weighted_importance.astype(np.float32),
                route_count=route_count,
                total_route_count=total_route_count,
                route_frequency=route_count / total_route_count if total_route_count else 0.0,
                prompt_ids=tuple(accumulator.prompt_ids),
                affinity_weighted_importance=(
                    accumulator.affinity_weighted_importance.astype(np.float32)
                    if accumulator.affinity_weighted_importance is not None
                    else None
                ),
                affinity_score_sum=(
                    float(accumulator.affinity_score_sum)
                    if accumulator.affinity_weighted_importance is not None
                    else None
                ),
            )
        )
    return tuple(entries)


__all__ = [
    "ImatrixAccumulator",
    "finalize_entries",
    "merge_entries",
    "parse_layer_selection",
    "parse_projection_selection",
    "selected_prompts",
]
