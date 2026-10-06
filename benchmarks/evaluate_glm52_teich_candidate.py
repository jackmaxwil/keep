#!/usr/bin/env python3
"""Automatically evaluate, but never promote, the canonical Teich adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _dummy_forward_row(source, *, positions=None, targets=None):  # noqa: ANN001, ANN202
    from mlx_vq.quality.glm52_adapter_training import PreparedGLM52TeichTeacherRow

    selected_positions = tuple(source.positions if positions is None else positions)
    selected_targets = tuple(source.target_token_ids if targets is None else targets)
    count = len(selected_positions)
    return PreparedGLM52TeichTeacherRow(
        row_index=source.row_index,
        prompt_id=source.prompt_id,
        split="train",
        tuning_eligible=True,
        input_token_ids=tuple(source.input_token_ids),
        positions=selected_positions,
        target_token_ids=selected_targets,
        teacher_logits=None,
        teacher_manifest_body_sha256=source.teacher_manifest_body_sha256,
        teacher_shard_sha256=source.teacher_shard_sha256,
        non_release_waiver=True,
        topk_logit_ids=mx.zeros((count, 1), dtype=mx.int32),
        topk_logit_values=mx.zeros((count, 1), dtype=mx.float16),
        tail_mass=mx.zeros((count,), dtype=mx.float16),
    )


def _selected_hidden(model, row, *, layer: int):  # noqa: ANN001, ANN202
    from mlx_vq.quality.glm52_adapter_training import (
        selected_glm52_hidden_selected_layer,
    )

    hidden = selected_glm52_hidden_selected_layer(
        model,
        row,
        layer=layer,
        surrogate_projections=frozenset(("gate_proj", "up_proj", "down_proj")),
        output_chunk_size=256,
    )
    mx.eval(hidden)
    return hidden


def _evaluate_holdout(model, rows, *, layer: int, slice_size: int = 64):  # noqa: ANN001, ANN202
    from mlx_vq.quality.glm52_adapter_training import glm52_topk_kl_loss

    correct: dict[str, list[bool]] = {}
    kl_total = 0.0
    position_total = 0
    for source in rows:
        forward_row = _dummy_forward_row(source)
        hidden = _selected_hidden(model, forward_row, layer=layer)
        session_correct: list[bool] = []
        for start in range(0, len(source.positions), slice_size):
            end = min(start + slice_size, len(source.positions))
            teacher = replace(
                source.load_distillation_window(start, end),
                split="train",
                tuning_eligible=True,
            )
            logits = model.lm_head(hidden[start:end]).astype(mx.float32)
            loss = glm52_topk_kl_loss(
                logits,
                teacher.topk_logit_ids,
                teacher.topk_logit_values,
                teacher.tail_mass,
            )
            predictions = mx.argmax(logits, axis=-1)
            mx.eval(loss, predictions)
            session_correct.extend(
                bool(value)
                for value in (
                    np.asarray(predictions, dtype=np.int32)
                    == np.asarray(teacher.target_token_ids, dtype=np.int32)
                )
            )
            count = end - start
            kl_total += float(loss.item()) * count
            position_total += count
            del teacher, logits, loss, predictions
            mx.clear_cache()
        correct[source.prompt_id] = session_correct
        del hidden
        mx.clear_cache()
    return correct, kl_total / position_total


class _FrozenRow:
    def __init__(self, index: int, value: dict[str, object]) -> None:
        self.row_index = index
        self.prompt_id = str(value["prompt_id"])
        self.input_token_ids = tuple(int(token) for token in value["encoded_token_ids"])
        positions = value.get("positions")
        self.positions = tuple(
            range(len(self.input_token_ids) - 1)
            if not isinstance(positions, list)
            else (int(position) for position in positions)
        )
        targets = value.get("target_token_ids")
        self.target_token_ids = tuple(
            self.input_token_ids[position + 1] for position in self.positions
        ) if not isinstance(targets, list) else tuple(int(token) for token in targets)
        self.teacher_manifest_body_sha256 = "0" * 64
        self.teacher_shard_sha256 = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _frozen_rows(path: Path) -> list[_FrozenRow]:
    payload = json.loads(path.read_bytes())
    values = payload.get("prompt_rows") if isinstance(payload, dict) else None
    if not isinstance(values, list) or len(values) != 66:
        raise ValueError("frozen regression pack must contain exactly 66 prompt rows")
    return [_FrozenRow(index, value) for index, value in enumerate(values)]


def _capture_frozen_baseline(
    model,
    rows: list[_FrozenRow],
    root: Path,
    *,
    layer: int,
    top_k: int = 2048,
    slice_size: int = 64,
):  # noqa: ANN001, ANN202
    root.mkdir(parents=True, exist_ok=True)
    correct_total = 0
    position_total = 0
    inventory = []
    for source in rows:
        path = root / f"{source.row_index:03d}.npz"
        forward_row = _dummy_forward_row(source)
        hidden = _selected_hidden(model, forward_row, layer=layer)
        ids_parts, values_parts, tail_parts = [], [], []
        for start in range(0, len(source.positions), slice_size):
            end = min(start + slice_size, len(source.positions))
            logits = model.lm_head(hidden[start:end]).astype(mx.float32)
            logsumexp = mx.logsumexp(logits, axis=-1)
            ids = mx.argsort(-logits, axis=-1)[:, :top_k]
            values = mx.take_along_axis(logits, ids, axis=-1)
            tail = mx.maximum(
                1.0 - mx.sum(mx.exp(values - logsumexp[:, None]), axis=-1),
                0.0,
            )
            predictions = mx.argmax(logits, axis=-1)
            mx.eval(ids, values, tail, predictions)
            ids_parts.append(np.asarray(ids, dtype=np.int32))
            values_parts.append(np.asarray(values, dtype=np.float16))
            tail_parts.append(np.asarray(tail, dtype=np.float16))
            correct_total += int(
                np.sum(
                    np.asarray(predictions, dtype=np.int32)
                    == np.asarray(source.target_token_ids[start:end], dtype=np.int32)
                )
            )
            position_total += end - start
            del logits, logsumexp, ids, values, tail, predictions
            mx.clear_cache()
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            temporary,
            topk_logit_ids=np.concatenate(ids_parts),
            topk_logit_values=np.concatenate(values_parts),
            tail_mass=np.concatenate(tail_parts),
        )
        os.replace(temporary, path)
        inventory.append({"prompt_id": source.prompt_id, "file_sha256": _sha256_file(path)})
        del hidden
        mx.clear_cache()
    return correct_total / position_total, inventory


def _evaluate_frozen_candidate(
    model,
    rows: list[_FrozenRow],
    baseline_root: Path,
    *,
    layer: int,
    slice_size: int = 64,
):  # noqa: ANN001, ANN202
    from mlx_vq.quality.glm52_adapter_training import glm52_topk_kl_loss

    correct_total = 0
    kl_total = 0.0
    position_total = 0
    for source in rows:
        with np.load(baseline_root / f"{source.row_index:03d}.npz", allow_pickle=False) as cache:
            ids = cache["topk_logit_ids"]
            values = cache["topk_logit_values"]
            tail = cache["tail_mass"]
        forward_row = _dummy_forward_row(source)
        hidden = _selected_hidden(model, forward_row, layer=layer)
        for start in range(0, len(source.positions), slice_size):
            end = min(start + slice_size, len(source.positions))
            logits = model.lm_head(hidden[start:end]).astype(mx.float32)
            loss = glm52_topk_kl_loss(
                logits,
                mx.array(ids[start:end], dtype=mx.int32),
                mx.array(values[start:end], dtype=mx.float16),
                mx.array(tail[start:end], dtype=mx.float16),
            )
            predictions = mx.argmax(logits, axis=-1)
            mx.eval(loss, predictions)
            correct_total += int(
                np.sum(
                    np.asarray(predictions, dtype=np.int32)
                    == np.asarray(source.target_token_ids[start:end], dtype=np.int32)
                )
            )
            count = end - start
            kl_total += float(loss.item()) * count
            position_total += count
            del logits, loss, predictions
            mx.clear_cache()
        del hidden
        mx.clear_cache()
    return correct_total / position_total, kl_total / position_total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache-dir", type=Path, required=True)
    parser.add_argument("--prompt-pack", type=Path, required=True)
    parser.add_argument("--frozen-prompt-pack", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=77)
    args = parser.parse_args()
    os.environ.pop("GLM_MLX_WIRED_LIMIT_GB", None)
    os.environ.pop("GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB", None)

    from mlx_vq.quality.glm52_adapter_training import (
        ValidatedGLM52TrainingBaseline,
        bind_glm52_low_rank_adapters,
        load_authenticated_glm52_adapter,
        prepare_glm52_teich_teacher_rows,
        target_glm52_projection,
    )
    from mlx_vq.quality.glm52_teich_candidate_evaluation import evaluate_candidate_gate
    from mlx_vq.quality.glm52_teich_training_cache import MANIFEST_FILENAME
    from mlx_vq.quality.glm52_training_baseline import accepted_baseline_provider

    manifest_path = args.teacher_cache_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    holdout = prepare_glm52_teich_teacher_rows(
        teacher_cache_dir=args.teacher_cache_dir,
        prompt_pack_path=args.prompt_pack,
        frozen_prompt_pack_path=args.frozen_prompt_pack,
        split="holdout",
        expected_manifest_sha256=_sha256_file(manifest_path),
        allow_non_release_teacher_cache=True,
        lazy=True,
    )
    baseline = accepted_baseline_provider(args)
    if not isinstance(baseline, ValidatedGLM52TrainingBaseline):
        raise TypeError("accepted baseline provider returned the wrong authority type")
    baseline_correct, baseline_kl = _evaluate_holdout(
        baseline.model, holdout, layer=args.layer
    )
    frozen_rows = _frozen_rows(args.frozen_prompt_pack)
    frozen_cache = args.output.parent / "frozen-baseline-topk"
    frozen_baseline_top1, frozen_inventory = _capture_frozen_baseline(
        baseline.model, frozen_rows, frozen_cache, layer=args.layer
    )

    adapter_manifest = json.loads(
        (args.adapter_dir / "glm52-low-rank-adapter-manifest.json").read_bytes()
    )
    projection_map = {
        (args.layer, projection): target_glm52_projection(
            baseline.model, layer=args.layer, projection=projection
        )
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    loaded = load_authenticated_glm52_adapter(
        args.adapter_dir,
        expected_parent_candidate_identity_sha256=baseline.candidate_identity_sha256,
        expected_teacher_manifest_body_sha256=manifest["manifest_body_sha256"],
        expected_manifest_body_sha256=adapter_manifest["manifest_body_sha256"],
        expected_candidate_identity_sha256=adapter_manifest["candidate_identity_sha256"],
        expected_num_experts=next(iter(projection_map.values())).num_experts,
    )
    bind_glm52_low_rank_adapters(projection_map, loaded)
    candidate_correct, candidate_kl = _evaluate_holdout(
        baseline.model, holdout, layer=args.layer
    )
    frozen_candidate_top1, frozen_candidate_kl = _evaluate_frozen_candidate(
        baseline.model, frozen_rows, frozen_cache, layer=args.layer
    )
    gate = evaluate_candidate_gate(
        baseline_correct=baseline_correct,
        candidate_correct=candidate_correct,
        baseline_topk_kl=baseline_kl,
        candidate_topk_kl=candidate_kl,
        frozen_baseline_top1=frozen_baseline_top1,
        frozen_candidate_top1=frozen_candidate_top1,
        frozen_baseline_mean_kld=0.0,
        frozen_candidate_mean_kld=frozen_candidate_kl,
    )
    report = {
        "record_type": "glm52_teich_canonical_candidate_evaluation_v1",
        "baseline_identity_sha256": baseline.candidate_identity_sha256,
        "candidate_identity_sha256": loaded.candidate_identity_sha256,
        "adapter_layer": args.layer,
        "teacher_manifest_sha256": _sha256_file(manifest_path),
        "teacher_manifest_body_sha256": manifest["manifest_body_sha256"],
        "holdout_session_count": len(holdout),
        "holdout_position_count": sum(len(row.positions) for row in holdout),
        "frozen_prompt_count": len(frozen_rows),
        "frozen_kld_reference": "untouched_student_top2048_aggregated_tail",
        "frozen_baseline_inventory": frozen_inventory,
        "gate": asdict(gate),
        "automatic_acceptance": False,
        "automatic_promotion": False,
        "release_eligible": False,
    }
    _write_json_atomic(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
