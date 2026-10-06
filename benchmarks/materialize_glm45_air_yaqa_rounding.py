from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID, fetch_hf_config
from mlx_vq.convert.stream_convert import load_safetensors_index
from mlx_vq.quant.rht import apply_rht_np
from mlx_vq.quality.kronecker_hessian import (
    blockldlq_full_reassign_codes,
    blockldlq_hin_only_reassign_codes,
    load_hessian_file,
    relative_symlink_target,
    require_full_yaqa_hessian_manifest,
    stats_dict,
    write_json,
)
from mlx_vq.validate.glm45_air_hessian_probe import _full_codebook_table_from_layer, _sample_output_rows
from mlx_vq.validate.glm45_air_vq import _expert_weight_name, _load_switch_glu, _read_named_tensor


SUPPORTED_PROJECTIONS = {"gate_proj", "up_proj", "down_proj"}


def _load_config(*, model_id: str, revision: str, config_path: str | None) -> dict[str, Any]:
    if config_path is not None:
        return json.loads(Path(config_path).read_text(encoding="utf-8"))
    try:
        cached_config = hf_hub_download(
            model_id,
            "config.json",
            revision=revision,
            local_files_only=True,
        )
        return json.loads(Path(cached_config).read_text(encoding="utf-8"))
    except Exception:
        return fetch_hf_config(model_id, revision=revision)


def _resolve_source_dir(*, model_id: str, revision: str, source_dir: str | None) -> Path:
    if source_dir is not None:
        return Path(source_dir)
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=["model.safetensors.index.json"],
            local_files_only=True,
        )
    )


def _group_filename(layer: int, projection: str) -> str:
    return f"layer-{layer:05d}-{projection}.safetensors"


def _switch_prefix(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.switch_mlp.{projection}"


def _projection_layer(switch_glu, projection: str):
    if projection == "gate_proj":
        return switch_glu.gate_proj
    if projection == "up_proj":
        return switch_glu.up_proj
    if projection == "down_proj":
        return switch_glu.down_proj
    raise ValueError(f"unsupported projection {projection!r}")


def _link_seed_groups(*, seed_artifact_dir: Path, output_dir: Path, rewritten_filenames: set[str]) -> int:
    linked_count = 0
    for source_path in sorted(seed_artifact_dir.glob("layer-*.safetensors")):
        target_path = output_dir / source_path.name
        if source_path.name in rewritten_filenames:
            continue
        if target_path.exists() or target_path.is_symlink():
            linked_count += 1
            continue
        target_path.symlink_to(relative_symlink_target(source_path, target_path.parent))
        linked_count += 1
    return linked_count


def _load_safetensors_metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="np") as handle:
        return dict(handle.metadata() or {})


def _select_rows(output_dim: int, max_output_rows: int | None) -> np.ndarray:
    if max_output_rows is None:
        return np.arange(output_dim, dtype=np.int64)
    return _sample_output_rows(output_dim, max_output_rows)


def _row_selection_summary(rows: np.ndarray, *, output_dim: int) -> dict[str, Any]:
    values = np.asarray(rows, dtype=np.int64)
    payload: dict[str, Any] = {
        "count": int(values.size),
        "output_dim": int(output_dim),
        "mode": "all" if values.size == output_dim else "sampled",
    }
    if values.size:
        payload["first"] = int(values[0])
        payload["last"] = int(values[-1])
    if values.size <= 256:
        payload["rows"] = [int(value) for value in values]
    return payload


def _source_weight_for_artifact_basis(source_weight: np.ndarray, projection_layer) -> np.ndarray:
    values = np.asarray(source_weight, dtype=np.float32)
    signs = getattr(projection_layer, "rht_signs", None)
    if signs is None:
        return values
    return apply_rht_np(values, np.asarray(signs, dtype=np.int8)).astype(np.float32, copy=False)


def _read_manifest(hessian_dir: Path) -> dict[str, Any]:
    manifest_path = hessian_dir / "kronecker-hessian-manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _filtered_entries(
    manifest: dict[str, Any],
    *,
    layers: set[int] | None,
    projections: set[str] | None,
    experts: set[int] | None,
) -> list[dict[str, Any]]:
    entries = []
    for entry in manifest.get("entries", []):
        layer = int(entry["layer"])
        projection = str(entry["projection"])
        expert = int(entry["expert"])
        if layers is not None and layer not in layers:
            continue
        if projections is not None and projection not in projections:
            continue
        if experts is not None and expert not in experts:
            continue
        entries.append(entry)
    if not entries:
        raise ValueError("no hessian manifest entries selected")
    return entries


def _parse_csv_ints(value: str | None) -> set[int] | None:
    if value is None:
        return None
    parsed = {int(part.strip()) for part in value.split(",") if part.strip()}
    return parsed or None


def _parse_csv_strings(value: str | None) -> set[str] | None:
    if value is None:
        return None
    parsed = {part.strip() for part in value.split(",") if part.strip()}
    unsupported = sorted(parsed - SUPPORTED_PROJECTIONS)
    if unsupported:
        raise ValueError(f"unsupported projection(s): {unsupported}")
    return parsed or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize GLM-4.5-Air YAQA/BlockLDLQ rounding from full_yaqa "
            "or blockldlq_hin_only Hessian manifests, labeling candidates accordingly."
        )
    )
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--source-dir")
    parser.add_argument("--hessian-dir", required=True)
    parser.add_argument("--seed-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers")
    parser.add_argument("--projections")
    parser.add_argument("--experts")
    parser.add_argument("--max-output-rows", type=int)
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()

    if args.sweeps <= 0:
        raise SystemExit("--sweeps must be positive")
    source_dir = _resolve_source_dir(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
    )
    index_path = Path(args.index_path) if args.index_path is not None else source_dir / "model.safetensors.index.json"
    _load_config(model_id=args.model_id, revision=args.revision, config_path=args.config_path)
    index = load_safetensors_index(index_path)
    hessian_root = Path(args.hessian_dir)
    seed_root = Path(args.seed_artifact_dir)
    output_root = Path(args.output_dir)
    if output_root.exists() and not args.allow_existing:
        raise FileExistsError(f"{output_root} already exists")

    manifest = _read_manifest(hessian_root)
    selected_entries = _filtered_entries(
        manifest,
        layers=_parse_csv_ints(args.layers),
        projections=_parse_csv_strings(args.projections),
        experts=_parse_csv_ints(args.experts),
    )
    method_kind = str(manifest.get("method", {}).get("kind", ""))
    full_yaqa_audit: dict[str, Any] | None = None
    if method_kind == "full_yaqa":
        full_yaqa_audit = require_full_yaqa_hessian_manifest(
            hessian_root,
            manifest=manifest,
            entries=selected_entries,
        )
    elif method_kind != "blockldlq_hin_only":
        raise ValueError(
            "this materializer accepts only full_yaqa or blockldlq_hin_only manifests; "
            f"found {method_kind!r}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    rewritten_filenames = {
        _group_filename(int(entry["layer"]), str(entry["projection"]))
        for entry in selected_entries
    }
    linked_count = _link_seed_groups(
        seed_artifact_dir=seed_root,
        output_dir=output_root,
        rewritten_filenames=rewritten_filenames,
    )

    entries_by_group: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for entry in selected_entries:
        entries_by_group.setdefault((int(entry["layer"]), str(entry["projection"])), []).append(entry)

    projection_results: dict[str, Any] = {}
    rewritten_group_count = 0
    source_tensors_read = 0
    peak_source_tensor_bytes = 0

    for (layer, projection), entries in sorted(entries_by_group.items()):
        switch_glu = _load_switch_glu(seed_root, layer)
        projection_layer = _projection_layer(switch_glu, projection)
        rows = _select_rows(projection_layer.output_dims, args.max_output_rows)
        full_codebook = _full_codebook_table_from_layer(projection_layer)
        codes_name = f"{_switch_prefix(layer, projection)}.codes"
        scales_name = f"{_switch_prefix(layer, projection)}.scales"
        seed_path = seed_root / _group_filename(layer, projection)
        target_path = output_root / _group_filename(layer, projection)
        arrays = mx.load(str(seed_path))
        codes = np.asarray(arrays[codes_name]).copy()
        scales = np.asarray(arrays[scales_name], dtype=np.float32)
        expert_stats: dict[str, Any] = {}
        current_error = 0.0
        new_error = 0.0
        changed_code_count = 0
        code_count = 0

        for entry in entries:
            expert = int(entry["expert"])
            source_weight = _read_named_tensor(
                source_dir,
                index,
                _expert_weight_name(layer, expert, projection),
            ).astype(np.float32, copy=False)
            source_weight = _source_weight_for_artifact_basis(source_weight, projection_layer)
            source_tensors_read += 1
            peak_source_tensor_bytes = max(peak_source_tensor_bytes, int(source_weight.nbytes))
            hessian_arrays, _ = load_hessian_file(hessian_root / str(entry["path"]))
            if method_kind == "full_yaqa":
                reassigned = blockldlq_full_reassign_codes(
                    source_weight,
                    codes[expert],
                    scales[expert],
                    hessian_arrays["h_in"],
                    hessian_arrays["h_out"],
                    codebook=full_codebook,
                    group_size=projection_layer.group_size,
                    code_bits=projection_layer.code_bits,
                    row_indices=rows,
                    sweeps=args.sweeps,
                )
                current_error += reassigned.stats.current_kronecker_weighted_error
                new_error += reassigned.stats.kronecker_weighted_error
            else:
                reassigned = blockldlq_hin_only_reassign_codes(
                    source_weight,
                    codes[expert],
                    scales[expert],
                    hessian_arrays["h_in"],
                    codebook=full_codebook,
                    group_size=projection_layer.group_size,
                    code_bits=projection_layer.code_bits,
                    row_indices=rows,
                    sweeps=args.sweeps,
                )
                current_error += reassigned.stats.current_hin_weighted_error
                new_error += reassigned.stats.hin_weighted_error
            codes[expert, rows] = reassigned.codes
            stats = stats_dict(reassigned.stats)
            expert_stats[f"expert_{expert}"] = {
                **stats,
                "hessian_path": str(entry["path"]),
                "sample_count": int(entry.get("sample_count", 0)),
            }
            changed_code_count += reassigned.stats.changed_code_count
            code_count += reassigned.stats.code_count

        metadata = _load_safetensors_metadata(seed_path)
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        arrays[codes_name] = mx.array(codes)
        mx.save_safetensors(str(target_path), arrays, metadata=metadata)
        rewritten_group_count += 1
        projection_payload: dict[str, Any] = {
            "layer": layer,
            "projection": projection,
            "selected_experts": [int(entry["expert"]) for entry in entries],
            "output_row_selection": _row_selection_summary(rows, output_dim=projection_layer.output_dims),
            "changed_code_count": int(changed_code_count),
            "code_count": int(code_count),
            "changed_code_fraction": float(changed_code_count / code_count) if code_count else 0.0,
            "expert_stats": expert_stats,
        }
        if method_kind == "full_yaqa":
            projection_payload.update(
                {
                    "current_kronecker_weighted_error": current_error,
                    "kronecker_weighted_error": new_error,
                    "kronecker_weighted_error_ratio": new_error / current_error
                    if abs(current_error) > 1.0e-12
                    else 1.0,
                }
            )
        else:
            projection_payload.update(
                {
                    "current_hin_weighted_error": current_error,
                    "hin_weighted_error": new_error,
                    "hin_weighted_error_ratio": new_error / current_error
                    if abs(current_error) > 1.0e-12
                    else 1.0,
                }
            )
        projection_results[f"layer_{layer}.{projection}"] = projection_payload

    method_payload = (
        {
            "kind": "full_yaqa",
            "full_yaqa": True,
            "h_out_collected": True,
            "seed_artifact_mutated": False,
            "candidate_artifact_mutated": False,
            "frozen_scales": True,
            "notes": (
                "Full YAQA/BlockLDLQ candidate using H_in and H_out factors; "
                "candidate still requires authority-cache evaluation before acceptance."
            ),
        }
        if method_kind == "full_yaqa"
        else {
            "kind": "blockldlq_hin_only",
            "full_yaqa": False,
            "h_out_collected": False,
            "seed_artifact_mutated": False,
            "candidate_artifact_mutated": False,
            "frozen_scales": True,
            "notes": "Rung 4 single-host fallback using H_in only; do not call this full YAQA.",
        }
    )
    output_manifest = {
        "schema_version": 1,
        "record_type": "air_vq_yaqa_rounding_manifest",
        "method": method_payload,
        "model_id": args.model_id,
        "revision": args.revision,
        "source_dir": str(source_dir),
        "seed_artifact_dir": str(seed_root),
        "hessian_dir": str(hessian_root),
        "output_dir": str(output_root),
        "linked_group_count": linked_count,
        "rewritten_group_count": rewritten_group_count,
        "source_tensors_read": source_tensors_read,
        "peak_source_tensor_bytes": peak_source_tensor_bytes,
        "projection_results": projection_results,
        "hessian_manifest": manifest,
    }
    if full_yaqa_audit is not None:
        output_manifest["full_yaqa_hessian_audit"] = full_yaqa_audit
    write_json(output_root / "yaqa-rounding-manifest.json", output_manifest)
    print(json.dumps(output_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
