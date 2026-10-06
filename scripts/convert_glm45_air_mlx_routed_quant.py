from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID
from mlx_vq.convert.mlx_routed_quant import (
    convert_mlx_routed_quant_layer,
    write_mlx_routed_quant_manifest,
)
from mlx_vq.convert.stream_convert import load_safetensors_index


def _load_config(path: str | None, *, model_id: str, revision: str) -> dict:
    if path is not None:
        return json.loads(Path(path).read_text())
    config_path = hf_hub_download(model_id, "config.json", revision=revision, local_files_only=True)
    return json.loads(Path(config_path).read_text())


def _resolve_source_dir(source_dir: str | None, *, model_id: str, revision: str) -> Path:
    if source_dir is not None:
        return Path(source_dir)
    return Path(snapshot_download(repo_id=model_id, revision=revision, local_files_only=True))


def _parse_layers(entries: list[str] | None, *, first_sparse: int, num_layers: int) -> list[int]:
    if not entries:
        return list(range(first_sparse, num_layers))
    layers: set[int] = set()
    for entry in entries:
        if "-" in entry:
            start, end = entry.split("-", maxsplit=1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(entry))
    return sorted(layer for layer in layers if first_sparse <= layer < num_layers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GLM-4.5-Air routed experts to MLX affine QuantizedSwitchLinear artifacts."
    )
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir")
    parser.add_argument("--config-path")
    parser.add_argument("--index-path")
    parser.add_argument("--output-dir", default="artifacts/glm-4.5-air-mlx-q2-routed-g128")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--mode", default="affine")
    parser.add_argument("--layer", action="append", help="Layer number or inclusive range, e.g. 1 or 1-4.")
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config_path, model_id=args.model_id, revision=args.revision)
    source_dir = _resolve_source_dir(args.source_dir, model_id=args.model_id, revision=args.revision)
    index_path = Path(args.index_path) if args.index_path else source_dir / "model.safetensors.index.json"
    index = load_safetensors_index(index_path)
    first_sparse = int(config.get("first_k_dense_replace", 0))
    layers = _parse_layers(
        args.layer,
        first_sparse=first_sparse,
        num_layers=int(config["num_hidden_layers"]),
    )
    if args.max_layers is not None:
        layers = layers[: args.max_layers]

    plan = {
        "model_id": args.model_id,
        "source_dir": str(source_dir),
        "index_path": str(index_path),
        "output_dir": args.output_dir,
        "layers": layers,
        "layer_count": len(layers),
        "num_experts": int(config["n_routed_experts"]),
        "group_size": args.group_size,
        "bits": args.bits,
        "mode": args.mode,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    records = []
    for layer in layers:
        records.extend(
            convert_mlx_routed_quant_layer(
                source_dir=source_dir,
                index=index,
                output_dir=args.output_dir,
                layer=layer,
                num_experts=int(config["n_routed_experts"]),
                group_size=args.group_size,
                bits=args.bits,
                mode=args.mode,
                skip_existing=not args.overwrite,
            )
        )
    manifest = write_mlx_routed_quant_manifest(
        output_dir=args.output_dir,
        model_id=args.model_id,
        records=records,
        group_size=args.group_size,
        bits=args.bits,
        mode=args.mode,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
