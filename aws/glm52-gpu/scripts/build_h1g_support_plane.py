#!/usr/bin/env python3
"""Build deterministic H.1g support artifacts from authenticated exact inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.canonical import canonical_json_bytes  # noqa: E402
from glm52_enforcement.support_plane import (  # noqa: E402
    build_pre_support_retained_runtime_fragment,
    build_retained_fence_bootstrap_fragment,
    build_support_contract_artifacts,
    build_support_precreate_plane,
    pre_support_runtime_inputs_from_mapping,
    retained_fence_bootstrap_inputs_from_mapping,
    support_build_inputs_from_mapping,
    support_price_card_from_mapping,
)


def _read_canonical(path: Path) -> Mapping[str, object]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError(f"{path}: canonical JSON must end in exactly one LF")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid ASCII JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{path}: input must be one exact JSON object")
    try:
        expected = canonical_json_bytes(value) + b"\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: noncanonical JSON value") from exc
    if expected != raw:
        raise ValueError(f"{path}: JSON bytes are not canonical")
    return value


def _artifact(file_name: str, value: object) -> tuple[str, bytes, dict[str, object]]:
    body = canonical_json_bytes(value) + b"\n"
    return (
        file_name,
        body,
        {
            "file_name": file_name,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "canonical_body_sha256": hashlib.sha256(body[:-1]).hexdigest(),
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--price-card", type=Path)
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output-dir", type=Path)
    outputs.add_argument("--contract-output-dir", type=Path)
    outputs.add_argument("--retained-runtime-output-dir", type=Path)
    outputs.add_argument("--retained-bootstrap-output-dir", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.contract_output_dir is not None:
        if args.inputs is not None or args.price_card is not None:
            raise ValueError("contract generation accepts no live input path")
        artifacts = build_support_contract_artifacts()
        paths = {
            file_name: args.contract_output_dir / file_name for file_name in artifacts
        }
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite contract artifacts: {existing}"
            )
        args.contract_output_dir.mkdir(parents=True, exist_ok=True)
        for file_name, projection in artifacts.items():
            paths[file_name].write_bytes(canonical_json_bytes(projection) + b"\n")
        return 0
    if args.retained_bootstrap_output_dir is not None:
        if args.inputs is None or args.price_card is not None:
            raise ValueError("retained bootstrap build requires only --inputs")
        if args.retained_bootstrap_output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite {args.retained_bootstrap_output_dir}"
            )
        inputs = retained_fence_bootstrap_inputs_from_mapping(
            _read_canonical(args.inputs)
        )
        fragment = build_retained_fence_bootstrap_fragment(inputs)
        artifact = _artifact(
            "retained-fence-bootstrap-v9.json",
            fragment,
        )
        args.retained_bootstrap_output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )
        (args.retained_bootstrap_output_dir / artifact[0]).write_bytes(artifact[1])
        return 0
    if args.retained_runtime_output_dir is not None:
        if args.inputs is None or args.price_card is not None:
            raise ValueError("retained runtime build requires only --inputs")
        if args.retained_runtime_output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite {args.retained_runtime_output_dir}"
            )
        inputs = pre_support_runtime_inputs_from_mapping(_read_canonical(args.inputs))
        retained_runtime = build_pre_support_retained_runtime_fragment(inputs)
        artifact = _artifact(
            "support-retained-runtime-v1.json",
            retained_runtime,
        )
        args.retained_runtime_output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )
        (args.retained_runtime_output_dir / artifact[0]).write_bytes(artifact[1])
        return 0
    if args.inputs is None or args.price_card is None:
        raise ValueError("bound build requires --inputs and --price-card")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    inputs = support_build_inputs_from_mapping(_read_canonical(args.inputs))
    price_card = support_price_card_from_mapping(
        _read_canonical(args.price_card),
        inputs=inputs,
    )
    bundle = build_support_precreate_plane(
        inputs=inputs,
        price_card=price_card,
    )
    artifacts = [
        _artifact("support-plane-v1.json", bundle.support_template),
        _artifact(
            "support-spend-descriptor-v1.json",
            bundle.spend_descriptor,
        ),
    ]
    manifest = deepcopy(bundle.manifest)
    manifest.pop("canonical_body_sha256")
    manifest["artifacts"] = [item[2] for item in artifacts]
    manifest["canonical_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    manifest_artifact = _artifact("support-manifest-v1.json", manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for file_name, body, _identity in artifacts + [manifest_artifact]:
        (args.output_dir / file_name).write_bytes(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
