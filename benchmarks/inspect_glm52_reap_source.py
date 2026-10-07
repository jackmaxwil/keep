from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from mlx_vq.convert.glm52_reap import audit_glm52_reap_source_index
from mlx_vq.convert.stream_convert import load_safetensors_index
from mlx_vq.models.profiles import load_profile


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the pinned GLM-5.2 REAP ModelOpt NVFP4 source index."
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    try:
        audit = audit_glm52_reap_source_index(
            config=_load_json_object(args.config_path),
            index=load_safetensors_index(args.index_path),
            model_id=args.model_id,
            revision=args.revision,
            profile=load_profile(args.profile_path),
            config_sha256=_sha256_file(args.config_path),
            index_sha256=_sha256_file(args.index_path),
        )
        payload = audit.to_json_dict()
    except Exception as error:
        audit = None
        payload = {
            "schema_version": 1,
            "record_type": "glm52_modelopt_nvfp4_source_audit",
            "model_id": args.model_id,
            "revision": args.revision,
            "profile_path": args.profile_path,
            "config_path": args.config_path,
            "index_path": args.index_path,
            "source_checks_pass": False,
            "audit_status": "glm52_modelopt_nvfp4_source_incompatible",
            "audit_blockers": ["input_load_or_validation_error"],
            "input_error": {"type": type(error).__name__, "message": str(error)},
        }
    if args.output_json:
        _write_json(args.output_json, payload)
    if args.append_jsonl:
        _append_jsonl(args.append_jsonl, payload)
    if not args.output_json and not args.append_jsonl:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if audit is None or not audit.source_checks_pass:
        print(
            f"{payload['audit_status']}: " + ", ".join(payload["audit_blockers"]),
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
