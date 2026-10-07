from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from keep.convert.glm52_reap import (
    audit_glm52_reap_source_index,
    audit_glm52_reap_source_payloads,
)
from keep.convert.stream_convert import load_safetensors_index
from ramp.models.profiles import load_profile


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


def _parse_group_keys(raw: str | None) -> tuple[tuple[int, str], ...] | None:
    if raw is None:
        return None
    keys: list[tuple[int, str]] = []
    for item in raw.split(","):
        try:
            layer_raw, projection = item.split(":", maxsplit=1)
            keys.append((int(layer_raw), projection))
        except ValueError as error:
            raise ValueError(
                "--groups entries must use '<layer>:<projection>'"
            ) from error
    return tuple(keys)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit local GLM-5.2 REAP NVFP4 shard headers before materialization."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--groups")
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    try:
        source_audit = audit_glm52_reap_source_index(
            config=_load_json_object(args.config_path),
            index=load_safetensors_index(args.index_path),
            model_id=args.model_id,
            revision=args.revision,
            profile=load_profile(args.profile_path),
            config_sha256=_sha256_file(args.config_path),
            index_sha256=_sha256_file(args.index_path),
        )
        payload = audit_glm52_reap_source_payloads(
            source_dir=args.source_dir,
            source_audit=source_audit,
            max_groups=args.max_groups,
            group_keys=_parse_group_keys(args.groups),
        )
    except Exception as error:
        payload = {
            "schema_version": 1,
            "record_type": "glm52_modelopt_nvfp4_source_payload_audit",
            "model_id": args.model_id,
            "revision": args.revision,
            "profile_path": args.profile_path,
            "config_path": args.config_path,
            "index_path": args.index_path,
            "source_dir": args.source_dir,
            "payload_status": "glm52_modelopt_nvfp4_source_audit_failed",
            "materialization_blocked": True,
            "materialization_blockers": ["input_load_or_validation_error"],
            "input_error": {"type": type(error).__name__, "message": str(error)},
            "missing_shards": [],
        }
    if args.output_json:
        _write_json(args.output_json, payload)
    if args.append_jsonl:
        _append_jsonl(args.append_jsonl, payload)
    if not args.output_json and not args.append_jsonl:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["materialization_blocked"]:
        details = payload["missing_shards"] or payload["materialization_blockers"]
        print(
            f"{payload['payload_status']}: " + ", ".join(str(value) for value in details),
            file=sys.stderr,
        )
        raise SystemExit(
            1
            if payload["payload_status"] == "glm52_modelopt_nvfp4_source_audit_failed"
            else 2
        )


if __name__ == "__main__":
    main()
