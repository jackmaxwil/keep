from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from ramp.models.profiles import load_profile
from keep.validate.glm52_runtime import preflight_glm52_full_bind


def _write_json(path: str | Path, payload: dict[str, object]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _failure_payload(
    args: argparse.Namespace,
    error: Exception,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_full_bind_preflight",
        "preflight_status": "glm52_full_bind_preflight_failed",
        "preflight_pass": False,
        "blockers": [str(error)],
        "profile_path": str(Path(args.profile_path).expanduser()),
        "config_path": str(Path(args.config_path).expanduser()),
        "source_index_path": str(Path(args.source_index_path).expanduser()),
        "non_vq_artifact_dir": str(
            Path(args.non_vq_artifact_dir).expanduser()
        ),
        "routed_artifact_dir": str(Path(args.routed_artifact_dir).expanduser()),
        "model_id": args.model_id,
        "source_revision": args.revision,
        "header_only": True,
        "tensor_payloads_read": False,
        "payload_hashes_verified": False,
        "full_model_constructed": False,
        "dense_routed_experts": False,
        "production_binding_proven": False,
        "production_generation_proven": False,
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validated_output_path(args: argparse.Namespace) -> Path:
    output = Path(args.output_json).expanduser().resolve(strict=False)
    if output.suffix != ".json":
        raise ValueError("output JSON path must use the .json suffix")

    artifact_roots = tuple(
        Path(raw).expanduser().resolve()
        for raw in (args.non_vq_artifact_dir, args.routed_artifact_dir)
    )
    for root in artifact_roots:
        if _is_within(output, root):
            raise ValueError(f"output JSON must be outside artifact root {root}")

    protected = [
        Path(args.profile_path).expanduser().resolve(),
        Path(args.config_path).expanduser().resolve(),
        Path(args.source_index_path).expanduser().resolve(),
    ]
    for root in artifact_roots:
        if root.is_dir():
            protected.extend(path for path in root.rglob("*") if path.is_file())
    if output.exists():
        for path in protected:
            try:
                aliases_input = output.samefile(path)
            except OSError:
                continue
            if aliases_input:
                raise ValueError(f"output JSON aliases protected input {path}")
    elif output in protected:
        raise ValueError(f"output JSON aliases protected input {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate GLM-5.2 full-bind lineage and artifact headers without "
            "reading tensor payloads or constructing the model."
        )
    )
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--source-index-path", required=True)
    parser.add_argument("--non-vq-artifact-dir", required=True)
    parser.add_argument("--routed-artifact-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help=(
            "Return zero for a valid blocked preflight after writing evidence; "
            "input/validation failures remain nonzero."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = _validated_output_path(args)
    except ValueError as error:
        print(f"invalid output JSON path: {error}", file=sys.stderr)
        return 2
    try:
        report = preflight_glm52_full_bind(
            profile=load_profile(args.profile_path),
            profile_path=args.profile_path,
            config_path=args.config_path,
            source_index_path=args.source_index_path,
            non_vq_artifact_dir=args.non_vq_artifact_dir,
            routed_artifact_dir=args.routed_artifact_dir,
            model_id=args.model_id,
            revision=args.revision,
        )
        payload = report.to_dict()
        return_code = 0 if report.preflight_pass or args.evidence_only else 2
    except Exception as error:
        payload = _failure_payload(args, error)
        return_code = 2
    output = _write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
