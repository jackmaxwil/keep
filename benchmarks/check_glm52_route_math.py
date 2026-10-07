#!/usr/bin/env python3
"""Capture, audit, and compare GLM-5.2 source/candidate router traces."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


def _load_api() -> Any:
    module_path = (
        Path(__file__).parents[1]
        / "src"
        / "mlx_vq"
        / "quality"
        / "glm52_route_diagnostics.py"
    )
    name = "_glm52_route_diagnostics_cli"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load route diagnostics module {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _json_object(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("authority JSON must contain an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish authenticated GLM-5.2 route traces from raw layer NPZ files "
            "and compare source against candidate without loading MLX."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture", help="publish one trace side")
    capture.add_argument("--side", choices=("source", "candidate"), required=True)
    capture.add_argument("--input-root", required=True)
    capture.add_argument("--trace-root", required=True)
    capture.add_argument("--authority-json")
    capture.add_argument("--schema-version", type=int, choices=(1, 2), default=1)
    capture.add_argument("--routed-scaling-factor", type=float, default=2.5)
    capture.add_argument("--expert-count", type=int, default=168)
    capture.add_argument(
        "--allow-nonfrozen-fixture",
        action="store_true",
        help="allow a synthetic layer/n_valid inventory; forbidden for release evidence",
    )

    audit = commands.add_parser("audit", help="strictly audit one trace root")
    audit.add_argument("--trace-root", required=True)
    audit.add_argument("--expected-side", choices=("source", "candidate"))

    compare = commands.add_parser("compare", help="compare authenticated trace roots")
    compare.add_argument("--source-trace-root", required=True)
    compare.add_argument("--candidate-trace-root", required=True)
    compare.add_argument("--source-authority-json", required=True)
    compare.add_argument("--candidate-authority-json", required=True)
    compare.add_argument("--family-policy-json")
    compare.add_argument("--evidence-json", required=True)
    compare.add_argument(
        "--allow-nonfrozen-fixture",
        action="store_true",
        help="allow synthetic traces; release-host comparisons require the frozen batch",
    )
    return parser


def _capture(args: argparse.Namespace, api: Any) -> int:
    layers = api.load_npz_trace_root(args.input_root)
    authority = _json_object(args.authority_json)
    authority["evidence_class"] = "fixture_only"
    if not args.allow_nonfrozen_fixture:
        indexes = tuple(layer.layer_index for layer in layers)
        n_valid = {int(layer.expert_ids.shape[0]) for layer in layers}
        if indexes != api.FROZEN_SPARSE_LAYERS or n_valid != {api.FROZEN_N_VALID}:
            raise ValueError(
                "release capture requires sparse layers 3..77 and n_valid=744; "
                "use --allow-nonfrozen-fixture only for synthetic tests"
            )
    manifest = api.write_route_trace_artifact(
        args.trace_root,
        layers,
        side=args.side,
        authority=authority,
        routed_scaling_factor=args.routed_scaling_factor,
        expert_count=args.expert_count,
        schema_version=args.schema_version,
    )
    api.audit_route_trace_artifact(args.trace_root, expected_side=args.side)
    print(
        json.dumps(
            {
                "status": "captured",
                "side": args.side,
                "trace_root": str(Path(args.trace_root).absolute()),
                "layer_count": manifest["layer_count"],
                "n_valid": manifest["n_valid"],
                "manifest_body_sha256": manifest["manifest_body_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _audit(args: argparse.Namespace, api: Any) -> int:
    audited = api.audit_route_trace_artifact(
        args.trace_root, expected_side=args.expected_side
    )
    print(
        json.dumps(
            {
                "status": "valid",
                "side": audited.manifest["side"],
                "layer_count": len(audited.layers),
                "n_valid": audited.manifest["n_valid"],
                "manifest_body_sha256": audited.manifest["manifest_body_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _compare(args: argparse.Namespace, api: Any) -> int:
    evidence = api.compare_route_trace_artifacts(
        args.source_trace_root,
        args.candidate_trace_root,
        policy_json=args.family_policy_json,
        require_frozen_batch=not args.allow_nonfrozen_fixture,
        expected_source_authority=_json_object(args.source_authority_json),
        expected_candidate_authority=_json_object(args.candidate_authority_json),
    )
    api.write_route_math_evidence(args.evidence_json, evidence)
    print(
        json.dumps(
            {
                "status": (
                    "diagnostics_pass"
                    if evidence["checks"]["diagnostics_pass"]
                    else "diagnostics_fail"
                ),
                "diagnostic_only": evidence["evaluation"]["diagnostic_only"],
                "release_gate_pass": evidence["checks"]["release_gate_pass"],
                "evidence_json": str(Path(args.evidence_json).absolute()),
                "evidence_body_sha256": evidence["evidence_body_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if evidence["checks"]["diagnostics_pass"] else 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        api = _load_api()
        if args.command == "capture":
            return _capture(args, api)
        if args.command == "audit":
            return _audit(args, api)
        return _compare(args, api)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
