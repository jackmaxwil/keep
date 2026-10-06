#!/usr/bin/env python3
"""Produce and compare the frozen GLM-5.2 full-vocabulary candidate cache."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4


def _candidate_api() -> Any:
    module_name = "mlx_vq.quality.glm52_candidate_eval"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/mlx_vq/quality/glm52_candidate_eval.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the GLM52 candidate-eval module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


candidate_api = _candidate_api()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    input_path = Path(path).expanduser().resolve(strict=False)
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} {input_path} must contain a JSON object")
    return payload


def _write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path).expanduser().resolve(strict=False)
    if output.suffix != ".json":
        raise ValueError("comparison output must use the .json suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.partial-{uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _add_produce_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "produce",
        help="produce the authenticated composite-candidate FP32 logit cache",
    )
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--source-index-path", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--tokenizer-readiness-json", required=True)
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--prompt-pack-json", required=True)
    parser.add_argument(
        "--teacher-cache-root",
        required=True,
        help="strict-audited source cache whose frozen identity envelope is reused",
    )
    parser.add_argument("--non-vq-artifact-dir", required=True)
    parser.add_argument("--non-vq-evidence-json", required=True)
    parser.add_argument("--routed-artifact-dir", required=True)
    parser.add_argument("--composite-audit-json", required=True)
    parser.add_argument("--materialization-runs-jsonl", required=True)
    parser.add_argument("--full-bind-preflight-json", required=True)
    parser.add_argument("--candidate-cache-root", required=True)
    parser.add_argument("--ledger-path", required=True)
    parser.add_argument("--route-trace-root")
    parser.add_argument("--recovery-dir")
    parser.add_argument("--expected-seed-manifest-sha256")
    parser.add_argument("--expected-stats-manifest-sha256")
    parser.add_argument("--expected-full-source-blob-inventory-sha256")
    parser.add_argument("--expected-routed-source-blob-inventory-sha256")
    parser.add_argument("--expected-recovery-lever")
    parser.add_argument("--expected-recovery-policy-json")
    parser.add_argument(
        "--expected-accepted-composite-audit-sha256",
        "--expected-composite-audit-sha256",
        dest="expected_accepted_composite_audit_sha256",
    )
    parser.add_argument(
        "--expected-recovery-candidate-identity-sha256",
        "--expected-recovery-mixed-artifact-identity-sha256",
        dest="expected_recovery_candidate_identity_sha256",
    )
    parser.add_argument("--expected-recovery-manifest-body-sha256")
    parser.add_argument(
        "--allow-non-release-teacher-cache",
        action="store_true",
        help=(
            "diagnostic only: permit a teacher/candidate diagnostic evidence chain "
            "while retaining strict payload audits"
        ),
    )


def _add_compare_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "compare",
        help="strict-audit both caches and evaluate the frozen family gate",
    )
    parser.add_argument("--teacher-cache-root", required=True)
    parser.add_argument("--candidate-cache-root", required=True)
    parser.add_argument("--prompt-pack-json", required=True)
    parser.add_argument("--family-policy-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--expected-recovery-candidate-identity-sha256")
    parser.add_argument("--expected-recovery-manifest-body-sha256")
    parser.add_argument("--expected-accepted-composite-audit-sha256")
    parser.add_argument(
        "--allow-non-release-teacher-cache",
        action="store_true",
        help=(
            "diagnostic only: permit a teacher/candidate diagnostic evidence chain "
            "while retaining strict payload audits"
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GLM-5.2 CANDIDATE full-vocabulary evaluation. The frozen prompt pack "
            "and policy are inputs; this CLI exposes no tuning controls."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_produce_parser(subparsers)
    _add_compare_parser(subparsers)
    return parser


def _produce_recovery_authority(args: argparse.Namespace) -> dict[str, Any] | None:
    authority = {
        "route_trace_root": args.route_trace_root,
        "recovery_dir": args.recovery_dir,
        "expected_seed_manifest_sha256": args.expected_seed_manifest_sha256,
        "expected_stats_manifest_sha256": args.expected_stats_manifest_sha256,
        "expected_full_source_blob_inventory_sha256": (
            args.expected_full_source_blob_inventory_sha256
        ),
        "expected_routed_source_blob_inventory_sha256": (
            args.expected_routed_source_blob_inventory_sha256
        ),
        "expected_recovery_lever": args.expected_recovery_lever,
        "expected_recovery_policy_json": args.expected_recovery_policy_json,
        "expected_accepted_composite_audit_sha256": (
            args.expected_accepted_composite_audit_sha256
        ),
        "expected_recovery_candidate_identity_sha256": (
            args.expected_recovery_candidate_identity_sha256
        ),
        "expected_recovery_manifest_body_sha256": (
            args.expected_recovery_manifest_body_sha256
        ),
    }
    supplied = [name for name, value in authority.items() if value is not None]
    if supplied and len(supplied) != len(authority):
        missing = [name for name, value in authority.items() if value is None]
        raise ValueError(
            "candidate route trace capture requires all recovery audit authorities; "
            "missing: " + ", ".join(missing)
        )
    if not supplied:
        return None
    authority["expected_recovery_policy"] = _load_json_object(
        authority.pop("expected_recovery_policy_json"),
        label="recovery policy",
    )
    return authority


def _comparison_recovery_authority(args: argparse.Namespace) -> dict[str, str | None]:
    authority = {
        "expected_recovery_candidate_identity_sha256": (
            args.expected_recovery_candidate_identity_sha256
        ),
        "expected_recovery_manifest_body_sha256": (
            args.expected_recovery_manifest_body_sha256
        ),
        "expected_accepted_composite_audit_sha256": (
            args.expected_accepted_composite_audit_sha256
        ),
    }
    supplied = [value for value in authority.values() if value is not None]
    if supplied and len(supplied) != len(authority):
        raise ValueError(
            "compare requires all three recovered comparison identities"
        )
    return authority


def _run_produce(args: argparse.Namespace) -> int:
    recovery_authority = _produce_recovery_authority(args)
    contract = candidate_api.build_glm52_candidate_contract_from_teacher_cache(
        args.teacher_cache_root,
        prompt_pack_path=args.prompt_pack_json,
    )
    if recovery_authority is not None:
        contract = candidate_api.build_glm52_candidate_cache_contract(
            contract,
            recovery_candidate_identity_sha256=(
                recovery_authority["expected_recovery_candidate_identity_sha256"]
            ),
            recovery_manifest_body_sha256=(
                recovery_authority["expected_recovery_manifest_body_sha256"]
            ),
            accepted_composite_audit_sha256=(
                recovery_authority["expected_accepted_composite_audit_sha256"]
            ),
        )
    result = candidate_api.produce_glm52_candidate_cache(
        contract=contract,
        profile_path=args.profile_path,
        config_path=args.config_path,
        source_index_path=args.source_index_path,
        tokenizer_dir=args.tokenizer_dir,
        tokenizer_readiness_json=args.tokenizer_readiness_json,
        family_policy_json=args.family_policy_json,
        non_vq_artifact_dir=args.non_vq_artifact_dir,
        non_vq_evidence_json=args.non_vq_evidence_json,
        routed_artifact_dir=args.routed_artifact_dir,
        composite_audit_json=args.composite_audit_json,
        materialization_runs_jsonl=args.materialization_runs_jsonl,
        full_bind_preflight_json=args.full_bind_preflight_json,
        teacher_cache_root=args.teacher_cache_root,
        prompt_pack_path=args.prompt_pack_json,
        cache_root=args.candidate_cache_root,
        ledger_path=args.ledger_path,
        allow_non_release_teacher_cache=args.allow_non_release_teacher_cache,
        route_trace_root=(
            None if recovery_authority is None else recovery_authority["route_trace_root"]
        ),
        recovery_dir=(
            None if recovery_authority is None else recovery_authority["recovery_dir"]
        ),
        expected_seed_manifest_sha256=(
            None
            if recovery_authority is None
            else recovery_authority["expected_seed_manifest_sha256"]
        ),
        expected_stats_manifest_sha256=(
            None
            if recovery_authority is None
            else recovery_authority["expected_stats_manifest_sha256"]
        ),
        expected_full_source_blob_inventory_sha256=(
            None
            if recovery_authority is None
            else recovery_authority["expected_full_source_blob_inventory_sha256"]
        ),
        expected_routed_source_blob_inventory_sha256=(
            None
            if recovery_authority is None
            else recovery_authority["expected_routed_source_blob_inventory_sha256"]
        ),
        expected_recovery_lever=(
            None
            if recovery_authority is None
            else recovery_authority["expected_recovery_lever"]
        ),
        expected_recovery_policy=(
            None
            if recovery_authority is None
            else recovery_authority["expected_recovery_policy"]
        ),
        expected_composite_audit_sha256=(
            None
            if recovery_authority is None
            else recovery_authority["expected_accepted_composite_audit_sha256"]
        ),
        expected_recovery_mixed_artifact_identity_sha256=(
            None
            if recovery_authority is None
            else recovery_authority["expected_recovery_candidate_identity_sha256"]
        ),
        expected_recovery_manifest_body_sha256=(
            None
            if recovery_authority is None
            else recovery_authority["expected_recovery_manifest_body_sha256"]
        ),
    )
    manifest = None if result.manifest is None else result.manifest.to_dict()
    payload = {
        "completed": result.completed,
        "candidate_kind": (
            candidate_api.GLM52_CANDIDATE_KIND
            if recovery_authority is None
            else candidate_api.GLM52_RECOVERED_CANDIDATE_KIND
        ),
        "candidate_artifact_identity_sha256": contract.bound_identity_sha256,
        "recovery_candidate_identity_sha256": (
            None
            if recovery_authority is None
            else recovery_authority["expected_recovery_candidate_identity_sha256"]
        ),
        "recovery_manifest_body_sha256": (
            None
            if recovery_authority is None
            else recovery_authority["expected_recovery_manifest_body_sha256"]
        ),
        "accepted_composite_audit_sha256": (
            None
            if recovery_authority is None
            else recovery_authority["expected_accepted_composite_audit_sha256"]
        ),
        "manifest_path": (
            None if result.manifest_path is None else str(result.manifest_path)
        ),
        "produced_prompt_ids": list(result.produced_prompt_ids),
        "resumed_prompt_ids": list(result.resumed_prompt_ids),
        "shard_count": len(result.shards),
        "payload_integrity_pass": (
            None if manifest is None else manifest["payload_integrity_pass"]
        ),
        "all_producer_memory_counters_known": (
            None
            if manifest is None
            else manifest["all_producer_memory_counters_known"]
        ),
        "all_producer_memory_clean": (
            None if manifest is None else manifest["all_producer_memory_clean"]
        ),
        "system_wired_default": (
            None if manifest is None else manifest["system_wired_default"]
        ),
        "release_eligible": (
            None if manifest is None else manifest["release_eligible"]
        ),
    }
    if manifest is not None and manifest.get("evidence_class") == "diagnostic_only":
        payload.update(
            {
                "evidence_class": manifest["evidence_class"],
                "reason": manifest["reason"],
            }
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["completed"] and (
        payload["release_eligible"]
        or payload.get("evidence_class") == "diagnostic_only"
    ) else 2


def _run_compare(args: argparse.Namespace) -> int:
    recovery_authority = _comparison_recovery_authority(args)
    report = candidate_api.compare_glm52_candidate_caches(
        args.teacher_cache_root,
        args.candidate_cache_root,
        policy_path=args.family_policy_json,
        prompt_pack_path=args.prompt_pack_json,
        allow_non_release_teacher_cache=args.allow_non_release_teacher_cache,
        **recovery_authority,
    )
    _write_json_atomic(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    comparison_completed = report["family_eval_gate_pass"] or (
        report.get("evidence_class") == "diagnostic_only"
        and all(report["checks"].values())
    )
    return 0 if comparison_completed else 2


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "produce":
        return _run_produce(args)
    if args.command == "compare":
        return _run_compare(args)
    raise AssertionError(f"unhandled command {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_cli(argv)
    except Exception as error:
        print(
            json.dumps(
                {
                    "completed": False,
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
