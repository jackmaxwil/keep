#!/usr/bin/env python3
"""Produce the strict GLM-5.2 FP32 source-teacher cache."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def _producer_api() -> Any:
    module_name = "keep.quality.glm52_teacher_cache_producer"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src/keep/quality/glm52_teacher_cache_producer.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the GLM52 teacher-cache producer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


producer_api = _producer_api()


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    input_path = Path(path)
    try:
        value = json.loads(input_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label} {input_path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} {input_path} must contain a JSON object")
    return value


def load_contract(
    prompt_pack_json: str | Path,
    artifact_identities_json: str | Path,
) -> Any:
    """Build the frozen contract from explicit artifact identity records."""

    identities = _load_json_object(
        artifact_identities_json,
        label="artifact identities",
    )
    contract_fields = identities.get("teacher_cache_contract", identities)
    if not isinstance(contract_fields, dict):
        raise ValueError("teacher_cache_contract must be an object")
    required = ("producer", "source_evidence", "non_vq_package")
    missing = [field for field in required if not isinstance(contract_fields.get(field), dict)]
    if missing:
        raise ValueError(
            "artifact identities must provide object fields: " + ", ".join(missing)
        )
    return producer_api.GLM52TeacherCacheContract.from_frozen_prompt_pack(
        prompt_pack_json,
        producer=contract_fields["producer"],
        source_evidence=contract_fields["source_evidence"],
        non_vq_package=contract_fields["non_vq_package"],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Produce the GLM-5.2 source-teacher cache under the global heavy-job "
            "and cache/run-specific locks."
        )
    )
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--prompt-pack-json", required=True)
    parser.add_argument("--non-vq-package-dir", required=True)
    parser.add_argument(
        "--artifact-identities-json",
        required=True,
        help=(
            "JSON containing producer, source_evidence, and non_vq_package "
            "contract identities"
        ),
    )
    parser.add_argument(
        "--profile-path",
        required=True,
        help="Pinned profile whose file SHA-256 must match the cache contract",
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument(
        "--route-trace-root",
        default=None,
        help=(
            "Optional sibling directory for authenticated source route traces. "
            "Capture occurs during the same teacher forward."
        ),
    )
    parser.add_argument("--ledger-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help=(
            "Bound prompt projection/publication to the canonical prefix. "
            "A limited run never publishes the completion manifest."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the read-only plan; acquire no lock and load no model",
    )
    return parser


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    contract_loader: Callable[[str | Path, str | Path], Any] = load_contract,
    producer: Callable[..., Any] = producer_api.produce_glm52_teacher_cache,
) -> int:
    args = _build_parser().parse_args(argv)
    contract = contract_loader(
        args.prompt_pack_json,
        args.artifact_identities_json,
    )
    if args.dry_run:
        plan = producer_api.plan_glm52_teacher_cache(
            contract=contract,
            cache_root=args.cache_root,
            ledger_path=args.ledger_path,
            checkpoint_dir=args.checkpoint_dir,
            limit_prompts=args.limit_prompts,
            route_trace_root=args.route_trace_root,
        )
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0

    source_blob_verification: dict[str, Any] | None = None

    def source_loader() -> Any:
        nonlocal source_blob_verification
        workload = producer_api.load_glm52_source_teacher_workload(
            snapshot_dir=args.snapshot_dir,
            non_vq_package_dir=args.non_vq_package_dir,
            profile_path=args.profile_path,
            contract=contract,
        )
        report = getattr(workload.source_blob_inventory, "report", None)
        if callable(report):
            source_blob_verification = report()
        return workload

    def non_vq_package_auditor() -> Any:
        return producer_api.audit_glm52_non_vq_package_for_teacher_cache(
            snapshot_dir=args.snapshot_dir,
            non_vq_package_dir=args.non_vq_package_dir,
            profile_path=args.profile_path,
            contract=contract,
        )

    # The public CLI has no unlocked execution mode.  The producer owns both
    # advisory lock contexts and calls this zero-argument loader only after
    # entering them.
    result = producer(
        contract=contract,
        cache_root=args.cache_root,
        ledger_path=args.ledger_path,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_identities=(
            producer_api.checkpoint_identity_values_for_contract(contract)
        ),
        limit_prompts=args.limit_prompts,
        route_trace_root=args.route_trace_root,
        non_vq_package_auditor=non_vq_package_auditor,
        source_loader=source_loader,
        source_runner=producer_api.run_glm52_source_teacher_to_sink,
    )
    manifest = None if result.manifest is None else result.manifest.to_dict()
    output = {
        "completed": result.completed,
        "manifest_path": (
            None if result.manifest_path is None else str(result.manifest_path)
        ),
        "resumed_prompt_ids": list(result.resumed_prompt_ids),
        "produced_prompt_ids": list(result.produced_prompt_ids),
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
        "source_blob_verification": source_blob_verification,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


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
