#!/usr/bin/env python3
"""Build the H.1g retain/import migration from archived readback evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from glm52_enforcement.cloudformation_stacks import (  # noqa: E402
    BootstrapCoordinate,
    MigrationEvidence,
    MigrationExecutionAuthority,
    StackIdentity,
    StackKind,
    TemplateCoordinate,
    build_stack_migration,
    canonical_json_bytes,
    initial_migration_execution_state,
    migration_bootstrap_result_from_state,
    migration_execution_state_from_projection,
    migration_execution_state_projection,
)


def _read_canonical(path: Path) -> object:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError(f"{path}: canonical JSON must end in exactly one LF")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid canonical JSON") from exc
    if canonical_json_bytes(value) + b"\n" != raw:
        raise ValueError(f"{path}: JSON is not canonical")
    return value


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be one exact JSON object")
    return value


def _stack_identity(value: object) -> StackIdentity:
    item = _mapping("stack identity", value)
    if set(item) != {
        "kind",
        "name",
        "stack_id",
        "account_id",
        "region",
        "termination_protection",
        "tags",
    }:
        raise ValueError("stack identity fields are not exact")
    tags = _mapping("stack tags", item["tags"])
    return StackIdentity(
        kind=StackKind(item["kind"]),
        name=item["name"],
        stack_id=item["stack_id"],
        account_id=item["account_id"],
        region=item["region"],
        termination_protection=item["termination_protection"],
        tags=tuple(sorted(tags.items())),
    )


def _evidence(value: object) -> MigrationEvidence:
    item = _mapping("live evidence", value)
    if set(item) != {
        "retained",
        "fence",
        "support",
        "current_policy_logical_id",
        "current_policy_physical_id",
        "current_policy_stack_id",
        "bucket_name",
        "import_identifier",
        "live_policy",
        "direct_policy_readbacks",
        "retained_deployment_role_id",
        "retained_resource_physical_ids",
        "retained_export_names",
        "support_export_names",
    }:
        raise ValueError("live evidence fields are not exact")
    import_identifier = _mapping(
        "import identifier", item["import_identifier"]
    )
    readbacks = item["direct_policy_readbacks"]
    if type(readbacks) is not list:
        raise ValueError("direct policy readbacks must be an exact JSON array")
    for field in (
        "retained_resource_physical_ids",
        "retained_export_names",
        "support_export_names",
    ):
        values = item[field]
        if (
            type(values) is not list
            or any(type(value) is not str or not value for value in values)
        ):
            raise ValueError(f"{field} must be an exact JSON string array")
    return MigrationEvidence(
        retained=_stack_identity(item["retained"]),
        fence=_stack_identity(item["fence"]),
        support=_stack_identity(item["support"]),
        current_policy_logical_id=item["current_policy_logical_id"],
        current_policy_physical_id=item["current_policy_physical_id"],
        current_policy_stack_id=item["current_policy_stack_id"],
        bucket_name=item["bucket_name"],
        import_identifier=tuple(sorted(import_identifier.items())),
        live_policy=_mapping("live policy", item["live_policy"]),
        direct_policy_readbacks=tuple(
            _mapping("direct policy readback", readback)
            for readback in readbacks
        ),
        retained_deployment_role_id=item["retained_deployment_role_id"],
        retained_resource_physical_ids=tuple(
            item["retained_resource_physical_ids"]
        ),
        retained_export_names=tuple(item["retained_export_names"]),
        support_export_names=tuple(item["support_export_names"]),
    )


def _coordinates(value: object) -> tuple[TemplateCoordinate, ...]:
    if type(value) is not list:
        raise ValueError("template coordinates must be one exact JSON array")
    result = []
    for item in value:
        coordinate = _mapping("template coordinate", item)
        if set(coordinate) != {
            "stage",
            "stack_id",
            "template_url",
            "version_id",
        }:
            raise ValueError("template coordinate fields are not exact")
        result.append(TemplateCoordinate(**coordinate))
    return tuple(result)


def _bootstrap_coordinate(value: object) -> BootstrapCoordinate:
    item = _mapping("bootstrap coordinate", value)
    if set(item) != {"kind", "template_url", "version_id"}:
        raise ValueError("bootstrap coordinate fields are not exact")
    return BootstrapCoordinate(
        kind=StackKind(item["kind"]),
        template_url=item["template_url"],
        version_id=item["version_id"],
    )


def _execution_authority(value: object) -> MigrationExecutionAuthority:
    item = _mapping("execution authority", value)
    if set(item) != {
        "deployment_role_arn",
        "deployment_role_id",
        "fence_bootstrap",
        "support_bootstrap",
        "import_change_set_name",
        "action_identity_sha256",
    }:
        raise ValueError("execution authority fields are not exact")
    return MigrationExecutionAuthority(
        deployment_role_arn=item["deployment_role_arn"],
        deployment_role_id=item["deployment_role_id"],
        fence_bootstrap=_bootstrap_coordinate(item["fence_bootstrap"]),
        support_bootstrap=_bootstrap_coordinate(item["support_bootstrap"]),
        import_change_set_name=item["import_change_set_name"],
        action_identity_sha256=item["action_identity_sha256"],
    )


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    phases = parser.add_subparsers(dest="phase", required=True)
    initialize = phases.add_parser("initialize-bootstrap")
    initialize.add_argument(
        "--execution-authority", type=Path, required=True
    )
    initialize.add_argument("--output-state", type=Path, required=True)
    build = phases.add_parser("build-migration")
    build.add_argument("--archived-template", type=Path, required=True)
    build.add_argument("--bootstrap-template", type=Path, required=True)
    build.add_argument("--live-evidence", type=Path, required=True)
    build.add_argument("--template-coordinates", type=Path, required=True)
    build.add_argument("--execution-authority", type=Path, required=True)
    build.add_argument("--execution-state", type=Path, required=True)
    build.add_argument("--support-inventory", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    execution_authority = _execution_authority(
        _read_canonical(args.execution_authority)
    )
    if args.phase == "initialize-bootstrap":
        _write_canonical(
            args.output_state,
            migration_execution_state_projection(
                initial_migration_execution_state(execution_authority)
            ),
        )
        return 0
    execution_state = migration_execution_state_from_projection(
        _read_canonical(args.execution_state)
    )
    if (
        execution_state.action_identity_sha256
        != execution_authority.action_identity_sha256
        or execution_state.deployment_role_arn
        != execution_authority.deployment_role_arn
        or execution_state.deployment_role_id
        != execution_authority.deployment_role_id
        or execution_state.fence_bootstrap
        != execution_authority.fence_bootstrap
        or execution_state.support_bootstrap
        != execution_authority.support_bootstrap
        or execution_state.import_change_set_name
        != execution_authority.import_change_set_name
    ):
        raise ValueError("execution state belongs to another authority")
    bootstrap_result = migration_bootstrap_result_from_state(execution_state)
    archived_template = _mapping(
        "archived template", _read_canonical(args.archived_template)
    )
    bootstrap_template = _mapping(
        "bootstrap template", _read_canonical(args.bootstrap_template)
    )
    live_evidence = _evidence(_read_canonical(args.live_evidence))
    coordinates = _coordinates(_read_canonical(args.template_coordinates))
    support_inventory = _mapping(
        "support inventory", _read_canonical(args.support_inventory)
    )
    bundle = build_stack_migration(
        archived_template=archived_template,
        bootstrap_template=bootstrap_template,
        bootstrap_result=bootstrap_result,
        evidence=live_evidence,
        template_coordinates=coordinates,
        reviewed_support_inventory=support_inventory,
        reviewed_support_inventory_sha256=hashlib.sha256(
            canonical_json_bytes(support_inventory)
        ).hexdigest(),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for artifact in bundle.artifacts:
        (args.output_dir / f"{artifact.stage}.template.json").write_bytes(
            artifact.body
        )
    _write_canonical(
        args.output_dir / "migration-manifest.json",
        bundle.manifest,
    )
    _write_canonical(
        args.output_dir / "migration-execution-state.json",
        migration_execution_state_projection(
            execution_state
        ),
    )
    _write_canonical(
        args.output_dir / "migration-bootstrap-result.json",
        {
            "schema_version": 1,
            "record_type": "glm52_h1g_migration_bootstrap_result_v1",
            "action_identity_sha256": (
                bootstrap_result.action_identity_sha256
            ),
            "state_revision": bootstrap_result.state_revision,
            "fence_stack_id": bootstrap_result.fence.stack_id,
            "support_stack_id": bootstrap_result.support.stack_id,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
