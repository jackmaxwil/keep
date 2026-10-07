from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from keep.convert.glm52_reap import (
    GLM52_REAP_EXPECTED_GROUPS,
    GLM52_REAP_EXPECTED_LAYER_IDS,
    GLM52_REAP_PROJECTIONS,
    GLM52_REAP_SOURCE_DECODER,
    audit_glm52_reap_source_index,
    audit_glm52_reap_source_payloads,
)
from keep.convert.stream_convert import (
    SourceWeightEncoding,
    VQExpertGroup,
    convert_vq_groups_from_safetensors,
    load_safetensors_index,
    plan_streaming_conversion_from_index,
    vq_group_output_filename,
)
from keep.io.schema import codebook_metadata_for_bits
from ramp.models.profiles import load_profile


CANONICAL_MANIFEST_NAME = "conversion-manifest.json"
PREVIOUS_MANIFEST_NAME = f"{CANONICAL_MANIFEST_NAME}.resume-previous.json"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json_atomic(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial-{uuid4().hex}")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)


def _append_jsonl(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True) + "\n"
    with output.open("a") as handle:
        handle.write(encoded)


def _emit(
    payload: dict[str, object],
    *,
    output_json: str | None,
    append_jsonl: str | None,
) -> None:
    if output_json is not None:
        _write_json_atomic(output_json, payload)
    if append_jsonl is not None:
        _append_jsonl(append_jsonl, payload)
    if output_json is None and append_jsonl is None:
        print(json.dumps(payload, indent=2, sort_keys=True))


def _parse_group_keys(raw: str | None) -> tuple[tuple[int, str], ...] | None:
    if raw is None:
        return None
    keys: list[tuple[int, str]] = []
    for item in raw.split(","):
        try:
            layer_raw, projection = item.split(":", maxsplit=1)
            layer = int(layer_raw)
        except ValueError as error:
            raise ValueError(
                "--groups entries must use '<layer>:<projection>'"
            ) from error
        if layer not in GLM52_REAP_EXPECTED_LAYER_IDS:
            raise ValueError(
                f"GLM52 materialization layer must be in 3..77, found {layer}"
            )
        if projection not in GLM52_REAP_PROJECTIONS:
            raise ValueError(
                f"GLM52 materialization projection must be one of "
                f"{GLM52_REAP_PROJECTIONS}, found {projection!r}"
            )
        keys.append((layer, projection))
    if not keys:
        raise ValueError("--groups must select at least one group")
    if len(set(keys)) != len(keys):
        raise ValueError("--groups contains duplicate layer/projection groups")
    return tuple(keys)


def _select_groups(
    groups: tuple[VQExpertGroup, ...],
    *,
    group_keys: tuple[tuple[int, str], ...] | None,
    max_groups: int | None,
    all_groups: bool,
) -> tuple[VQExpertGroup, ...]:
    selection_modes = sum(
        (group_keys is not None, max_groups is not None, all_groups)
    )
    if selection_modes != 1:
        raise ValueError(
            "select exactly one of --groups, --max-groups, or --all-groups"
        )
    if all_groups:
        return groups
    if max_groups is not None:
        if max_groups <= 0:
            raise ValueError("--max-groups must be positive")
        if max_groups > len(groups):
            raise ValueError(
                f"--max-groups cannot exceed the {len(groups)} planned groups"
            )
        return groups[:max_groups]
    assert group_keys is not None
    available = {(group.layer, group.projection): group for group in groups}
    unknown = tuple(key for key in group_keys if key not in available)
    if unknown:
        raise ValueError(f"requested groups are not in the pinned plan: {unknown}")
    requested = set(group_keys)
    return tuple(
        group for group in groups if (group.layer, group.projection) in requested
    )


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _load_prior_canonical_manifest(output_dir: str | Path) -> dict[str, object] | None:
    root = Path(output_dir)
    for name in (PREVIOUS_MANIFEST_NAME, CANONICAL_MANIFEST_NAME):
        path = root / name
        if not path.is_file():
            continue
        payload = _load_json_object(path)
        if payload.get("record_type") == (
            "glm52_modelopt_nvfp4_materialization_manifest"
        ):
            return payload
    return None


def _validate_prior_artifact_identity(
    *,
    output_dir: str | Path,
    prior: dict[str, object] | None,
    selected: tuple[VQExpertGroup, ...],
    profile: str,
    model_id: str,
    revision: str,
    config_sha256: str,
    index_sha256: str,
    code_bits: int,
    group_size: int,
    scale_estimator: str,
) -> dict[tuple[int, str], str]:
    root = Path(output_dir)
    expected_names = {vq_group_output_filename(group) for group in selected}
    unexpected_files = sorted(
        path.name
        for path in root.glob("layer-*.safetensors")
        if path.name not in expected_names
    )
    if unexpected_files:
        raise ValueError(
            "output directory contains group artifacts outside the selected plan: "
            f"{unexpected_files[:10]}"
        )
    if prior is None:
        return {}

    expected_top_level = {
        "profile": profile,
        "model_id": model_id,
        "source_revision": revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "source_weight_encoding": "modelopt_nvfp4",
        "source_decoder": GLM52_REAP_SOURCE_DECODER,
        "code_bits": code_bits,
        "group_size": group_size,
        "scale_estimator": scale_estimator,
        "selected_group_keys": [
            f"{group.layer}:{group.projection}" for group in selected
        ],
    }
    for name, expected in expected_top_level.items():
        actual = prior.get(name)
        if actual != expected:
            raise ValueError(
                f"prior materialization manifest {name} must be {expected!r}, "
                f"found {actual!r}"
            )

    raw_records = prior.get("groups")
    if not isinstance(raw_records, list):
        raise ValueError("prior materialization manifest groups must be a list")
    records: dict[tuple[int, str], dict[str, object]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise ValueError("prior materialization group record must be an object")
        key = (raw_record.get("layer"), raw_record.get("projection"))
        if not isinstance(key[0], int) or not isinstance(key[1], str):
            raise ValueError("prior materialization group record has invalid identity")
        records[(key[0], key[1])] = raw_record

    verified_hashes: dict[tuple[int, str], str] = {}
    for group in selected:
        key = (group.layer, group.projection)
        output_path = root / vq_group_output_filename(group)
        if not output_path.is_file():
            continue
        record = records.get(key)
        if record is None:
            raise ValueError(f"prior materialization manifest is missing group {key}")
        expected_bytes = record.get("artifact_bytes")
        if output_path.stat().st_size != expected_bytes:
            raise ValueError(f"existing group {key} artifact byte size drifted")
        expected_sha256 = record.get("artifact_sha256")
        if not isinstance(expected_sha256, str):
            raise ValueError(f"prior materialization group {key} is missing SHA-256")
        actual_sha256 = _sha256_file(output_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"existing group {key} artifact SHA-256 drifted")
        verified_hashes[key] = actual_sha256
    return verified_hashes


def _failure_payload(
    args: argparse.Namespace,
    *,
    status: str,
    error: Exception | None = None,
    blockers: list[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_modelopt_nvfp4_materialization",
        "materialization_status": status,
        "materialization_blocked": True,
        "materialization_blockers": blockers or [],
        "profile_path": args.profile_path,
        "model_id": args.model_id,
        "source_revision": args.revision,
        "config_path": args.config_path,
        "index_path": args.index_path,
        "source_dir": args.source_dir,
        "output_dir": args.output_dir,
        "dense_checkpoint_written": False,
    }
    if error is not None:
        payload["input_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return payload


def _run(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    started = time.perf_counter()
    group_keys = _parse_group_keys(args.groups)
    source_root = Path(args.source_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("--output-dir must not be inside the pinned source directory")

    config_sha256 = _sha256_file(args.config_path)
    index_sha256 = _sha256_file(args.index_path)
    config = _load_json_object(args.config_path)
    index = load_safetensors_index(args.index_path)
    profile = load_profile(args.profile_path)
    if args.group_size != 512 or set(profile.group_size_policy.values()) != {512}:
        raise ValueError("pinned GLM52 REAP materialization requires group_size=512")
    source_audit = audit_glm52_reap_source_index(
        config=config,
        index=index,
        model_id=args.model_id,
        revision=args.revision,
        profile=profile,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
    )
    if not source_audit.source_checks_pass:
        payload = _failure_payload(
            args,
            status="glm52_modelopt_nvfp4_source_incompatible",
            blockers=list(source_audit.to_json_dict()["audit_blockers"]),
        )
        payload["source_audit"] = source_audit.to_json_dict()
        return payload, 1

    plan = plan_streaming_conversion_from_index(
        config,
        index,
        model_id=args.model_id,
        revision=args.revision,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        source_profile=profile.name,
        group_size=args.group_size,
        code_bits=args.code_bits,
    )
    planned_keys = tuple((group.layer, group.projection) for group in plan.vq_groups)
    audited_keys = tuple(
        (group.layer, group.projection) for group in source_audit.groups
    )
    if (
        plan.source_weight_encoding != SourceWeightEncoding.MODELOPT_NVFP4
        or plan.source_decoder != GLM52_REAP_SOURCE_DECODER
        or plan.missing_vq_groups
        or len(plan.vq_groups) != GLM52_REAP_EXPECTED_GROUPS
        or planned_keys != audited_keys
    ):
        raise ValueError(
            "requested conversion plan does not match the complete pinned GLM52 REAP plan"
        )
    selected = _select_groups(
        plan.vq_groups,
        group_keys=group_keys,
        max_groups=args.max_groups,
        all_groups=args.all_groups,
    )
    selected_keys = tuple((group.layer, group.projection) for group in selected)
    payload_audit = audit_glm52_reap_source_payloads(
        source_dir=args.source_dir,
        source_audit=source_audit,
        group_keys=selected_keys,
    )
    if payload_audit["materialization_blocked"]:
        payload = _failure_payload(
            args,
            status=str(payload_audit["payload_status"]),
            blockers=list(payload_audit["materialization_blockers"]),
        )
        payload["payload_audit"] = payload_audit
        return payload, 2

    prior_manifest = _load_prior_canonical_manifest(args.output_dir)
    verified_hashes = _validate_prior_artifact_identity(
        output_dir=args.output_dir,
        prior=prior_manifest,
        selected=selected,
        profile=profile.name,
        model_id=args.model_id,
        revision=args.revision,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        code_bits=args.code_bits,
        group_size=args.group_size,
        scale_estimator=args.scale_estimator,
    )

    manifest = convert_vq_groups_from_safetensors(
        source_dir=args.source_dir,
        index=index,
        plan=plan,
        output_dir=args.output_dir,
        selected_groups=selected,
        skip_existing=args.skip_existing,
        scale_estimator=args.scale_estimator,
        expert_workers=args.expert_workers,
    )
    converted_keys = {
        (Path(group.output_path).name, group.codes_name)
        for group in manifest.converted_groups
    }
    source_groups = {
        (group.layer, group.projection): group for group in source_audit.groups
    }
    codebook_name, codebook_sha256 = codebook_metadata_for_bits(args.code_bits)
    group_records: list[dict[str, object]] = []
    run_group_statuses: list[dict[str, object]] = []
    peak_decoded_expert_bytes = 0
    artifact_total_bytes = 0
    for group in selected:
        output_path = Path(args.output_dir) / vq_group_output_filename(group)
        source_group = source_groups[(group.layer, group.projection)]
        decoded_expert_bytes = group.input_dims * group.output_dims * 4
        peak_decoded_expert_bytes = max(
            peak_decoded_expert_bytes, decoded_expert_bytes
        )
        artifact_bytes = output_path.stat().st_size
        artifact_total_bytes += artifact_bytes
        artifact_sha256 = verified_hashes.get((group.layer, group.projection))
        if artifact_sha256 is None:
            artifact_sha256 = _sha256_file(output_path)
        group_record = {
            "status": "ready",
            "layer": group.layer,
            "projection": group.projection,
            "expert_count": len(group.experts),
            "source_shards": list(source_group.to_json_dict()["required_shards"]),
            "cross_shard_bundle_count": sum(
                bundle.cross_shard for bundle in source_group.bundles
            ),
            "artifact_path": str(output_path),
            "artifact_bytes": artifact_bytes,
            "artifact_sha256": artifact_sha256,
            "codes_name": group.codes_name,
            "codes_shape": list(group.codes_shape),
            "codes_dtype": "uint8" if group.code_bits == 8 else "uint16",
            "scales_name": group.scales_name,
            "scales_shape": list(group.scales_shape),
            "scales_dtype": "float16",
            "source_bundles": len(group.experts),
            "source_bundle_members": len(group.experts) * 3,
            "decoded_expert_bytes": decoded_expert_bytes,
        }
        group_records.append(group_record)
        run_group_statuses.append(
            {
                "layer": group.layer,
                "projection": group.projection,
                "run_status": (
                    "converted"
                    if (output_path.name, group.codes_name) in converted_keys
                    else "existing"
                ),
            }
        )

    full_group_coverage = len(selected) == len(plan.vq_groups)
    artifact_payload: dict[str, object] = {
        "schema_version": 1,
        "record_type": "glm52_modelopt_nvfp4_materialization_manifest",
        "materialization_status": "glm52_modelopt_nvfp4_groups_ready",
        "materialization_scope": "full" if full_group_coverage else "bounded",
        "full_group_coverage": full_group_coverage,
        "materialization_blocked": False,
        "materialization_blockers": [],
        "profile": profile.name,
        "model_id": args.model_id,
        "source_revision": args.revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "source_weight_encoding": plan.source_weight_encoding.value,
        "source_decoder": plan.source_decoder,
        "planned_vq_groups": len(plan.vq_groups),
        "selected_vq_groups": len(selected),
        "ready_vq_groups": len(selected),
        "skipped_vq_groups": len(plan.vq_groups) - len(selected),
        "selected_group_keys": [
            f"{group.layer}:{group.projection}" for group in selected
        ],
        "code_bits": args.code_bits,
        "group_size": args.group_size,
        "scale_estimator": args.scale_estimator,
        "working_set_policy": "one_decoded_expert_per_worker_v1",
        "peak_decoded_expert_bytes": peak_decoded_expert_bytes,
        "source_bundles": sum(len(group.experts) for group in selected),
        "source_bundle_members": sum(len(group.experts) * 3 for group in selected),
        "codebook_name": codebook_name,
        "codebook_sha256": codebook_sha256,
        "artifact_total_bytes": artifact_total_bytes,
        "dense_checkpoint_written": False,
        "groups": group_records,
    }
    manifest_path = Path(args.output_dir) / "conversion-manifest.json"
    _write_json_atomic(manifest_path, artifact_payload)
    (Path(args.output_dir) / PREVIOUS_MANIFEST_NAME).unlink(missing_ok=True)
    payload: dict[str, object] = {
        **artifact_payload,
        "record_type": "glm52_modelopt_nvfp4_materialization_run",
        "artifact_manifest_path": str(manifest_path),
        "artifact_manifest_sha256": _sha256_file(manifest_path),
        "converted_vq_groups": len(manifest.converted_groups),
        "existing_vq_groups": len(manifest.existing_groups),
        "skipped_existing_outputs": manifest.skipped_existing_outputs,
        "expert_workers": args.expert_workers,
        "max_decoded_experts_in_flight": args.expert_workers,
        "source_bundles_read": sum(
            group.source_tensors_read for group in manifest.converted_groups
        ),
        "source_bundle_members_read": sum(
            group.source_tensors_read * 3 for group in manifest.converted_groups
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": _max_rss_bytes(),
        "run_groups": run_group_statuses,
    }
    return payload, 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize pinned GLM-5.2 REAP ModelOpt NVFP4 groups to KEEP VQ."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--groups")
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--all-groups", action="store_true")
    parser.add_argument("--code-bits", type=int, choices=(8, 16), default=8)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument(
        "--scale-estimator",
        choices=("max_abs", "percentile_99"),
        default="max_abs",
    )
    parser.add_argument("--expert-workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    args = parser.parse_args()

    try:
        if args.expert_workers <= 0:
            raise ValueError("--expert-workers must be positive")
        payload, exit_code = _run(args)
    except Exception as error:
        payload = _failure_payload(
            args,
            status="glm52_modelopt_nvfp4_materialization_input_failed",
            error=error,
            blockers=["repair_materialization_inputs_or_plan"],
        )
        exit_code = 1
    _emit(
        payload,
        output_json=args.output_json,
        append_jsonl=args.append_jsonl,
    )
    if exit_code:
        print(payload["materialization_status"], file=sys.stderr)
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
