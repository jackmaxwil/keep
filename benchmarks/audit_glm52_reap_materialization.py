from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from mlx_vq.convert.glm52_reap import (
    GLM52_REAP_EXPECTED_GROUPS,
    GLM52_REAP_EXPECTED_LAYER_IDS,
    GLM52_REAP_PROJECTIONS,
    GLM52_REAP_SOURCE_DECODER,
    audit_glm52_reap_source_index,
)
from mlx_vq.convert.glm52_non_vq import audit_glm52_non_vq_package
from mlx_vq.convert.stream_convert import (
    SourceWeightEncoding,
    VQExpertGroup,
    load_safetensors_index,
    plan_streaming_conversion_from_index,
)
from mlx_vq.models.profiles import load_profile
from mlx_vq.validate.glm52_artifact import (
    audit_glm52_reap_materialization_manifest,
    audit_glm52_reap_source_accounting,
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


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
                f"GLM52 artifact-audit layer must be in 3..77, found {layer}"
            )
        if projection not in GLM52_REAP_PROJECTIONS:
            raise ValueError(
                f"GLM52 artifact-audit projection must be one of "
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
    with output.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


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


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validated_output_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    outputs: list[Path] = []
    for field, suffix in (("output_json", ".json"), ("append_jsonl", ".jsonl")):
        raw = getattr(args, field, None)
        if raw is None:
            continue
        lexical_output = Path(raw).expanduser()
        if lexical_output.suffix != suffix:
            raise ValueError(f"--{field.replace('_', '-')} must use the {suffix} suffix")
        output = lexical_output.resolve(strict=False)
        outputs.append(output)
    output_alias = len(outputs) != len(set(outputs))
    if not output_alias and len(outputs) == 2 and all(path.exists() for path in outputs):
        output_alias = outputs[0].samefile(outputs[1])
    if output_alias:
        raise ValueError("audit JSON and append JSONL outputs must be distinct")

    roots = [
        Path(args.source_dir).expanduser().resolve(strict=False),
        Path(args.artifact_dir).expanduser().resolve(strict=False),
    ]
    if getattr(args, "non_vq_artifact_dir", None) is not None:
        roots.append(
            Path(args.non_vq_artifact_dir).expanduser().resolve(strict=False)
        )
    protected_raw = [
        args.profile_path,
        args.config_path,
        args.index_path,
        getattr(args, "non_vq_evidence_json", None),
        getattr(args, "materialization_runs_jsonl", None),
        *(getattr(args, "materialization_evidence", None) or ()),
    ]
    protected = {
        Path(raw).expanduser().resolve(strict=False)
        for raw in protected_raw
        if raw is not None
    }
    for output in outputs:
        if any(_is_within(output, root) for root in roots):
            raise ValueError(
                "audit outputs must remain outside authority and artifact roots"
            )
        if output in protected:
            raise ValueError(f"audit output aliases protected input {output}")
        if output.exists() and any(
            path.exists() and output.samefile(path) for path in protected
        ):
            raise ValueError(f"audit output aliases protected input {output}")
    return tuple(outputs)


def _failure_payload(
    args: argparse.Namespace,
    *,
    error: Exception,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "glm52_modelopt_nvfp4_artifact_audit",
        "audit_status": "glm52_modelopt_nvfp4_artifact_audit_failed",
        "audit_pass": False,
        "artifact_integrity_pass": False,
        "model_id": args.model_id,
        "source_revision": args.revision,
        "artifact_dir": args.artifact_dir,
        "source_dir": args.source_dir,
        "profile_path": args.profile_path,
        "config_path": args.config_path,
        "index_path": args.index_path,
        "dense_routed_experts": None,
        "actual_whole_model_artifact_bytes": None,
        "actual_whole_model_bpw": None,
        "whole_model_values_actual": False,
        "actual_whole_model_tensor_payload_bytes": None,
        "actual_whole_model_tensor_payload_bpw": None,
        "whole_model_tensor_payload_values_actual": False,
        "non_vq_artifact_dir": args.non_vq_artifact_dir,
        "non_vq_evidence_json": args.non_vq_evidence_json,
        "audit_blockers": ["repair_glm52_artifact_or_audit_inputs"],
        "input_error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def _require_composite_inputs(args: argparse.Namespace) -> None:
    non_vq_artifact = getattr(args, "non_vq_artifact_dir", None)
    non_vq_evidence = getattr(args, "non_vq_evidence_json", None)
    if bool(non_vq_artifact) != bool(non_vq_evidence):
        raise ValueError(
            "--non-vq-artifact-dir and --non-vq-evidence-json must be supplied together"
        )
    if args.all_groups and not non_vq_artifact:
        raise ValueError(
            "full GLM52 audit requires the audited non-VQ package and its evidence"
        )


def _validate_non_vq_evidence(
    *,
    evidence_path: str | Path,
    artifact_dir: str | Path,
    non_vq_audit: Any,
    profile_name: str,
    model_id: str,
    revision: str,
    config_sha256: str,
    index_sha256: str,
) -> dict[str, Any]:
    evidence = _load_json_object(evidence_path)
    expected = {
        "record_type": "glm52_non_vq_package_manifest",
        "pack_status": "glm52_non_vq_package_ready",
        "production_ready": True,
        "package_audit_pass": True,
        "profile": profile_name,
        "model_id": model_id,
        "source_revision": revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
    }
    for field, value in expected.items():
        if evidence.get(field) != value:
            raise ValueError(
                f"non-VQ evidence {field} must be {value!r}, "
                f"found {evidence.get(field)!r}"
            )
    fresh_audit = non_vq_audit.to_dict()
    if evidence.get("package_audit") != fresh_audit:
        raise ValueError("non-VQ evidence package_audit does not match the fresh audit")
    recorded_root = Path(str(fresh_audit.get("artifact_dir", ""))).resolve()
    if recorded_root != Path(artifact_dir).resolve():
        raise ValueError("non-VQ evidence artifact path does not match the supplied package")
    return evidence


def _actual_composite_accounting(
    *,
    routed_audit: Any,
    non_vq_audit: Any,
    source_accounting: Any,
) -> dict[str, object]:
    if routed_audit.audit_pass is not True:
        raise ValueError("routed artifact audit must pass before composite accounting")
    if routed_audit.full_routed_artifact_ready is not True:
        raise ValueError("full routed artifact must be ready before composite accounting")
    if non_vq_audit.audit_pass is not True:
        raise ValueError("non-VQ package audit must pass before composite accounting")
    if (
        non_vq_audit.tensor_payload_bytes
        != source_accounting.main_non_routed_tensor_payload_bytes
    ):
        raise ValueError("non-VQ tensor payload bytes do not match source accounting")
    if (
        non_vq_audit.parameter_count
        != source_accounting.main_non_routed_parameter_count
    ):
        raise ValueError("non-VQ parameter count does not match source accounting")

    parameter_count = source_accounting.main_model_parameter_count_excluding_mtp
    tensor_payload_bytes = (
        routed_audit.actual_routed_payload_bytes
        + routed_audit.actual_routed_codebook_bytes
        + non_vq_audit.tensor_payload_bytes
    )
    artifact_tree_bytes = (
        routed_audit.artifact_tree_bytes + non_vq_audit.artifact_tree_bytes
    )
    return {
        "actual_whole_model_tensor_payload_bytes": tensor_payload_bytes,
        "actual_whole_model_tensor_payload_bpw": (
            tensor_payload_bytes * 8.0 / parameter_count
        ),
        "actual_whole_model_artifact_tree_bytes": artifact_tree_bytes,
        "actual_whole_model_physical_bpw": (
            artifact_tree_bytes * 8.0 / parameter_count
        ),
        "whole_model_tensor_payload_values_actual": True,
        "whole_model_physical_values_actual": True,
    }


def _run(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    _require_composite_inputs(args)
    config_sha256 = _sha256_file(args.config_path)
    index_sha256 = _sha256_file(args.index_path)
    config = _load_json_object(args.config_path)
    index = load_safetensors_index(args.index_path)
    profile = load_profile(args.profile_path)
    if profile.hf_model_id != args.model_id:
        raise ValueError(
            f"profile model_id must be {args.model_id!r}, found {profile.hf_model_id!r}"
        )
    if profile.revision != args.revision:
        raise ValueError(
            f"profile revision must be {args.revision!r}, found {profile.revision!r}"
        )

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
        audit_blockers = source_audit.to_json_dict()["audit_blockers"]
        raise ValueError(
            "pinned GLM52 source contract failed before artifact audit: "
            + ", ".join(str(item) for item in audit_blockers[:10])
        )
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
    if (
        len(plan.vq_groups) != GLM52_REAP_EXPECTED_GROUPS
        or plan.missing_vq_groups
        or plan.source_weight_encoding != SourceWeightEncoding.MODELOPT_NVFP4
        or plan.source_decoder != GLM52_REAP_SOURCE_DECODER
    ):
        raise ValueError("artifact audit plan does not match the complete pinned GLM52 plan")
    selected = _select_groups(
        plan.vq_groups,
        group_keys=_parse_group_keys(args.groups),
        max_groups=args.max_groups,
        all_groups=args.all_groups,
    )
    selected_keys = tuple(
        f"{group.layer}:{group.projection}" for group in selected
    )
    if any(group.group_size != args.group_size for group in selected):
        raise ValueError("requested group size does not match the canonical selected plan")

    accounting = audit_glm52_reap_source_accounting(
        args.source_dir,
        index=index,
        profile=profile,
    )
    non_vq_audit = None
    non_vq_evidence = None
    if args.non_vq_artifact_dir is not None:
        non_vq_audit = audit_glm52_non_vq_package(
            args.non_vq_artifact_dir,
            source_dir=args.source_dir,
            source_index=index,
            profile=profile,
            expected_model_id=args.model_id,
            expected_revision=args.revision,
            expected_config_sha256=config_sha256,
            expected_index_sha256=index_sha256,
            config_path=args.config_path,
            index_path=args.index_path,
            enforce_pinned_source=True,
        )
        non_vq_evidence = _validate_non_vq_evidence(
            evidence_path=args.non_vq_evidence_json,
            artifact_dir=args.non_vq_artifact_dir,
            non_vq_audit=non_vq_audit,
            profile_name=profile.name,
            model_id=args.model_id,
            revision=args.revision,
            config_sha256=config_sha256,
            index_sha256=index_sha256,
        )

    evidence_paths: list[str] = []
    if args.materialization_runs_jsonl is not None:
        evidence_paths.append(args.materialization_runs_jsonl)
    evidence_paths.extend(args.materialization_evidence or ())
    audit = audit_glm52_reap_materialization_manifest(
        args.artifact_dir,
        profile=profile,
        expected_group_keys=selected_keys,
        expected_code_bits=args.code_bits,
        expected_config_sha256=config_sha256,
        expected_index_sha256=index_sha256,
        evidence_paths=evidence_paths,
        non_routed_artifact_bytes=(
            None if non_vq_audit is None else non_vq_audit.artifact_tree_bytes
        ),
        whole_model_parameter_count=(
            None
            if non_vq_audit is None
            else accounting.main_model_parameter_count_excluding_mtp
        ),
    )
    payload = audit.to_dict()
    payload.update(accounting.to_dict())
    payload.update(
        {
            "schema_version": 1,
            "record_type": "glm52_modelopt_nvfp4_artifact_audit",
            "audit_status": "glm52_modelopt_nvfp4_artifact_audit_ready",
            "artifact_integrity_pass": audit.audit_pass,
            "model_id": profile.hf_model_id,
            "source_revision": profile.revision,
            "config_sha256": config_sha256,
            "index_sha256": index_sha256,
            "source_weight_encoding": SourceWeightEncoding.MODELOPT_NVFP4.value,
            "source_decoder": GLM52_REAP_SOURCE_DECODER,
            "requested_code_bits": args.code_bits,
            "requested_group_size": args.group_size,
            "requested_scale_estimator": args.scale_estimator,
            "accounting_scope": (
                "full_composite_actual"
                if non_vq_audit is not None
                else "selected_routed_actual"
            ),
            "whole_model_artifact_blocker": (
                None
                if audit.whole_model_values_actual
                else "lean_non_routed_package_not_materialized_and_audited"
            ),
        }
    )
    if non_vq_audit is not None:
        payload.update(
            _actual_composite_accounting(
                routed_audit=audit,
                non_vq_audit=non_vq_audit,
                source_accounting=accounting,
            )
        )
        payload.update(
            {
                "non_vq_artifact_dir": str(Path(args.non_vq_artifact_dir)),
                "non_vq_evidence_json": str(Path(args.non_vq_evidence_json)),
                "non_vq_evidence_sha256": _sha256_file(args.non_vq_evidence_json),
                "non_vq_package_audit": non_vq_audit.to_dict(),
                "non_vq_package_evidence_authenticated": non_vq_evidence is not None,
            }
        )
    else:
        payload.update(
            {
                "actual_whole_model_tensor_payload_bytes": None,
                "actual_whole_model_tensor_payload_bpw": None,
                "actual_whole_model_artifact_tree_bytes": None,
                "actual_whole_model_physical_bpw": None,
                "whole_model_tensor_payload_values_actual": False,
                "whole_model_physical_values_actual": False,
                "non_vq_package_evidence_authenticated": False,
            }
        )
    if args.scale_estimator != _load_json_object(audit.manifest_path).get(
        "scale_estimator"
    ):
        raise ValueError(
            "artifact scale estimator does not match the explicitly requested estimator"
        )
    if audit.full_routed_artifact_ready:
        projected_bytes = (
            audit.actual_routed_payload_bytes
            + audit.actual_routed_codebook_bytes
            + accounting.main_non_routed_tensor_payload_bytes
        )
        projected_bpw = (
            projected_bytes
            * 8.0
            / accounting.main_model_parameter_count_excluding_mtp
        )
        payload["projected_whole_model_payload_bytes_if_source_precision_non_routed_repacked"] = projected_bytes
        payload["projected_whole_model_payload_bpw_if_source_precision_non_routed_repacked"] = projected_bpw
    else:
        payload["projected_whole_model_payload_bytes_if_source_precision_non_routed_repacked"] = None
        payload["projected_whole_model_payload_bpw_if_source_precision_non_routed_repacked"] = None

    resume_required_but_missing = args.require_resume_proof and not audit.resume_verified
    if resume_required_but_missing:
        payload["audit_status"] = "glm52_modelopt_nvfp4_artifact_resume_proof_missing"
        payload["audit_pass"] = False
        payload["audit_blockers"] = ["rerun_materializer_then_repeat_artifact_audit"]
        return payload, 2
    payload["audit_pass"] = audit.audit_pass
    payload["audit_blockers"] = []
    return payload, 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly audit pinned GLM-5.2 REAP routed VQ artifacts."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--non-vq-artifact-dir")
    parser.add_argument("--non-vq-evidence-json")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--profile-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--groups")
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--all-groups", action="store_true")
    parser.add_argument("--code-bits", type=int, choices=(8, 16), required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument(
        "--scale-estimator",
        choices=("max_abs", "percentile_99"),
        required=True,
    )
    parser.add_argument("--materialization-runs-jsonl")
    parser.add_argument("--materialization-evidence", action="append")
    parser.add_argument("--require-resume-proof", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--append-jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        _validated_output_paths(args)
    except ValueError as error:
        print(f"invalid audit output path: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    try:
        payload, exit_code = _run(args)
    except Exception as error:
        payload = _failure_payload(args, error=error)
        exit_code = 1
    _emit(
        payload,
        output_json=args.output_json,
        append_jsonl=args.append_jsonl,
    )
    if exit_code:
        print(payload["audit_status"])
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
