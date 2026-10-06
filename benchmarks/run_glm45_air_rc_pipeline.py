from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from mlx_vq.benchmark.quant_compare import audit_vq_artifact_prefill_compatibility
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID
from keep import verify_environment
from mlx_vq.quality.plan_next import (
    _quality_plan_candidates,
    _select_quality_plan_rows,
)
from mlx_vq.quality.rc_gates import (
    BALANCED_HARD_TARGETS,
    COMMUNITY_WOW_TARGETS,
    LANE_S_TIMING_RELATIVE_SPREAD_MAX,
    PARENT_VM_DELTA_KEYS,
    QUALITY_PLAN_CATEGORY_WEIGHTS,
    QUALITY_PLAN_DOMAIN_QUOTAS,
    QUALITY_PLAN_DOMAIN_WEIGHTS,
    SPLIT_PREFIXES,
    _all,
    _build_quality_focus,
    _check_ge,
    _check_le,
    _domain_from_record,
    _float_or_none,
    _focus_row,
    _is_clean_benchmark_row,
    _lane_s_failure_reasons,
    _load_manifest,
    _quality_checks,
    _read_json_object,
    _read_jsonl,
    _selected_parent_vm_deltas,
    _sum_delta_maps,
    _summarize_benchmark,
    _summarize_benchmark_memory,
    _summarize_eval_domains,
    _summarize_eval_split,
    _summarize_parent_vm_stats,
    _take_ranked,
    _timing_summary,
    _top_token_kld,
    assert_engine_tier_compatible,
)


DEFAULT_ARTIFACT_DIR = (
    "artifacts/glm-4.5-air-dynamic3p0-r26-lora-l45-gud-math8-r4-init0p05-"
    "s12-lr0p5-w2-m1-nll0p5-20260701"
)
DEFAULT_SEED_ARTIFACT_DIR = (
    "artifacts/glm-4.5-air-dynamic3p0-target23-joint-gate-up-down-nextcycle-"
    "r25-r26-s48-lr1-w2-m1-nll0p5-20260701"
)
DEFAULT_REPORT_JSONL = (
    "artifacts/quality/glm45-air-dynamic3p0-r26-lora-l45-gud-math8-r4-"
    "init0p05-s12-lr0p5-w2-m1-nll0p5-report128-20260701.jsonl"
)
DEFAULT_SELECTION_JSONL = (
    "artifacts/quality/glm45-air-dynamic3p0-r26-lora-l45-gud-math8-r4-"
    "init0p05-s12-lr0p5-w2-m1-nll0p5-select128-20260701.jsonl"
)
DEFAULT_HOLDOUT_JSONL = (
    "artifacts/quality/glm45-air-dynamic3p0-r26-lora-l45-gud-math8-r4-"
    "init0p05-s12-lr0p5-w2-m1-nll0p5-holdout128-20260701.jsonl"
)
DEFAULT_LANE_S_JSONL = (
    "artifacts/benchmarks/glm45-air-dynamic3p0-r26-lora-l45-gud-math8-r4-"
    "init0p05-s12-lr0p5-w2-m1-nll0p5-lane-s-prefill1k-quiet-20260701.jsonl"
)
DEFAULT_Q2_CONTROL_JSONL = (
    "artifacts/benchmarks/glm45-air-target23-nextcycle-r26-q2-control-"
    "prefill1k-quiet-20260701.jsonl"
)
DEFAULT_QUALITY_PLAN1_ARTIFACT_DIR = "artifacts/glm45-air-quality-r4-route-math-plan1"
DEFAULT_QUALITY_PLAN1_COMPARE_JSONS = {
    "report": "artifacts/quality/glm45-air-quality-r4-route-math-plan1-report128-compare.json",
    "selection": "artifacts/quality/glm45-air-quality-r4-route-math-plan1-select128-compare.json",
    "holdout": "artifacts/quality/glm45-air-quality-r4-route-math-plan1-holdout128-compare.json",
}
DEFAULT_QUALITY_PLAN1_LANE_S_JSONL = (
    "artifacts/benchmarks/glm45-air-quality-r4-route-math-plan1-lane-s-prefill1k-quiet-rerun3.jsonl"
)
DEFAULT_QUALITY_PLAN1_Q2_CONTROL_JSONL = (
    "artifacts/benchmarks/glm45-air-quality-r4-route-math-plan1-q2-control-prefill1k-quiet-rerun3.jsonl"
)

DEFAULT_TEACHER_JSONLS = {
    "report": "artifacts/quality/glm45-air-teacher-cache-air-vq-ladder-report-v1-full-logits-route-trace-clean/metadata.jsonl",
    "selection": "artifacts/quality/glm45-air-teacher-cache-air-vq-ladder-select-v1-full-logits-route-trace-clean/metadata.jsonl",
    "holdout": "artifacts/quality/glm45-air-teacher-cache-air-vq-ladder-holdout-v1-full-logits-clean-merged-20260701/metadata.jsonl",
}
DEFAULT_IMATRIX_MANIFEST = (
    "artifacts/imatrix/glm45-air-public-calibration/imatrix-manifest.json"
)
DEFAULT_HIGH_BIT_ARTIFACT_DIR = (
    "artifacts/glm-4.5-air-highbit-routed-imatrix-reference-20260701"
)
DEFAULT_MODEL_ID = GLM45_AIR_MODEL_ID
DEFAULT_REVISION = "main"

REQUIRED_ENTRYPOINTS = (
    "scripts/glm45_air_rc.sh",
    "benchmarks/run_glm45_air_rc_pipeline.py",
    "benchmarks/eval_glm45_air_teacher_cache.py",
    "benchmarks/bench_glm45_air_quant_compare.py",
    "benchmarks/collect_glm45_air_imatrix.py",
    "benchmarks/materialize_glm45_air_dynamic_imatrix_sweep.py",
    "benchmarks/finetune_glm45_air_vq_continuous.py",
    "benchmarks/analyze_glm45_air_teacher_cache_attribution.py",
)
REQUIRED_PUBLIC_DOCS = (
    "README.md",
    "docs/GLM45_AIR_RC_PIPELINE.md",
    "docs/KEEP_NEW_MODEL_FAMILY_TEMPLATE.md",
)
REQUIRED_PUBLIC_DOC_PATTERNS = {
    "README.md": (
        "## What This Repo Does",
        "## Architecture",
        "## Quick Start",
        "scripts/glm45_air_rc.sh env-preflight",
        "scripts/glm45_air_rc.sh reproduce",
        "scripts/glm45_air_rc.sh verify",
        "scripts/glm45_air_rc.sh attribute",
        "source model availability",
        "## Current Limitations",
        "docs/KEEP_NEW_MODEL_FAMILY_TEMPLATE.md",
    ),
    "docs/GLM45_AIR_RC_PIPELINE.md": (
        "## One-Command RC Summary",
        "scripts/glm45_air_rc.sh env-preflight",
        "scripts/glm45_air_rc.sh reproduce",
        "scripts/glm45_air_rc.sh verify",
        "scripts/glm45_air_rc.sh attribute",
        "source model availability",
        "## Regenerate Evaluation Evidence",
        "## Regenerate Lane S",
        "## Inspect Tail Attribution",
        "## Artifact Audit",
        "## Train A Rank-4 Sidecar",
        "## Extension Path For Another Model Family",
        "## Current Limitations",
    ),
    "docs/KEEP_NEW_MODEL_FAMILY_TEMPLATE.md": (
        "# New Model Family Template For KEEP",
        "## Adapter Contract",
        "## Source Tensor Mapping",
        "## Eval Contract",
        "## Benchmark Contract",
        "## Required Tests",
        "## Publication Gate",
    ),
}
EXECUTE_OUTPUTS = {
    "eval_report": [("report_jsonl", "report128.jsonl")],
    "eval_selection": [("selection_jsonl", "selection128.jsonl")],
    "eval_holdout": [("holdout_jsonl", "holdout128.jsonl")],
    "lane_s_q2_control": [("q2_control_jsonl", "lane-s-q2-control.jsonl")],
    "lane_s_candidate": [("lane_s_jsonl", "lane-s-candidate.jsonl")],
    "lane_s_candidate_cache1_diagnostic": [
        (None, "lane-s-candidate-cache1-diagnostic.jsonl")
    ],
    "lane_s_candidate_cache2_diagnostic": [
        (None, "lane-s-candidate-cache2-diagnostic.jsonl")
    ],
    "lane_s_candidate_cache4_diagnostic": [
        (None, "lane-s-candidate-cache4-diagnostic.jsonl")
    ],
}
EXECUTE_ARTIFACT_ENGINE_CHECKS = {
    "eval_report": "vq_e1_routed_nax_e8p",
    "eval_selection": "vq_e1_routed_nax_e8p",
    "eval_holdout": "vq_e1_routed_nax_e8p",
    "lane_s_candidate": "vq_e1_routed_nax_e8p",
    "lane_s_candidate_cache1_diagnostic": "vq_e1_routed_nax_e8p",
    "lane_s_candidate_cache2_diagnostic": "vq_e1_routed_nax_e8p",
    "lane_s_candidate_cache4_diagnostic": "vq_e1_routed_nax_e8p",
}


def _path_check(
    name: str,
    path: str | Path,
    *,
    kind: str,
    executable: bool = False,
) -> dict[str, Any]:
    candidate = Path(path)
    exists = candidate.exists()
    if kind == "file":
        type_ok = candidate.is_file()
    elif kind == "dir":
        type_ok = candidate.is_dir()
    else:
        raise ValueError(f"unsupported path check kind: {kind}")
    executable_ok = not executable or os.access(candidate, os.X_OK)
    ok = type_ok and executable_ok
    return {
        "name": name,
        "path": str(candidate),
        "kind": kind,
        "exists": exists,
        "type_ok": type_ok,
        "executable_required": executable,
        "executable_ok": executable_ok,
        "ok": ok,
    }


def _is_runtime_tensor_name(tensor_name: str, *, num_hidden_layers: int) -> bool:
    prefix = "model.layers."
    if not tensor_name.startswith(prefix):
        return True
    suffix = tensor_name[len(prefix):]
    layer_text = suffix.split(".", 1)[0]
    try:
        layer_index = int(layer_text)
    except ValueError:
        return True
    return layer_index < num_hidden_layers


def _source_model_check(
    *,
    model_id: str,
    revision: str,
    source_dir: str | None,
) -> dict[str, Any]:
    errors: list[str] = []
    if source_dir:
        source_root = Path(source_dir)
        config_path = source_root / "config.json"
        index_path = source_root / "model.safetensors.index.json"
        source_resolution = "source_dir"
    else:
        source_resolution = "huggingface_cache_local_files_only"
        try:
            config_path = Path(
                hf_hub_download(
                    model_id,
                    "config.json",
                    revision=revision,
                    local_files_only=True,
                )
            )
            index_path = Path(
                hf_hub_download(
                    model_id,
                    "model.safetensors.index.json",
                    revision=revision,
                    local_files_only=True,
                )
            )
            source_root = index_path.parent
        except Exception as error:  # pragma: no cover - Hugging Face exception types vary.
            return {
                "name": "source_model",
                "model_id": model_id,
                "revision": revision,
                "source_resolution": source_resolution,
                "source_dir": source_dir,
                "config_path": None,
                "index_path": None,
                "config_exists": False,
                "index_exists": False,
                "num_hidden_layers": None,
                "required_shards": None,
                "present_shards": None,
                "missing_shards": None,
                "ignored_optional_shards": [],
                "present_bytes": None,
                "ok": False,
                "errors": [f"source model metadata not available in local cache: {error}"],
            }

    config_exists = config_path.is_file()
    index_exists = index_path.is_file()
    if not config_exists:
        errors.append(f"missing source config: {config_path}")
    if not index_exists:
        errors.append(f"missing source safetensors index: {index_path}")

    num_hidden_layers = None
    if config_exists:
        try:
            config = _read_json_object(config_path)
            if "num_hidden_layers" in config:
                num_hidden_layers = int(config["num_hidden_layers"])
        except Exception as error:
            errors.append(f"invalid source config {config_path}: {error}")

    required_shards = None
    present_shards = None
    missing_shards: list[str] | None = None
    present_bytes = None
    ignored_optional_shards: list[str] = []
    if index_exists:
        try:
            index = _read_json_object(index_path)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, Mapping):
                raise ValueError("source index missing weight_map object")
            runtime_weight_map = dict(weight_map)
            if num_hidden_layers is not None:
                runtime_weight_map = {
                    str(tensor_name): shard_name
                    for tensor_name, shard_name in weight_map.items()
                    if _is_runtime_tensor_name(str(tensor_name), num_hidden_layers=num_hidden_layers)
                }
            shard_names = sorted({str(name) for name in runtime_weight_map.values()})
            ignored_optional_shards = sorted(
                {str(name) for name in weight_map.values()} - set(shard_names)
            )
            required_shards = len(shard_names)
            missing = []
            present = 0
            bytes_total = 0
            for shard_name in shard_names:
                shard_path = source_root / shard_name
                if shard_path.is_file():
                    present += 1
                    bytes_total += shard_path.stat().st_size
                else:
                    missing.append(str(shard_path))
            present_shards = present
            missing_shards = missing
            present_bytes = bytes_total
            if missing:
                preview = ", ".join(missing[:3])
                suffix = "" if len(missing) <= 3 else f", ... ({len(missing)} missing)"
                errors.append(f"missing source safetensors shards: {preview}{suffix}")
        except Exception as error:
            errors.append(f"invalid source safetensors index {index_path}: {error}")

    return {
        "name": "source_model",
        "model_id": model_id,
        "revision": revision,
        "source_resolution": source_resolution,
        "source_dir": str(source_root),
        "config_path": str(config_path),
        "index_path": str(index_path),
        "config_exists": config_exists,
        "index_exists": index_exists,
        "num_hidden_layers": num_hidden_layers,
        "required_shards": required_shards,
        "present_shards": present_shards,
        "missing_shards": missing_shards,
        "ignored_optional_shards": ignored_optional_shards,
        "present_bytes": present_bytes,
        "ok": not errors,
        "errors": errors,
    }


def _public_doc_check(path: str | Path) -> dict[str, Any]:
    check = _path_check(f"public_doc:{path}", path, kind="file")
    required_patterns = REQUIRED_PUBLIC_DOC_PATTERNS.get(str(path), ())
    missing_patterns: list[str] = []
    if check["type_ok"]:
        text = Path(path).read_text()
        missing_patterns = [pattern for pattern in required_patterns if pattern not in text]
    else:
        missing_patterns = list(required_patterns)
    content_ok = not missing_patterns
    check.update(
        {
            "required_patterns": list(required_patterns),
            "missing_patterns": missing_patterns,
            "content_ok": content_ok,
            "ok": bool(check["ok"] and content_ok),
        }
    )
    return check


def _build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.artifact_dir)
    seed_artifact_dir = Path(args.seed_artifact_dir)
    output_dir = Path(args.output_dir)
    paths = [
        _path_check("artifact_dir", artifact_dir, kind="dir"),
        _path_check("artifact_manifest", artifact_dir / "conversion-manifest.json", kind="file"),
        _path_check("seed_artifact_dir", seed_artifact_dir, kind="dir"),
        _path_check("report_jsonl", args.report_jsonl, kind="file"),
        _path_check("selection_jsonl", args.selection_jsonl, kind="file"),
        _path_check("holdout_jsonl", args.holdout_jsonl, kind="file"),
        _path_check("lane_s_jsonl", args.lane_s_jsonl, kind="file"),
        _path_check("q2_control_jsonl", args.q2_control_jsonl, kind="file"),
        _path_check("report_teacher_jsonl", args.report_teacher_jsonl, kind="file"),
        _path_check("selection_teacher_jsonl", args.selection_teacher_jsonl, kind="file"),
        _path_check("holdout_teacher_jsonl", args.holdout_teacher_jsonl, kind="file"),
        _path_check("report_teacher_cache_root", Path(args.report_teacher_jsonl).parent, kind="dir"),
        _path_check("selection_teacher_cache_root", Path(args.selection_teacher_jsonl).parent, kind="dir"),
        _path_check("holdout_teacher_cache_root", Path(args.holdout_teacher_jsonl).parent, kind="dir"),
    ]
    entrypoints = [
        _path_check(
            f"entrypoint:{path}",
            path,
            kind="file",
            executable=str(path).endswith(".sh"),
        )
        for path in REQUIRED_ENTRYPOINTS
    ]
    public_docs = [_public_doc_check(path) for path in REQUIRED_PUBLIC_DOCS]
    environment = verify_environment().to_dict()
    source_model = _source_model_check(
        model_id=args.model_id,
        revision=args.revision,
        source_dir=args.source_dir,
    )
    manifest = _load_manifest(artifact_dir)
    manifest_seed = manifest.get("seed_artifact_dir")
    manifest_checks = {
        "manifest_exists": bool(manifest.get("exists")),
        "seed_artifact_recorded": bool(manifest_seed),
        "seed_artifact_matches_expected": manifest_seed in (None, str(seed_artifact_dir)),
        "artifact_is_separate_from_seed": artifact_dir != seed_artifact_dir,
        "continuous_sidecars_present": int(manifest.get("sidecar_count") or 0) > 0,
        "low_rank_residual_recorded": manifest.get("trainable") == "low_rank_residual",
    }
    output_checks = {
        "output_dir": str(output_dir),
        "output_dir_exists": output_dir.exists(),
        "summary_json_exists": (output_dir / "rc-summary.json").exists(),
        "summary_markdown_exists": (output_dir / "RC_SUMMARY.md").exists(),
        "overwrite_required_for_summary": (
            ((output_dir / "rc-summary.json").exists() or (output_dir / "RC_SUMMARY.md").exists())
            and not bool(args.overwrite)
        ),
    }
    path_ok = all(item["ok"] for item in paths)
    entrypoint_ok = all(item["ok"] for item in entrypoints)
    public_docs_ok = all(item["ok"] for item in public_docs)
    environment_ok = bool(environment["ok"])
    source_model_ok = bool(source_model["ok"])
    manifest_ok = all(manifest_checks.values())
    preflight_ok = (
        path_ok
        and entrypoint_ok
        and public_docs_ok
        and environment_ok
        and source_model_ok
        and manifest_ok
    )
    return {
        "schema_version": 1,
        "status": "pass" if preflight_ok else "fail",
        "artifact_dir": str(artifact_dir),
        "seed_artifact_dir": str(seed_artifact_dir),
        "checks": {
            "paths": paths,
            "entrypoints": entrypoints,
            "public_docs": public_docs,
            "environment": environment,
            "source_model": source_model,
            "manifest": manifest_checks,
            "output": output_checks,
        },
        "summary": {
            "path_checks_pass": path_ok,
            "entrypoint_checks_pass": entrypoint_ok,
            "public_doc_checks_pass": public_docs_ok,
            "environment_checks_pass": environment_ok,
            "source_model_checks_pass": source_model_ok,
            "manifest_checks_pass": manifest_ok,
            "preflight_pass": preflight_ok,
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "cwd": os.getcwd(),
        },
    }


def _render_preflight_markdown(preflight: Mapping[str, Any]) -> str:
    rows = []
    for group in ("paths", "entrypoints", "public_docs"):
        for item in preflight["checks"][group]:
            rows.append(
                "| {group} | {name} | `{status}` | `{path}` |".format(
                    group=group,
                    name=item["name"],
                    status="pass" if item["ok"] else "fail",
                    path=item["path"],
                )
            )
    manifest_rows = [
        f"- {name}: `{'pass' if value else 'fail'}`"
        for name, value in preflight["checks"]["manifest"].items()
    ]
    output_rows = [
        f"- {name}: `{value}`"
        for name, value in preflight["checks"]["output"].items()
    ]
    environment = preflight["checks"]["environment"]
    env_rows = [
        f"- MLX version: `{environment.get('mlx_version')}`",
        f"- mlx-lm available: `{environment.get('mlx_lm_available')}`",
        f"- Metal kernel available: `{environment.get('metal_kernel_available')}`",
        f"- Hadamard sizes: `{environment.get('hadamard_sizes_ok')}`",
        f"- Safetensors integer roundtrip: `{environment.get('safetensors_integer_roundtrip')}`",
        f"- Status: `{'pass' if environment.get('ok') else 'fail'}`",
    ]
    env_rows.extend(f"- Error: `{error}`" for error in environment.get("failures") or [])
    source = preflight["checks"]["source_model"]
    source_rows = [
        f"- Model: `{source.get('model_id')}` @ `{source.get('revision')}`",
        f"- Resolution: `{source.get('source_resolution')}`",
        f"- Source: `{source.get('source_dir')}`",
        f"- Config: `{source.get('config_path')}` exists `{source.get('config_exists')}`",
        f"- Index: `{source.get('index_path')}` exists `{source.get('index_exists')}`",
        f"- Runtime layers: `{source.get('num_hidden_layers')}`",
        "- Shards: `{present}/{required}` present, missing `{missing}`".format(
            present=source.get("present_shards"),
            required=source.get("required_shards"),
            missing=len(source.get("missing_shards") or []),
        ),
        f"- Ignored optional shards: `{len(source.get('ignored_optional_shards') or [])}`",
        f"- Status: `{'pass' if source.get('ok') else 'fail'}`",
    ]
    source_rows.extend(f"- Error: `{error}`" for error in source.get("errors") or [])
    public_doc_rows = []
    for item in preflight["checks"]["public_docs"]:
        missing_patterns = item.get("missing_patterns") or []
        public_doc_rows.append(
            "- {path}: `{status}`{missing}".format(
                path=item["path"],
                status="pass" if item.get("content_ok") else "fail",
                missing=(
                    " missing `" + "`, `".join(str(pattern) for pattern in missing_patterns) + "`"
                    if missing_patterns
                    else ""
                ),
            )
        )
    return "\n".join(
        [
            "# GLM-4.5-Air VQ RC Preflight",
            "",
            f"- Status: `{preflight['status']}`",
            f"- Artifact: `{preflight['artifact_dir']}`",
            f"- Seed artifact: `{preflight['seed_artifact_dir']}`",
            f"- Python: `{preflight['summary']['python_version']}`",
            f"- CWD: `{preflight['summary']['cwd']}`",
            "",
            "## Paths And Entrypoints",
            "",
            "| Group | Name | Status | Path |",
            "| --- | --- | --- | --- |",
            *rows,
            "",
            "## Public Doc Content",
            "",
            *public_doc_rows,
            "",
            "## Environment",
            "",
            *env_rows,
            "",
            "## Source Model",
            "",
            *source_rows,
            "",
            "## Manifest Checks",
            "",
            *manifest_rows,
            "",
            "## Output Safety",
            "",
            *output_rows,
            "",
        ]
    )


def _summary_cli_errors(args: argparse.Namespace) -> list[str]:
    preflight = _build_preflight(args)
    errors: list[str] = []
    for item in preflight["checks"]["paths"]:
        if not item["ok"]:
            errors.append(f"missing required {item['kind']} for {item['name']}: {item['path']}")
    for item in preflight["checks"]["entrypoints"]:
        if not item["ok"]:
            if item.get("type_ok") and item.get("executable_required") and not item.get("executable_ok"):
                errors.append(f"required entrypoint is not executable: {item['path']}")
            else:
                errors.append(f"missing required entrypoint: {item['path']}")
    for item in preflight["checks"]["public_docs"]:
        if not item["ok"]:
            if item.get("type_ok") and item.get("missing_patterns"):
                missing = ", ".join(str(pattern) for pattern in item["missing_patterns"])
                errors.append(f"public doc is missing required content in {item['path']}: {missing}")
            else:
                errors.append(f"missing required public doc: {item['path']}")
    environment = preflight["checks"]["environment"]
    if not environment["ok"]:
        for error in environment.get("failures") or ["environment check failed"]:
            errors.append(f"environment check failed: {error}")
    source_model = preflight["checks"]["source_model"]
    if not source_model["ok"]:
        for error in source_model.get("errors") or ["source model is unavailable"]:
            errors.append(f"source model check failed: {error}")
    for name, passed in preflight["checks"]["manifest"].items():
        if not passed:
            errors.append(f"artifact manifest check failed: {name}")
    output = preflight["checks"]["output"]
    if output["overwrite_required_for_summary"]:
        errors.append(
            "summary outputs already exist; pass --overwrite to replace "
            f"{Path(args.output_dir) / 'rc-summary.json'} and "
            f"{Path(args.output_dir) / 'RC_SUMMARY.md'}"
        )
    if args.write_focus_json and Path(args.write_focus_json).exists() and not args.overwrite:
        errors.append(f"focus JSON already exists; pass --overwrite to replace {args.write_focus_json}")
    if args.write_quality_plan_json and Path(args.write_quality_plan_json).exists() and not args.overwrite:
        errors.append(
            f"quality plan JSON already exists; pass --overwrite to replace {args.write_quality_plan_json}"
        )
    if (
        args.write_quality_frontier_json
        and Path(args.write_quality_frontier_json).exists()
        and not args.overwrite
    ):
        errors.append(
            "quality frontier JSON already exists; pass --overwrite to replace "
            f"{args.write_quality_frontier_json}"
        )
    return errors


def _format_optional_float(value: Any) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return ""
    return f"{numeric:.6f}"


def _format_delta_map(deltas: Any) -> str:
    if not isinstance(deltas, Mapping) or not deltas:
        return ""
    parts = []
    for key in PARENT_VM_DELTA_KEYS:
        value = deltas.get(key)
        if isinstance(value, int):
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _format_timing_seconds(values: Any) -> str:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ""
    formatted = [_format_optional_float(value) for value in values]
    return ", ".join(value for value in formatted if value)


def _compact_split_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_count": summary.get("record_count"),
        "clean_record_count": summary.get("clean_record_count"),
        "all_memory_clean": summary.get("all_memory_clean"),
        "mean_kld": summary.get("mean_kld"),
        "mean_top1_agreement": summary.get("mean_top1_agreement"),
        "mean_ppl_ratio": summary.get("mean_ppl_ratio"),
        "max_ppl_ratio": summary.get("max_ppl_ratio"),
        "p999_kld": summary.get("p999_kld"),
    }


GOLDEN_Q2_BASELINE_PATH = Path("artifacts/quality/golden-q2-baseline-prefill1k.json")


def _load_golden_q2_baseline() -> float | None:
    """Return the golden q2 prefill_1k median seconds, measured once under quiet
    conditions, used as a stable Lane S denominator reference. Absent -> None, and
    the ratio guard degrades to a warning (behavior unchanged) until it is measured.
    """
    override = os.environ.get("GLM45_AIR_GOLDEN_Q2_SECONDS")
    if override:
        try:
            return float(override)
        except ValueError:
            return None
    if not GOLDEN_Q2_BASELINE_PATH.exists():
        return None
    try:
        data = json.loads(GOLDEN_Q2_BASELINE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    value = data.get("median_seconds") if isinstance(data, Mapping) else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _load_quality_compare(path: Path) -> dict[str, Any]:
    compare = _read_json_object(path)
    decision = compare.get("decision") if isinstance(compare.get("decision"), Mapping) else {}
    return {
        "path": str(path),
        "decision_label": decision.get("label"),
        "quality_label_without_speed": decision.get("quality_label_without_speed"),
        "candidate_summary": _compact_split_summary(
            compare.get("candidate_summary") if isinstance(compare.get("candidate_summary"), Mapping) else {}
        ),
        "baseline_summary": _compact_split_summary(
            compare.get("baseline_summary") if isinstance(compare.get("baseline_summary"), Mapping) else {}
        ),
        "summary_delta": compare.get("summary_delta") if isinstance(compare.get("summary_delta"), Mapping) else {},
    }


def _build_quality_frontier(
    *,
    compare_jsons: Mapping[str, str | Path] = DEFAULT_QUALITY_PLAN1_COMPARE_JSONS,
    lane_s_jsonl: str | Path = DEFAULT_QUALITY_PLAN1_LANE_S_JSONL,
    q2_control_jsonl: str | Path = DEFAULT_QUALITY_PLAN1_Q2_CONTROL_JSONL,
    artifact_dir: str = DEFAULT_QUALITY_PLAN1_ARTIFACT_DIR,
) -> dict[str, Any]:
    paths = [Path(path) for path in compare_jsons.values()] + [Path(lane_s_jsonl), Path(q2_control_jsonl)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return {
            "schema_version": 1,
            "status": "quality_frontier_evidence_unavailable",
            "promoted": "balanced",
            "missing_paths": missing,
            "attempts": [],
        }

    split_compares = {
        split_name: _load_quality_compare(Path(path))
        for split_name, path in compare_jsons.items()
    }
    candidate_bench = _summarize_benchmark(Path(lane_s_jsonl))
    q2_control_bench = _summarize_benchmark(Path(q2_control_jsonl))
    candidate_median = candidate_bench.get("median_seconds")
    q2_median = q2_control_bench.get("median_seconds")
    lane_s_ratio = (
        float(candidate_median) / float(q2_median)
        if candidate_median is not None and q2_median not in (None, 0)
        else None
    )
    # Lane S denominator guard (added 2026-07-07 after the plan1 false-rejection):
    # candidate and q2 are measured in SEPARATE processes, so the q2 denominator floats
    # per window. An anomalously fast q2 (e.g. plan1's window at 0.94s vs the ~1.5s seen
    # in every other window) inflates the ratio and wrongly rejects a candidate that is
    # actually speed-tied with the accepted RC. We compare the measured q2 against a
    # golden baseline (measured once under quiet conditions) and, when the window is
    # inconsistent, flag the ratio as suspect rather than hard-failing the candidate.
    golden_q2 = _load_golden_q2_baseline()
    q2_window_consistent: bool | None = None
    if golden_q2 is not None and q2_median not in (None, 0):
        rel = float(q2_median) / float(golden_q2)
        q2_window_consistent = 0.75 <= rel <= 1.35
    ratio_check = _check_le(lane_s_ratio, BALANCED_HARD_TARGETS["lane_s_ratio_max"])
    lane_s_checks = {
        "candidate_clean": bool(candidate_bench.get("all_memory_clean")),
        "q2_control_clean": bool(q2_control_bench.get("all_memory_clean")),
        "candidate_timing_stable": bool(candidate_bench.get("timing_stable")),
        "q2_control_timing_stable": bool(q2_control_bench.get("timing_stable")),
        "ratio_le_1p15": ratio_check,
        "effective_bpw_le_2p5": _check_le(candidate_bench.get("effective_bits_per_weight"), 2.5),
        "no_dense_routed_experts": candidate_bench.get("dense_expert_params") is False,
        "no_unbound_vq_experts": candidate_bench.get("unbound_vq_experts") is False,
        "non_expert_dtype_verified": candidate_bench.get("non_expert_dtype_verified") is True,
    }
    # A ratio computed against a suspect (inconsistent-window) q2 denominator is not a
    # trustworthy speed verdict. If the ONLY failing check is the ratio and the q2 window
    # is inconsistent with the golden baseline, the correct outcome is "remeasure", not a
    # hard speed rejection.
    lane_s_measurement_suspect = (
        q2_window_consistent is False
        and not ratio_check
        and all(passed for key, passed in lane_s_checks.items() if key != "ratio_le_1p15")
    )
    split_checks = {
        split_name: _quality_checks(compare["candidate_summary"])
        for split_name, compare in split_compares.items()
    }
    quality_target_misses = {
        split_name: {
            key: passed
            for key, passed in checks.items()
            if key in {"mean_kld_le_0p30", "p999_kld_le_3p0", "top1_ge_0p85"} and not passed
        }
        for split_name, checks in split_checks.items()
    }
    promoted = _all(lane_s_checks) and all(not misses for misses in quality_target_misses.values())
    promotion_status = "quality_candidate_promoted" if promoted else "quality_candidate_rejected"
    rejection_reasons: list[str] = []
    if not _all(lane_s_checks):
        if lane_s_measurement_suspect:
            rejection_reasons.append("lane_s_ratio_suspect_q2_window_inconsistent_remeasure_needed")
        else:
            rejection_reasons.append("lane_s_speed_or_invariant_gate_failed")
    if any(misses for misses in quality_target_misses.values()):
        rejection_reasons.append("community_wow_quality_targets_still_missed")
    if not promoted and lane_s_measurement_suspect and all(
        not misses for misses in quality_target_misses.values()
    ):
        decision = "RECOVER_lane_s_remeasure_needed"
    elif promoted:
        decision = "PROMOTE"
    else:
        decision = "RECOVER_quality_rejected_speed"

    return {
        "schema_version": 1,
        "status": "balanced_promoted_quality_attempt_recorded",
        "promoted": "quality" if promoted else "balanced",
        "attempts": [
            {
                "label": "quality-r4-route-math-plan1",
                "artifact_dir": artifact_dir,
                "promotion_status": promotion_status,
                "decision": decision,
                "rejection_reasons": rejection_reasons,
                "split_compares": split_compares,
                "split_checks": split_checks,
                "quality_target_misses": quality_target_misses,
                "lane_s": {
                    "candidate": candidate_bench,
                    "q2_control": q2_control_bench,
                    "ratio": lane_s_ratio,
                    "golden_q2_baseline_seconds": golden_q2,
                    "q2_window_consistent": q2_window_consistent,
                    "measurement_suspect": lane_s_measurement_suspect,
                    "checks": lane_s_checks,
                    "failure_reasons": _lane_s_failure_reasons(lane_s_checks),
                },
            }
        ],
    }


def _build_rerun_commands(
    *,
    artifact_dir: str,
    output_dir: str,
    report_teacher_jsonl: str,
    selection_teacher_jsonl: str,
    holdout_teacher_jsonl: str,
) -> dict[str, str]:
    eval_base = (
        "uv run python benchmarks/eval_glm45_air_teacher_cache.py "
        "--engine vq_e1_routed_nax_e8p "
        f"--artifact-dir {artifact_dir}"
    )
    bench_base = (
        "uv run python benchmarks/bench_glm45_air_quant_compare.py "
        "--scenario prefill_1k --warmup-repetitions 1 --repetitions 3 "
        "--timing-stability-max-relative-spread 0.2 "
        "--memory-quiet-window-seconds 3 --memory-quiet-max-attempts 5 "
        "--require-memory-quiet-preflight --parent-vm-stat-diagnostics"
    )
    def cache_diagnostic_command(cache_limit_gb: int) -> str:
        return (
            "uv run python benchmarks/bench_glm45_air_quant_compare.py "
            "--scenario prefill_1k --warmup-repetitions 1 --repetitions 1 "
            "--timing-stability-max-relative-spread 0.2 "
            "--memory-quiet-window-seconds 3 --memory-quiet-max-attempts 5 "
            f"--require-memory-quiet-preflight --parent-vm-stat-diagnostics --mlx-cache-limit-gb {cache_limit_gb} "
            "--mlx-clear-cache-before-run "
            f"--engine vq_e1_routed_nax_e8p --artifact-dir {artifact_dir} "
            f"--append-jsonl {output_dir}/lane-s-candidate-cache{cache_limit_gb}-diagnostic.jsonl"
        )
    return {
        "audit": (
            "uv run python benchmarks/bench_glm45_air_quant_compare.py "
            "--audit-prefill-compatibility --engine vq_e1_routed_nax_e8p "
            f"--artifact-dir {artifact_dir}"
        ),
        "eval_report": (
            f"{eval_base} --teacher-jsonl {report_teacher_jsonl} "
            f"--append-jsonl {output_dir}/report128.jsonl"
        ),
        "eval_selection": (
            f"{eval_base} --teacher-jsonl {selection_teacher_jsonl} "
            f"--append-jsonl {output_dir}/selection128.jsonl"
        ),
        "eval_holdout": (
            f"{eval_base} --teacher-jsonl {holdout_teacher_jsonl} "
            f"--append-jsonl {output_dir}/holdout128.jsonl"
        ),
        "lane_s_q2_control": (
            f"{bench_base} --engine mlx_q2_routed_g128 "
            f"--append-jsonl {output_dir}/lane-s-q2-control.jsonl"
        ),
        "lane_s_candidate": (
            f"{bench_base} --engine vq_e1_routed_nax_e8p --artifact-dir {artifact_dir} "
            f"--append-jsonl {output_dir}/lane-s-candidate.jsonl"
        ),
        "lane_s_candidate_cache1_diagnostic": (
            cache_diagnostic_command(1)
        ),
        "lane_s_candidate_cache2_diagnostic": (
            cache_diagnostic_command(2)
        ),
        "lane_s_candidate_cache4_diagnostic": (
            cache_diagnostic_command(4)
        ),
    }


def _execute_output_paths(
    args: argparse.Namespace,
    execute_names: Sequence[str],
) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    paths: dict[str, Path] = {}
    for name in execute_names:
        for arg_name, filename in EXECUTE_OUTPUTS.get(name, []):
            path = output_dir / filename
            if arg_name is not None:
                paths[arg_name] = path
    return paths


def _all_execute_output_files(
    args: argparse.Namespace,
    execute_names: Sequence[str],
) -> list[Path]:
    output_dir = Path(args.output_dir)
    paths: list[Path] = []
    for name in execute_names:
        for _, filename in EXECUTE_OUTPUTS.get(name, []):
            path = output_dir / filename
            if path not in paths:
                paths.append(path)
    return paths


def _execute_output_errors(args: argparse.Namespace, execute_names: Sequence[str]) -> list[str]:
    if bool(args.overwrite):
        return []
    errors = []
    for path in _all_execute_output_files(args, execute_names):
        if path.exists():
            errors.append(f"execute output already exists: {path}; pass --overwrite to replace it")
    return errors


def _prepare_execute_outputs(args: argparse.Namespace, execute_names: Sequence[str]) -> None:
    if not bool(args.overwrite):
        return
    for path in _all_execute_output_files(args, execute_names):
        if path.exists():
            path.unlink()


def _args_with_execute_outputs(
    args: argparse.Namespace,
    execute_names: Sequence[str],
) -> argparse.Namespace:
    updated = argparse.Namespace(**vars(args))
    for arg_name, path in _execute_output_paths(args, execute_names).items():
        setattr(updated, arg_name, str(path))
    return updated


def _build_workflow_commands(
    *,
    seed_artifact_dir: str,
    output_dir: str,
    report_teacher_jsonl: str,
    selection_teacher_jsonl: str,
    holdout_teacher_jsonl: str,
    rerun_commands: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    return {
        "environment_preflight": {
            "status": "verified_this_goal",
            "command": "scripts/glm45_air_rc.sh env-preflight",
            "purpose": "Check MLX, mlx-lm, Metal kernel, Hadamard, and safetensors prerequisites.",
        },
        "rc_preflight": {
            "status": "verified_this_goal",
            "command": "scripts/glm45_air_rc.sh preflight",
            "purpose": "Check artifact, seed, evidence, teacher caches, and command entrypoints without loading the model.",
        },
        "host_memory_preflight": {
            "status": "verified_this_goal",
            "command": "scripts/glm45_air_rc.sh memory-preflight",
            "purpose": "Check host pageout/swapout quietness before expensive Lane S benchmark reruns.",
        },
        "rc_summary": {
            "status": "verified_this_goal",
            "command": "scripts/glm45_air_rc.sh summary --overwrite",
            "purpose": "Rebuild the cheap balanced-RC JSON and Markdown summary from recorded evidence.",
        },
        "reproduce_rc_from_seed": {
            "status": "heavy_not_rerun_in_this_slice",
            "command": f"scripts/glm45_air_rc.sh reproduce --overwrite --output-dir {output_dir}",
            "purpose": "Train the public rank-4 low-rank residual recipe from the protected seed, then run full verification on the reproduced artifact.",
        },
        "verify_rc": {
            "status": "heavy_not_rerun_in_this_slice",
            "command": f"scripts/glm45_air_rc.sh verify --overwrite --output-dir {output_dir}",
            "purpose": "Run artifact audit, report/selection/holdout evals, paired Lane S, and rebuild the full RC packet from fresh outputs.",
        },
        "artifact_audit": {
            "status": "available_via_execute_not_rerun_in_this_slice",
            "command": f"scripts/glm45_air_rc.sh audit --overwrite --output-dir {output_dir}",
            "purpose": "Check projection counts, sidecars, precision fallbacks, and prefill fast-path compatibility.",
        },
        "eval_report": {
            "status": "heavy_not_rerun_in_this_slice",
            "command": f"scripts/glm45_air_rc.sh eval-report --overwrite --output-dir {output_dir}",
            "purpose": "Regenerate the report split VQ-vs-BF16 teacher-cache comparison.",
        },
        "eval_selection": {
            "status": "heavy_not_rerun_in_this_slice",
            "command": f"scripts/glm45_air_rc.sh eval-selection --overwrite --output-dir {output_dir}",
            "purpose": "Regenerate the selection split VQ-vs-BF16 teacher-cache comparison.",
        },
        "eval_holdout": {
            "status": "heavy_not_rerun_in_this_slice",
            "command": f"scripts/glm45_air_rc.sh eval-holdout --overwrite --output-dir {output_dir}",
            "purpose": "Regenerate the holdout split VQ-vs-BF16 teacher-cache comparison.",
        },
        "benchmark_lane_s": {
            "status": "heavy_not_rerun_in_this_slice",
            "command": f"scripts/glm45_air_rc.sh benchmark-lane-s --overwrite --output-dir {output_dir}",
            "purpose": "Regenerate q2 and balanced-RC Lane S row sets in one invocation so the summary uses both fresh files.",
        },
        "benchmark_q2_control": {
            "status": "heavy_not_rerun_in_this_slice",
            "command": f"scripts/glm45_air_rc.sh benchmark-q2 --overwrite --output-dir {output_dir}",
            "purpose": "Regenerate only the quiet-window q2 routed control Lane S measurement for diagnosis.",
        },
        "benchmark_candidate": {
            "status": "heavy_not_rerun_in_this_slice",
            "command": f"scripts/glm45_air_rc.sh benchmark-candidate --overwrite --output-dir {output_dir}",
            "purpose": "Regenerate only the quiet-window balanced-RC Lane S measurement for diagnosis.",
        },
        "benchmark_candidate_cache1_diagnostic": {
            "status": "diagnostic_not_acceptance_gate",
            "command": f"scripts/glm45_air_rc.sh benchmark-candidate-cache1 --overwrite --output-dir {output_dir}",
            "purpose": "Run a one-measured-row candidate diagnostic with MLX cache limit 1 GiB and cache clear.",
        },
        "benchmark_candidate_cache_sweep": {
            "status": "diagnostic_not_acceptance_gate",
            "command": f"scripts/glm45_air_rc.sh benchmark-candidate-cache-sweep --overwrite --output-dir {output_dir}",
            "purpose": "Run one-measured-row candidate diagnostics with MLX cache limits 1, 2, and 4 GiB.",
        },
        "collect_calibration_imatrix": {
            "status": "heavy_not_run_requires_resident_source_model",
            "command": "scripts/glm45_air_rc.sh collect-imatrix",
            "purpose": "Collect routed projection calibration/imatrix sidecars for a new VQ materialization.",
        },
        "materialize_dynamic_imatrix_sweep": {
            "status": "heavy_not_run_requires_imatrix_and_reference_artifacts",
            "command": "scripts/glm45_air_rc.sh materialize-sweep",
            "purpose": "Materialize a reproducible dynamic-imatrix Pareto sweep from calibration data.",
        },
        "train_low_rank_residual_sidecars": {
            "status": "heavy_not_rerun_in_this_slice",
            "command": "scripts/glm45_air_rc.sh train-low-rank",
            "purpose": "Recreate the current rank-4 layer-45 low-rank residual sidecar recipe.",
        },
        "new_model_family_template": {
            "status": "verified_this_goal",
            "command": "scripts/glm45_air_rc.sh new-model-template",
            "purpose": "Print the reusable checklist for adding a public VQ workflow for another model family.",
        },
        "inspect_tail_metrics": {
            "status": "verified_this_goal",
            "command": f"scripts/glm45_air_rc.sh focus --overwrite --output-dir {output_dir}",
            "purpose": "Export compact domain and tail-focus evidence from the accepted RC rows.",
        },
        "inspect_token_attribution": {
            "status": "verified_this_goal",
            "command": f"scripts/glm45_air_rc.sh attribute --overwrite --output-dir {output_dir}",
            "purpose": "Export prompt/token attribution JSON for report, selection, and holdout accepted RC rows.",
        },
        "export_quality_frontier": {
            "status": "verified_this_goal",
            "command": f"scripts/glm45_air_rc.sh frontier --overwrite --output-dir {output_dir}",
            "purpose": "Export compact balanced-vs-quality frontier evidence without scraping the full RC summary.",
        },
        "plan_next_quality_slice": {
            "status": "verified_this_goal",
            "command": f"scripts/glm45_air_rc.sh plan --overwrite --output-dir {output_dir}",
            "purpose": "Build a holdout-safe rank-4 route/math quality plan from existing RC evidence.",
        },
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    split_rows = []
    domain_rows = []
    focus_rows = []
    quality_frontier_rows = []
    quality_frontier_speed_rows = []
    lane_s_diagnostic_rows = []
    lane_s_vm_rows = []
    for split_name, split in report["eval_splits"].items():
        split_rows.append(
            "| {name} | {clean}/{total} | {kld:.6f} | {top1:.6f} | {ppl:.6f} | {max_ppl:.6f} | {p999:.6f} | {hard} | {wow} |".format(
                name=split_name,
                clean=int(split.get("clean_record_count") or 0),
                total=int(split.get("record_count") or 0),
                kld=float(split.get("mean_kld") or 0.0),
                top1=float(split.get("mean_top1_agreement") or 0.0),
                ppl=float(split.get("mean_ppl_ratio") or 0.0),
                max_ppl=float(split.get("max_ppl_ratio") or 0.0),
                p999=float(split.get("p999_kld") or 0.0),
                hard="pass" if _all(report["checks"]["split_hard"][split_name]) else "fail",
                wow="pass" if _all(report["checks"]["split_wow"][split_name]) else "miss",
            )
        )
        for domain_name, domain in split.get("domains", {}).items():
            domain_rows.append(
                "| {split} | {domain} | {rows} | {top1:.6f} | {kld:.6f} | {ppl:.6f} | {p999:.6f} | {check} |".format(
                    split=split_name,
                    domain=domain_name,
                    rows=int(domain.get("record_count") or 0),
                    top1=float(domain.get("mean_top1_agreement") or 0.0),
                    kld=float(domain.get("mean_kld") or 0.0),
                    ppl=float(domain.get("mean_ppl_ratio") or 0.0),
                    p999=float(domain.get("p999_kld") or 0.0),
                    check="pass" if domain.get("top1_ge_0p80") else "miss",
                )
            )
        for category, rows in split.get("quality_focus", {}).items():
            for row in rows[:5]:
                focus_rows.append(
                    "| {split} | {category} | {prompt} | {domain} | {top1} | {kld} | {ppl} | {token_kld} | {position} |".format(
                        split=split_name,
                        category=category,
                        prompt=row.get("prompt_id"),
                        domain=row.get("domain"),
                        top1=_format_optional_float(row.get("top1_agreement")),
                        kld=_format_optional_float(row.get("mean_kld")),
                        ppl=_format_optional_float(row.get("ppl_ratio")),
                        token_kld=_format_optional_float(row.get("max_token_kld")),
                        position=row.get("max_token_position"),
                )
            )
    for attempt in report.get("quality_frontier", {}).get("attempts", []):
        for split_name, compare in attempt.get("split_compares", {}).items():
            candidate = compare.get("candidate_summary", {})
            delta = compare.get("summary_delta", {})
            quality_frontier_rows.append(
                "| {candidate_name} | {promotion} | {split} | {clean}/{total} | {kld} | {top1} | {ppl} | {p999} | {delta_kld} | {delta_top1} | {delta_p999} | {decision} |".format(
                    candidate_name=attempt.get("label"),
                    promotion=attempt.get("promotion_status"),
                    split=split_name,
                    clean=int(candidate.get("clean_record_count") or 0),
                    total=int(candidate.get("record_count") or 0),
                    kld=_format_optional_float(candidate.get("mean_kld")),
                    top1=_format_optional_float(candidate.get("mean_top1_agreement")),
                    ppl=_format_optional_float(candidate.get("mean_ppl_ratio")),
                    p999=_format_optional_float(candidate.get("p999_kld")),
                    delta_kld=_format_optional_float(delta.get("mean_kld")),
                    delta_top1=_format_optional_float(delta.get("mean_top1_agreement")),
                    delta_p999=_format_optional_float(delta.get("p999_kld")),
                    decision=compare.get("decision_label"),
                )
            )
        lane_s = attempt.get("lane_s", {})
        quality_frontier_speed_rows.append(
            "| {candidate_name} | {candidate_median} | {q2_median} | {ratio} | {clean} | {ratio_gate} | {decision} |".format(
                candidate_name=attempt.get("label"),
                candidate_median=_format_optional_float(lane_s.get("candidate", {}).get("median_seconds")),
                q2_median=_format_optional_float(lane_s.get("q2_control", {}).get("median_seconds")),
                ratio=_format_optional_float(lane_s.get("ratio")),
                clean=(
                    "pass"
                    if lane_s.get("checks", {}).get("candidate_clean")
                    and lane_s.get("checks", {}).get("q2_control_clean")
                    else "fail"
                ),
                ratio_gate="pass" if lane_s.get("checks", {}).get("ratio_le_1p15") else "fail",
                decision=attempt.get("decision"),
            )
        )
    for label, bench in (
        ("candidate", report["lane_s"]["candidate"]),
        ("q2_control", report["lane_s"]["q2_control"]),
    ):
        attempted = bench.get("attempted_timing") if isinstance(bench, Mapping) else {}
        attempted = attempted if isinstance(attempted, Mapping) else {}
        memory = bench.get("memory_pressure_summary") if isinstance(bench, Mapping) else {}
        memory = memory if isinstance(memory, Mapping) else {}
        lane_s_diagnostic_rows.append(
            "| {label} | {clean}/{total} | {accepted_median} | {accepted_spread} | {attempted_median} | {attempted_spread} | {attempted_seconds} | {timing_stable} | {dirty} | {pageouts} | {swapouts} |".format(
                label=label,
                clean=int(bench.get("clean_record_count") or 0),
                total=int(bench.get("record_count") or 0),
                accepted_median=_format_optional_float(bench.get("median_seconds")),
                accepted_spread=_format_optional_float(bench.get("relative_spread")),
                attempted_median=_format_optional_float(attempted.get("median_seconds")),
                attempted_spread=_format_optional_float(attempted.get("relative_spread")),
                attempted_seconds=_format_timing_seconds(attempted.get("seconds")),
                timing_stable="pass" if bench.get("timing_stable") else "fail",
                dirty=int(memory.get("dirty_record_count") or 0),
                pageouts=int(memory.get("pageouts_delta_sum") or 0),
                swapouts=int(memory.get("swapouts_delta_sum") or 0),
            )
        )
        parent_vm = bench.get("parent_vm_stat_summary") if isinstance(bench, Mapping) else {}
        parent_vm = parent_vm if isinstance(parent_vm, Mapping) else {}
        lane_s_vm_rows.append(
            "| {label} | {available}/{total} | {dirty} | {all_deltas} | {dirty_deltas} |".format(
                label=label,
                available=int(parent_vm.get("available_record_count") or 0),
                total=int(bench.get("record_count") or 0),
                dirty=int(parent_vm.get("dirty_available_record_count") or 0),
                all_deltas=_format_delta_map(parent_vm.get("selected_delta_sums")),
                dirty_deltas=_format_delta_map(parent_vm.get("dirty_selected_delta_sums")),
            )
        )
    commands = "\n".join(
        f"- `{name}`:\n\n  ```bash\n  {command}\n  ```"
        for name, command in report["rerun_commands"].items()
    )
    workflow_sections = []
    for name, step in report["workflow_commands"].items():
        workflow_sections.append(
            "### {name}\n\n- Status: `{status}`\n- Purpose: {purpose}\n\n```bash\n{command}\n```".format(
                name=name,
                status=step["status"],
                purpose=step["purpose"],
                command=step["command"],
            )
        )
    return "\n".join(
        [
            "# GLM-4.5-Air VQ RC Summary",
            "",
            f"- Artifact: `{report['artifact_dir']}`",
            f"- Seed artifact: `{report['seed_artifact_dir']}`",
            f"- Overall status: `{report['overall_status']}`",
            f"- Hard balanced RC gates: `{'pass' if report['checks']['balanced_hard_pass'] else 'fail'}`",
            f"- Community-wow targets: `{'pass' if report['checks']['community_wow_pass'] else 'miss'}`",
            "",
            "## Evidence",
            "",
            "| Split | Clean rows | Mean KLD | Top1 | Mean PPL | Max PPL | p999 KLD | Hard gate | Wow target |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            *split_rows,
            "",
            "## Domain Evidence",
            "",
            "| Split | Domain | Rows | Top1 | Mean KLD | Mean PPL | p999 KLD | Top1 >= 0.80 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            *domain_rows,
            "",
            "## Quality Focus",
            "",
            "| Split | Category | Prompt | Domain | Top1 | Mean KLD | PPL | Max Token KLD | Position |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            *focus_rows,
            "",
            "## Lane S",
            "",
            f"- Candidate median: `{report['lane_s']['candidate'].get('median_seconds')}` seconds",
            f"- Q2 control median: `{report['lane_s']['q2_control'].get('median_seconds')}` seconds",
            f"- Ratio: `{report['lane_s'].get('ratio')}`",
            f"- Effective routed bpw: `{report['lane_s']['candidate'].get('effective_bits_per_weight')}`",
            f"- Dense routed experts: `{report['lane_s']['candidate'].get('dense_expert_params')}`",
            f"- Unbound VQ experts: `{report['lane_s']['candidate'].get('unbound_vq_experts')}`",
            f"- Non-expert dtype verified: `{report['lane_s']['candidate'].get('non_expert_dtype_verified')}`",
            f"- Failure reasons: `{', '.join(report['lane_s'].get('failure_reasons') or []) or 'none'}`",
            "",
            "### Lane S Diagnostics",
            "",
            "| Row set | Clean rows | Accepted median | Accepted spread | Attempted median | Attempted spread | Attempted seconds | Timing stable | Dirty rows | Pageouts | Swapouts |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
            *lane_s_diagnostic_rows,
            "",
            "### Parent VM Diagnostics",
            "",
            "| Row set | Parent VM rows | Dirty rows | Selected delta sums | Dirty selected delta sums |",
            "| --- | ---: | ---: | --- | --- |",
            *lane_s_vm_rows,
            "",
            "## Quality Frontier",
            "",
            f"- Promoted preset: `{report.get('quality_frontier', {}).get('promoted', 'balanced')}`",
            f"- Status: `{report.get('quality_frontier', {}).get('status')}`",
            "",
            "| Candidate | Promotion | Split | Clean rows | Mean KLD | Top1 | Mean PPL | p999 KLD | Delta KLD | Delta Top1 | Delta p999 | Compare |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            *(quality_frontier_rows or ["| no quality frontier evidence available |  |  |  |  |  |  |  |  |  |  |  |"]),
            "",
            "| Candidate | Candidate median | Q2 median | Ratio | Clean rows | Ratio <= 1.15 | Decision |",
            "| --- | ---: | ---: | ---: | --- | --- | --- |",
            *(quality_frontier_speed_rows or ["| no quality frontier speed evidence available |  |  |  |  |  |  |"]),
            "",
            "## Artifact Audit",
            "",
            f"- Projections: `{report['artifact_audit'].get('projection_count')}`",
            f"- Symlinked projections: `{report['artifact_audit'].get('symlink_projection_count')}`",
            f"- High-precision routed projections: `{report['artifact_audit'].get('high_precision_projection_count')}`",
            f"- Continuous sidecars: `{report['artifact_audit'].get('continuous_sidecar_count')}`",
            f"- Auto prefill all layers NAX-fast compatible: `{report['artifact_audit'].get('auto_prefill_all_layers_nax_fast_compatible')}`",
            "",
            "## Rerun Commands",
            "",
            "These are implementation-level commands retained for auditability. Prefer the public workflow commands below for normal RC use.",
            "",
            commands,
            "",
            "## Public Workflow Commands",
            "",
            *workflow_sections,
            "",
        ]
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.artifact_dir)
    report_summary = _summarize_eval_split(Path(args.report_jsonl))
    selection_summary = _summarize_eval_split(Path(args.selection_jsonl))
    holdout_summary = _summarize_eval_split(Path(args.holdout_jsonl))
    candidate_bench = _summarize_benchmark(Path(args.lane_s_jsonl))
    q2_control_bench = _summarize_benchmark(Path(args.q2_control_jsonl))
    q2_median = q2_control_bench["median_seconds"]
    candidate_median = candidate_bench["median_seconds"]
    lane_s_ratio = (
        float(candidate_median) / float(q2_median)
        if candidate_median is not None and q2_median not in (None, 0)
        else None
    )

    eval_splits = {
        "report": report_summary,
        "selection": selection_summary,
        "holdout": holdout_summary,
    }
    split_checks = {name: _quality_checks(summary) for name, summary in eval_splits.items()}
    split_hard = {
        name: {
            "clean_128": checks["clean_128"],
            "mean_ppl_ratio_le_1p05": checks["mean_ppl_ratio_le_1p05"],
        }
        for name, checks in split_checks.items()
    }
    split_wow = {
        name: {
            "mean_kld_le_0p30": checks["mean_kld_le_0p30"],
            "p999_kld_le_3p0": checks["p999_kld_le_3p0"],
            "top1_ge_0p85": checks["top1_ge_0p85"],
            "all_domain_top1_ge_0p80": all(
                bool(domain.get("top1_ge_0p80"))
                for domain in summary.get("domains", {}).values()
            ),
        }
        for name, checks in split_checks.items()
        for summary in [eval_splits[name]]
    }
    artifact_audit = audit_vq_artifact_prefill_compatibility(artifact_dir)
    manifest = _load_manifest(artifact_dir)
    lane_s_checks = {
        "candidate_clean": bool(candidate_bench["all_memory_clean"]),
        "q2_control_clean": bool(q2_control_bench["all_memory_clean"]),
        "candidate_timing_stable": bool(candidate_bench.get("timing_stable")),
        "q2_control_timing_stable": bool(q2_control_bench.get("timing_stable")),
        "ratio_le_1p15": _check_le(lane_s_ratio, BALANCED_HARD_TARGETS["lane_s_ratio_max"]),
        "effective_bpw_le_2p1": _check_le(
            candidate_bench.get("effective_bits_per_weight"),
            BALANCED_HARD_TARGETS["effective_bpw_max"],
        ),
        "no_dense_routed_experts": candidate_bench.get("dense_expert_params") is False,
        "no_unbound_vq_experts": candidate_bench.get("unbound_vq_experts") is False,
        "non_expert_dtype_verified": candidate_bench.get("non_expert_dtype_verified") is True,
    }
    artifact_checks = {
        "artifact_exists": artifact_dir.exists(),
        "manifest_exists": bool(manifest.get("exists")),
        "seed_artifact_recorded": bool(manifest.get("seed_artifact_dir") or args.seed_artifact_dir),
        "all_projection_shards_present": artifact_audit.get("projection_count") == 135,
        "no_high_precision_routed_projections": artifact_audit.get("high_precision_projection_count") == 0,
        "no_metal_fallback_layers": artifact_audit.get("metal_fallback_layer_count") == 0,
        "has_continuous_sidecars": int(artifact_audit.get("continuous_sidecar_count") or 0) > 0,
    }
    balanced_hard_pass = (
        all(_all(checks) for checks in split_hard.values())
        and _all(lane_s_checks)
        and _all(artifact_checks)
    )
    community_wow_pass = all(_all(checks) for checks in split_wow.values())
    status = "balanced_rc_pass_community_wow_miss"
    if community_wow_pass and balanced_hard_pass:
        status = "balanced_rc_pass_community_wow_pass"
    elif not balanced_hard_pass:
        status = "balanced_rc_fail"

    output_dir = str(Path(args.output_dir))
    rerun_commands = _build_rerun_commands(
        artifact_dir=str(artifact_dir),
        output_dir=output_dir,
        report_teacher_jsonl=args.report_teacher_jsonl,
        selection_teacher_jsonl=args.selection_teacher_jsonl,
        holdout_teacher_jsonl=args.holdout_teacher_jsonl,
    )
    return {
        "schema_version": 1,
        "artifact_dir": str(artifact_dir),
        "seed_artifact_dir": manifest.get("seed_artifact_dir") or args.seed_artifact_dir,
        "overall_status": status,
        "targets": {
            "balanced_hard": BALANCED_HARD_TARGETS,
            "community_wow": COMMUNITY_WOW_TARGETS,
        },
        "manifest": manifest,
        "artifact_audit": {
            key: value
            for key, value in artifact_audit.items()
            if key not in {"projection_rows", "layer_rows"}
        },
        "eval_splits": eval_splits,
        "lane_s": {
            "candidate": candidate_bench,
            "q2_control": q2_control_bench,
            "ratio": lane_s_ratio,
            "failure_reasons": _lane_s_failure_reasons(lane_s_checks),
        },
        "quality_frontier": _build_quality_frontier(),
        "evidence_sources": {
            "report_jsonl": args.report_jsonl,
            "selection_jsonl": args.selection_jsonl,
            "holdout_jsonl": args.holdout_jsonl,
            "lane_s_jsonl": args.lane_s_jsonl,
            "q2_control_jsonl": args.q2_control_jsonl,
        },
        "checks": {
            "split_hard": split_hard,
            "split_wow": split_wow,
            "lane_s": lane_s_checks,
            "artifact": artifact_checks,
            "balanced_hard_pass": balanced_hard_pass,
            "community_wow_pass": community_wow_pass,
        },
        "rerun_commands": rerun_commands,
        "workflow_commands": _build_workflow_commands(
            seed_artifact_dir=str(manifest.get("seed_artifact_dir") or args.seed_artifact_dir),
            output_dir=output_dir,
            report_teacher_jsonl=args.report_teacher_jsonl,
            selection_teacher_jsonl=args.selection_teacher_jsonl,
            holdout_teacher_jsonl=args.holdout_teacher_jsonl,
            rerun_commands=rerun_commands,
        ),
    }


def _write_report(report: Mapping[str, Any], output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rc-summary.json"
    markdown_path = output_dir / "RC_SUMMARY.md"
    for path in (json_path, markdown_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_render_markdown(report))


def _build_focus_export(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_dir": report["artifact_dir"],
        "overall_status": report["overall_status"],
        "targets": {
            "domain_top1_min": report["targets"]["community_wow"]["domain_top1_min"],
            "mean_kld_max": report["targets"]["community_wow"]["mean_kld_max"],
            "p999_kld_max": report["targets"]["community_wow"]["p999_kld_max"],
            "top1_min": report["targets"]["community_wow"]["top1_min"],
        },
        "splits": {
            split_name: {
                "domains": split.get("domains", {}),
                "quality_focus": split.get("quality_focus", {}),
            }
            for split_name, split in report["eval_splits"].items()
        },
    }


def _build_quality_frontier_export(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_dir": report["artifact_dir"],
        "overall_status": report["overall_status"],
        "promoted": report["quality_frontier"]["promoted"],
        "targets": {
            "balanced_hard": report["targets"]["balanced_hard"],
            "community_wow": report["targets"]["community_wow"],
        },
        "quality_frontier": report["quality_frontier"],
    }


def _build_quality_train_command(
    *,
    args: argparse.Namespace,
    report: Mapping[str, Any],
    row_indices_csv: str,
) -> str:
    selection_cache_root = str(Path(args.selection_teacher_jsonl).parent)
    validation_cache_root = str(Path(args.report_teacher_jsonl).parent)
    output_dir = "artifacts/glm45-air-quality-r4-route-math-plan1"
    train_log = "artifacts/quality/glm45-air-quality-r4-route-math-plan1.jsonl"
    return (
        "uv run python benchmarks/finetune_glm45_air_vq_continuous.py "
        f"--selection-teacher-jsonl {args.selection_teacher_jsonl} "
        f"--selection-cache-root {selection_cache_root} "
        f"--validation-teacher-jsonl {args.report_teacher_jsonl} "
        f"--validation-cache-root {validation_cache_root} "
        f"--seed-artifact-dir {report['artifact_dir']} "
        f"--output-dir {output_dir} "
        "--prefill-engine nax_e8p "
        "--layer 45 "
        "--projections gate_proj up_proj down_proj "
        "--trainable low_rank_residual "
        "--low-rank 4 "
        "--low-rank-init-scale 0.05 "
        "--steps 12 "
        "--learning-rate 0.5 "
        "--target-nll-weight 0.5 "
        "--teacher-top1-margin-weight 2 "
        "--teacher-top1-margin 1 "
        "--loss-scope final_layer_selected "
        "--surrogate-projections gate_proj up_proj down_proj "
        "--surrogate-output-chunk-size 256 "
        "--train-cache both "
        "--max-train-rows 8 "
        f"--train-row-indices {row_indices_csv} "
        "--max-positions 16 "
        f"--append-jsonl {train_log}"
    )


def _build_quality_plan(report: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    train_domains = set(QUALITY_PLAN_DOMAIN_QUOTAS)
    train_candidates = _quality_plan_candidates(
        report,
        split_names=("report", "selection"),
        allowed_domains=train_domains,
    )
    train_rows = _select_quality_plan_rows(
        train_candidates,
        quotas=QUALITY_PLAN_DOMAIN_QUOTAS,
    )
    row_indices = [int(row["row_index"]) for row in train_rows]
    row_indices_csv = ",".join(str(index) for index in row_indices)
    holdout_focus = _quality_plan_candidates(
        report,
        split_names=("holdout",),
        allowed_domains=train_domains,
    )[:8]
    train_command = _build_quality_train_command(
        args=args,
        report=report,
        row_indices_csv=row_indices_csv,
    )
    output_artifact = "artifacts/glm45-air-quality-r4-route-math-plan1"
    eval_command_base = (
        "uv run python benchmarks/eval_glm45_air_teacher_cache.py "
        "--engine vq_e1_routed_nax_e8p "
        f"--artifact-dir {output_artifact}"
    )
    return {
        "schema_version": 1,
        "source_summary": str(Path(args.output_dir) / "rc-summary.json"),
        "artifact_dir": report["artifact_dir"],
        "seed_artifact_dir": report["seed_artifact_dir"],
        "overall_status": report["overall_status"],
        "policy": {
            "purpose": "Bound the next quality run toward route/math top1 and token-tail misses while preserving the accepted rank-4 recipe.",
            "training_splits": ["report", "selection"],
            "holdout_usage": "validation_focus_only_not_training",
            "domain_quotas": QUALITY_PLAN_DOMAIN_QUOTAS,
            "target_trainable": "low_rank_residual",
            "low_rank": 4,
            "loss_scope": "final_layer_selected",
            "surrogate_projections": ["gate_proj", "up_proj", "down_proj"],
            "forbidden_changes": [
                "do_not_train_on_holdout",
                "do_not_return_to_rank_8_without_runtime_change",
                "do_not_accept_dirty_eval_rows",
                "do_not_promote_if_lane_s_or_mean_ppl_regresses_materially",
            ],
        },
        "train_row_indices": row_indices,
        "train_row_indices_csv": row_indices_csv,
        "train_rows": train_rows,
        "holdout_validation_focus": holdout_focus,
        "commands": {
            "train_quality_sidecar": {
                "status": "proposed_heavy_not_run_in_this_slice",
                "command": train_command,
                "purpose": "Train a bounded rank-4 layer45 gate/up/down low-rank residual from report/selection focus rows.",
            },
            "eval_report": {
                "status": "run_after_training_only",
                "command": (
                    f"{eval_command_base} --teacher-jsonl {args.report_teacher_jsonl} "
                    " --append-jsonl artifacts/quality/glm45-air-quality-r4-route-math-plan1-report128.jsonl"
                ),
                "purpose": "Regenerate report split evidence for the proposed quality candidate.",
            },
            "eval_selection": {
                "status": "run_after_training_only",
                "command": (
                    f"{eval_command_base} --teacher-jsonl {args.selection_teacher_jsonl} "
                    " --append-jsonl artifacts/quality/glm45-air-quality-r4-route-math-plan1-select128.jsonl"
                ),
                "purpose": "Regenerate selection split evidence for the proposed quality candidate.",
            },
            "eval_holdout": {
                "status": "run_after_training_only",
                "command": (
                    f"{eval_command_base} --teacher-jsonl {args.holdout_teacher_jsonl} "
                    " --append-jsonl artifacts/quality/glm45-air-quality-r4-route-math-plan1-holdout128.jsonl"
                ),
                "purpose": "Use holdout only after training to detect leakage or overfitting.",
            },
        },
    }


def _write_focus_export(report: Mapping[str, Any], path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_build_focus_export(report), indent=2, sort_keys=True) + "\n")


def _write_quality_frontier_export(report: Mapping[str, Any], path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_build_quality_frontier_export(report), indent=2, sort_keys=True) + "\n")


def _write_quality_plan(report: Mapping[str, Any], args: argparse.Namespace, path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_build_quality_plan(report, args), indent=2, sort_keys=True) + "\n")


def _run_command(command: str) -> None:
    subprocess.run(command, shell=True, check=True)


def _assert_execute_engine_tier_compatible(
    args: argparse.Namespace,
    execute_names: Sequence[str],
) -> None:
    for name in execute_names:
        engine = EXECUTE_ARTIFACT_ENGINE_CHECKS.get(name)
        if engine is None:
            continue
        assert_engine_tier_compatible(args.artifact_dir, engine)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reproducible GLM-4.5-Air VQ RC evidence summary and command packet."
    )
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--seed-artifact-dir", default=DEFAULT_SEED_ARTIFACT_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--source-dir",
        help="Optional local source model snapshot. Defaults to Hugging Face cache with local_files_only.",
    )
    parser.add_argument("--report-jsonl", default=DEFAULT_REPORT_JSONL)
    parser.add_argument("--selection-jsonl", default=DEFAULT_SELECTION_JSONL)
    parser.add_argument("--holdout-jsonl", default=DEFAULT_HOLDOUT_JSONL)
    parser.add_argument("--lane-s-jsonl", default=DEFAULT_LANE_S_JSONL)
    parser.add_argument("--q2-control-jsonl", default=DEFAULT_Q2_CONTROL_JSONL)
    parser.add_argument("--report-teacher-jsonl", default=DEFAULT_TEACHER_JSONLS["report"])
    parser.add_argument("--selection-teacher-jsonl", default=DEFAULT_TEACHER_JSONLS["selection"])
    parser.add_argument("--holdout-teacher-jsonl", default=DEFAULT_TEACHER_JSONLS["holdout"])
    parser.add_argument("--output-dir", default="artifacts/rc/glm45-air-balanced-r4-20260701")
    parser.add_argument(
        "--write-focus-json",
        help="Write a compact domain and quality-focus JSON export to this path.",
    )
    parser.add_argument(
        "--write-quality-plan-json",
        help="Write a bounded next-quality-slice plan JSON from existing report/selection/holdout evidence.",
    )
    parser.add_argument(
        "--write-quality-frontier-json",
        help="Write a compact balanced-vs-quality frontier JSON export to this path.",
    )
    parser.add_argument(
        "--execute",
        action="append",
        choices=[
            "audit",
            "eval_report",
            "eval_selection",
            "eval_holdout",
            "lane_s_q2_control",
            "lane_s_candidate",
            "lane_s_candidate_cache1_diagnostic",
            "lane_s_candidate_cache2_diagnostic",
            "lane_s_candidate_cache4_diagnostic",
        ],
        help="Run one generated command before summarizing. Repeat for multiple commands.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check local RC inputs and command entrypoints without loading the model or writing summary outputs.",
    )
    parser.add_argument(
        "--preflight-format",
        choices=["json", "markdown"],
        default="json",
        help="Output format for --preflight.",
    )
    parser.add_argument(
        "--strict-impressive",
        action="store_true",
        help="Exit nonzero when community-wow KLD/top1/tail targets are missed.",
    )
    args = parser.parse_args()

    if args.preflight:
        preflight = _build_preflight(args)
        if args.preflight_format == "markdown":
            print(_render_preflight_markdown(preflight))
        else:
            print(json.dumps(preflight, indent=2, sort_keys=True))
        if preflight["status"] != "pass":
            raise SystemExit(1)
        return

    execute_names = list(args.execute or [])
    errors = _summary_cli_errors(args)
    errors.extend(_execute_output_errors(args, execute_names))
    if errors:
        parser.error("RC summary inputs are not ready:\n  - " + "\n  - ".join(errors))

    _prepare_execute_outputs(args, execute_names)
    _assert_execute_engine_tier_compatible(args, execute_names)
    probe_report = build_report(args)
    for name in execute_names:
        _run_command(probe_report["rerun_commands"][name])
    summary_args = _args_with_execute_outputs(args, execute_names)
    report = build_report(summary_args)
    _write_report(report, Path(args.output_dir), overwrite=args.overwrite)
    if args.write_focus_json:
        _write_focus_export(report, Path(args.write_focus_json), overwrite=args.overwrite)
    if args.write_quality_frontier_json:
        _write_quality_frontier_export(report, Path(args.write_quality_frontier_json), overwrite=args.overwrite)
    if args.write_quality_plan_json:
        _write_quality_plan(report, summary_args, Path(args.write_quality_plan_json), overwrite=args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["checks"]["balanced_hard_pass"]:
        raise SystemExit(1)
    if args.strict_impressive and not report["checks"]["community_wow_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
